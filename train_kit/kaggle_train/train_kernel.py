# RSNA knee -- train one Raptor-geometry model (96-slice volume, 3-slice windows, per-finding
# attention pooling) on the cached train shards. Labels: mean(pilkwang report_labels_v2,
# stevenleehans v4_blend); where stevenleehans says "not addressed" (0.25) pilkwang alone.
# Gold-58 is held out for monitoring. Output checkpoint loads with the r224 inference script.
import os, glob, time, json, math, random, gc, ctypes
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import timm
from sklearn.metrics import roc_auc_score

ARCH = "__ARCH__"
TAG = "__TAG__"
REQUIRE_CHECKPOINT = False
USE_DATA_PARALLEL = True
LR_BB = __LR_BB__
LR_HEAD = 1.95e-3
WD = 0.0588
RES = 224
EPOCHS = 16
WIN = 12            # random windows per study per step (Raptor recipe)
BS = 4              # studies per step
ACCUM = 4           # -> 16 studies per optimizer step
WARM = 0.15
CLIP = 3.0
SWA_LAST = 4
K_EVAL = 94
TRAIN_BUDGET = 90 * 60
RAM_GUARD_BYTES = 22 * 2**30
CONTINUE_EPOCHS = 0    # > 0: extend a COMPLETE run by this many epochs (fresh warm-up + cosine at CONT_LR_SCALE x the base LRs, new SWA)
CONT_LR_SCALE = 0.3
MIRROR_P = 0.0         # > 0: training-time anatomical mirror probability per study
PREDICT_HELDOUT = False   # True: at the end also predict every cached non-gold study that is not trained on ({TAG}_heldout.csv)
SEED = 42
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OUT = "/kaggle/working"
T0 = time.time()
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
_libc = ctypes.CDLL("libc.so.6")
_libc.malloc_trim.argtypes = [ctypes.c_size_t]
_libc.malloc_trim.restype = ctypes.c_int


def release_cpu_memory():
    gc.collect()
    _libc.malloc_trim(0)


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def log_memory():
    import resource
    memory = {"peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    status = Path("/proc/self/status").read_text()
    for line in status.splitlines():
        if line.startswith(("VmRSS:", "RssAnon:", "RssFile:")):
            name, amount, _ = line.split()
            memory[name.rstrip(":") + "_mb"] = int(amount) / 1024
    for name in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"):
        path = Path(name)
        if path.exists():
            memory[path.name] = path.read_text().strip()
    for i in range(torch.cuda.device_count()):
        memory[f"gpu{i}_allocated_mb"] = torch.cuda.memory_allocated(i) / 2**20
        memory[f"gpu{i}_reserved_mb"] = torch.cuda.memory_reserved(i) / 2**20
    log("MEMORY", json.dumps(memory))
    stat = Path("/sys/fs/cgroup/memory.stat")
    if stat.exists():
        values = dict(line.split() for line in stat.read_text().splitlines())
        log("CGROUP_BREAKDOWN", {k: int(values[k]) for k in ("anon", "file", "kernel") if k in values})


def ram_guard_reached():
    status = Path("/proc/self/status").read_text()
    rss = next(int(line.split()[1]) * 1024 for line in status.splitlines()
               if line.startswith("VmRSS:"))
    return rss >= RAM_GUARD_BYTES


def shallow_find(name):
    pats = ["/kaggle/input/*/" + name, "/kaggle/input/*/*/" + name, "/kaggle/input/*/*/*/" + name,
            "/kaggle/input/*/*/*/*/" + name]
    hits = []
    for p in pats:
        hits += [h for h in glob.glob(p) if "abnormality-detection" not in h]
    return sorted(set(hits))


# ---------------- data ----------------
shards = []
for vp in shallow_find("vols.npy"):
    d = os.path.dirname(vp)
    ids = open(d + "/ids.txt").read().split("\n")
    shards.append(dict(dir=d, ids=ids, masks=np.load(d + "/masks.npy")))
    log("shard", d, len(ids))
assert shards, "no cache shards attached"
index = {sid: (s, r) for s, sh in enumerate(shards) for r, sid in enumerate(sh["ids"])}

root = next(d for d in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
                        "/kaggle/input/rsna-knee-abnormality-detection"] if os.path.exists(d + "/train.csv"))
tr = pd.read_csv(root + "/train.csv")
gold = tr[tr[LAB].notna().any(axis=1)].set_index("StudyInstanceUID")[LAB].astype(float)
st = pd.read_csv(shallow_find("llm_labels_v4_blend.csv")[0]).set_index("StudyInstanceUID")[LAB]
pk = pd.read_csv(shallow_find("report_labels_v2.csv")[0]).set_index("StudyInstanceUID")[LAB]
st, pk = st.align(pk, join="outer")
tgt = pd.DataFrame(np.where(np.isclose(st.values, 0.25) | st.isna().values, pk.values,
                            np.where(pk.isna().values, st.values, (st.values + pk.values) / 2)),
                   index=st.index, columns=LAB).dropna()
TARGETS_FILE = ""      # non-empty: a ready target table (StudyInstanceUID + the 12 labels) replaces the blend above
if TARGETS_FILE:
    tgt = pd.read_csv(shallow_find(TARGETS_FILE)[0]).set_index("StudyInstanceUID")[LAB].astype(float).dropna()
    log("TARGETS_FILE", TARGETS_FILE, len(tgt))
train_ids =[s for s in tgt.index if s in index and s not in gold.index]
gold_ids = [s for s in gold.index if s in index]
log(f"train {len(train_ids)} | gold {len(gold_ids)} | target mean {tgt.loc[train_ids].values.mean(0).round(3)}")


def valid_centers(mask):
    D = len(mask)
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    return cs or [max(1, min((lo + hi) // 2, D - 2))]


def eval_centers(mask, k):
    cs = valid_centers(mask)
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [max(1, min(cs[i], len(mask) - 2)) for i in idx]


def read_windows(s, r, centers):
    shard = shards[s]
    path = shard["dir"] + "/vols.npy"
    if "layout" not in shard:
        vol = np.load(path, mmap_mode="r")
        shard["layout"] = (vol.offset, vol.shape, vol.dtype)
        vol._mmap.close()
    offset, shape, dtype = shard["layout"]
    assert dtype == np.uint8 and len(shape) == 4
    centers = np.asarray(centers)
    slices = np.stack([centers - 1, centers, centers + 1], 1)
    result = np.empty((*slices.shape, *shape[-2:]), dtype=dtype)
    plane_bytes = shape[-2] * shape[-1]
    with open(path, "rb", buffering=0) as stream:
        for i, plane in enumerate(slices.ravel()):
            stream.seek(offset + (r * shape[1] + int(plane)) * plane_bytes)
            target = memoryview(result.reshape(-1, *shape[-2:])[i]).cast("B")
            while target:
                count = stream.readinto(target)
                if not count:
                    raise EOFError(f"truncated cache: {path}, study {r}, plane {plane}")
                target = target[count:]
    return result


SLOT_SIZES = [26, 22, 18, 12, 18]     # the 96-slice layout (cache_kernel SLOTS); slots 0-1 are sagittal


def mirror_windows(x, centers):
    """nartaa's 0.949 notebook _mirror_tri on uint8 (k,3,H,W) triplets: the slot of each window's centre slice decides the axis -
    sagittal: reverse the 3 slice channels; coronal / axial: flip the width."""
    b = np.cumsum([0] + SLOT_SIZES)
    slot = np.searchsorted(b, np.asarray(centers), side="right") - 1
    sag = np.isin(slot, (0, 1))
    out = x[:, :, :, ::-1].copy()
    out[sag] = x[sag][:, ::-1]
    return np.ascontiguousarray(out)


class TrainDS(torch.utils.data.Dataset):
    def __init__(self, ids):
        self.ids = ids
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]; s, r = index[sid]
        cs = valid_centers(shards[s]["masks"][r])
        rng = np.random.default_rng(SEED + self.epoch * len(self.ids) + i)
        pick = np.sort(rng.choice(cs, WIN, replace=len(cs) < WIN))
        x = read_windows(s, r, pick)
        if MIRROR_P > 0 and rng.random() < MIRROR_P:   # anatomical L/R mirror of the whole study (no label swap: left and right knees both occur)
            x = mirror_windows(x, pick)
        return torch.from_numpy(x), torch.tensor(tgt.loc[sid].values, dtype=torch.float32)


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def gpu_aug(x):
    """x: (B,K,3,H,W) uint8 on GPU -> (B,K,3,RES,RES) normalised float. One affine + intensity per study."""
    B, K = x.shape[:2]
    x = x.float().div_(255.0)
    ang = (torch.rand(B, device=x.device) * 2 - 1) * math.radians(10)
    sc = 1.0 / (0.9 + torch.rand(B, device=x.device) * 0.2)
    tx = (torch.rand(B, device=x.device) * 2 - 1) * 0.06
    ty = (torch.rand(B, device=x.device) * 2 - 1) * 0.06
    th = torch.stack([torch.stack([sc * torch.cos(ang), -sc * torch.sin(ang), tx], 1),
                      torch.stack([sc * torch.sin(ang), sc * torch.cos(ang), ty], 1)], 1)
    th = th.repeat_interleave(K, 0)
    xf = x.flatten(0, 1)
    grid = F.affine_grid(th, (B * K, 3, RES, RES), align_corners=False)
    xf = F.grid_sample(xf, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    a = (0.9 + torch.rand(B, device=x.device) * 0.2).repeat_interleave(K).view(-1, 1, 1, 1)
    b = ((torch.rand(B, device=x.device) * 2 - 1) * 0.05).repeat_interleave(K).view(-1, 1, 1, 1)
    xf = (xf * a + b).clamp_(0, 1)
    xf = (xf - MEAN.to(x.device)) / STD.to(x.device)
    return xf.view(B, K, 3, RES, RES)


def eval_input(s, r, mask, dev):
    cs = np.array(eval_centers(mask, K_EVAL))
    tri = torch.from_numpy(read_windows(s, r, cs)).to(dev)
    x = tri.float() / 255.0
    x = F.interpolate(x, size=(RES, RES), mode="bilinear", align_corners=False)
    x = (x - MEAN.to(dev)) / STD.to(dev)
    return x.unsqueeze(0)


# ---------------- model (verbatim structure from the Raptor inference script) ----------------
def build_backbone(arch, pretrained=False):
    hybrid = arch.startswith(("maxvit", "maxxvit", "coatnet", "coat_", "convnext"))
    is_vit = (not hybrid) and any(k in arch for k in ("vit", "deit", "dinov2", "eva", "beit"))
    kw = dict(pretrained=pretrained, num_classes=0, in_chans=3)
    kw.update(global_pool="token", dynamic_img_size=True) if is_vit else kw.update(global_pool="avg")
    return timm.create_model(arch, **kw)


class RaptorClassifier(nn.Module):
    def __init__(self, backbone, F_dim=768, n=12, drop=0.2):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.LayerNorm(F_dim)
        self.att = nn.Sequential(nn.Linear(F_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
        self.clsW = nn.Parameter(torch.zeros(n, F_dim))
        self.clsb = nn.Parameter(torch.zeros(n))
        nn.init.trunc_normal_(self.clsW, std=0.02)
        self.n = n

    def encode(self, x):
        B, K = x.shape[:2]
        return self.backbone(x.flatten(0, 1)).view(B, K, -1)

    def head(self, feats):
        h = self.norm(feats)
        a = torch.softmax(self.att(h), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h)
        return (pooled * self.clsW).sum(-1) + self.clsb

    def forward(self, x):
        return self.head(self.encode(x))


dev = "cuda"
resume_paths = shallow_find(f"{TAG}_resume.pt")
warm_paths = shallow_find(f"{TAG}_partial.pt") if not resume_paths else []
assert len(resume_paths) <= 1 and len(warm_paths) <= 1, "ambiguous checkpoint inputs"
assert not REQUIRE_CHECKPOINT or resume_paths or warm_paths, "required checkpoint is missing"
bb = build_backbone(ARCH, pretrained=not (resume_paths or warm_paths))
model = RaptorClassifier(bb, F_dim=bb.num_features).to(dev)
ngpu = torch.cuda.device_count()
net = nn.DataParallel(model) if ngpu > 1 and USE_DATA_PARALLEL else model
log(f"arch {ARCH} | feat {bb.num_features} | gpus {ngpu} | params {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
log("execution", "DataParallel" if isinstance(net, nn.DataParallel) else "single GPU",
    "| single-GPU resume changes BatchNorm and dropout sampling, not optimizer progress")

bb_params = list(model.backbone.parameters())
bb_ids = {id(p) for p in bb_params}
head_params = [p for p in model.parameters() if id(p) not in bb_ids]
opt = torch.optim.AdamW([{"params": bb_params, "lr": LR_BB}, {"params": head_params, "lr": LR_HEAD}],
                        weight_decay=WD)
base_lrs = [LR_BB, LR_HEAD]
scaler = torch.cuda.amp.GradScaler()
dataset = TrainDS(train_ids)
batches_per_epoch = len(dataset) // BS
steps_per_epoch = math.ceil(batches_per_epoch / ACCUM)


def save_weights(path):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    temporary = path + ".tmp"
    torch.save({"model": state, "arch": ARCH, "res": RES}, temporary)
    os.replace(temporary, path)


def set_lr(step, total):
    scale = 1.0
    if cont_start is not None:   # continuation phase: its own warm-up + cosine over the added epochs
        step -= cont_start * steps_per_epoch
        total = (total_epochs - cont_start) * steps_per_epoch
        scale = CONT_LR_SCALE
    p = min(1.0, max(0, step) / max(1, total))
    f = p / WARM if p < WARM else 0.5 * (1 + math.cos(math.pi * (p - WARM) / (1 - WARM)))
    for g, b in zip(opt.param_groups, base_lrs):
        g["lr"] = b * scale * max(f, 0.0)


@torch.no_grad()
def predict(m, ids):
    m.eval()
    out = np.zeros((len(ids), len(LAB)), np.float32)
    for j, sid in enumerate(ids):
        s, r = index[sid]
        with torch.autocast("cuda", dtype=torch.float16):
            x = eval_input(s, r, shards[s]["masks"][r], dev)
            feats = torch.cat([m.encode(chunk) for chunk in x.split(16, dim=1)], dim=1)
            out[j] = torch.sigmoid(m.head(feats).float())[0].cpu().numpy()
    return out


def gold_auc(p):
    y = gold.loc[gold_ids].values; aucs = []
    for c in range(len(LAB)):
        if 0 < y[:, c].sum() < len(y):
            aucs.append(roc_auc_score(y[:, c], p[:, c]))
    return float(np.mean(aucs)), [round(a, 3) for a in aucs]


def bn_refresh(m, n=256):
    bns = [x for x in m.modules() if isinstance(x, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    for x in bns:
        x.reset_running_stats(); x.momentum = None
    m.train(); k = 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for xb, _ in dl:
            m(gpu_aug(xb.to(dev, non_blocking=True))); k += xb.shape[0]
            if k >= n:
                break
    log(f"bn refreshed on {k} studies")


history = []; swa = None; n_swa = 0
total_epochs = EPOCHS; cont_start = None
step = 0; ep = 0; next_batch = 0
epoch_loss = 0.0; epoch_batches = 0


def cpu_state(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_state(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_state(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_state(v) for v in value)
    return value


def save_resume():
    state = dict(
        model=cpu_state(model.state_dict()), optimizer=cpu_state(opt.state_dict()),
        scaler=scaler.state_dict(), arch=ARCH, res=RES, epoch=ep, next_batch=next_batch,
        step=step, history=history, swa=swa, n_swa=n_swa,
        epoch_loss=epoch_loss, epoch_batches=epoch_batches, train_ids=train_ids,
        epochs=EPOCHS, bs=BS, accum=ACCUM, total_epochs=total_epochs, cont_start=cont_start,
        rng_python=random.getstate(), rng_numpy=np.random.get_state(),
        rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(),
    )
    path = f"{OUT}/{TAG}_resume.pt"
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    log("RESUME_SAVED", f"epoch={ep} next_batch={next_batch} step={step}")


if resume_paths:
    saved = torch.load(resume_paths[0], map_location="cpu", weights_only=False)
    assert saved["arch"] == ARCH and saved["res"] == RES
    assert saved["train_ids"] == train_ids, "training data changed"
    assert (saved["epochs"], saved["bs"], saved["accum"]) == (EPOCHS, BS, ACCUM)
    model.load_state_dict(saved["model"])
    opt.load_state_dict(saved["optimizer"]); scaler.load_state_dict(saved["scaler"])
    ep = saved["epoch"]; next_batch = saved["next_batch"]; step = saved["step"]
    history = saved["history"]; swa = saved["swa"]; n_swa = saved["n_swa"]
    epoch_loss = saved["epoch_loss"]; epoch_batches = saved["epoch_batches"]
    total_epochs = saved.get("total_epochs", EPOCHS); cont_start = saved.get("cont_start")
    random.setstate(saved["rng_python"]); np.random.set_state(saved["rng_numpy"])
    torch.set_rng_state(saved["rng_torch"]); torch.cuda.set_rng_state_all(saved["rng_cuda"])
    del saved
    log("FULL_RESUME", ep, next_batch, step)
    if CONTINUE_EPOCHS and cont_start is None:
        assert ep == total_epochs and next_batch == 0, "continuation needs a COMPLETE run"
        cont_start = ep; total_epochs = ep + CONTINUE_EPOCHS
        swa = None; n_swa = 0            # the new SWA averages the last SWA_LAST epochs of the continuation
        log("CONTINUE", f"epochs {cont_start}..{total_epochs - 1} at {CONT_LR_SCALE} x base LR")
elif warm_paths:
    saved = torch.load(warm_paths[0], map_location="cpu", weights_only=True)
    assert saved["arch"] == ARCH and saved["res"] == RES
    model.load_state_dict(saved["model"])
    del saved
    log("WARM_START: recovered weights only; optimizer and schedule restart, not exact resume")

save_resume()
log_memory()
paused = False
while ep < total_epochs:
    dataset.epoch = ep
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(SEED + ep)).tolist()
    batches = [order[i * BS:(i + 1) * BS] for i in range(batches_per_epoch)]
    dl = torch.utils.data.DataLoader(
        dataset, batch_sampler=batches[next_batch:], num_workers=0, pin_memory=False,
        generator=torch.Generator().manual_seed(SEED + ep),
    )
    start_batch = next_batch
    net.train(); te = time.time(); run = epoch_loss; nb = epoch_batches
    opt.zero_grad(set_to_none=True)
    for bi, (xb, yb) in enumerate(dl, start=start_batch):
        x = gpu_aug(xb.to(dev, non_blocking=True)); y = yb.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = net(x)
        group_size = min(ACCUM, batches_per_epoch - (bi // ACCUM) * ACCUM)
        loss = F.binary_cross_entropy_with_logits(logits.float(), y) / group_size
        scaler.scale(loss).backward()
        run += loss.item() * group_size; nb += 1
        boundary = (bi + 1) % ACCUM == 0 or bi + 1 == batches_per_epoch
        if boundary:
            set_lr(step, total_epochs * steps_per_epoch)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
            next_batch = bi + 1
            epoch_loss = run; epoch_batches = nb
            if step % 25 == 0:
                log(f"  ep {ep} batch {bi}/{batches_per_epoch} loss {run/nb:.4f}")
                save_resume()
                release_cpu_memory()
                log_memory()
            memory_stop = ram_guard_reached() if step % 25 == 0 else False
            if (time.time() - T0 >= TRAIN_BUDGET or memory_stop) and next_batch < batches_per_epoch:
                paused = True
                log("SESSION_PAUSE", "RAM guard" if memory_stop else "time budget")
                save_resume()
                break
    if paused:
        break
    ep_time = time.time() - te
    save_weights(f"{OUT}/{TAG}_last.pt")
    pg = predict(model, gold_ids); auc, per = gold_auc(pg)
    history.append(dict(epoch=ep, loss=run / nb, gold_auc=auc, per=per, sec=ep_time))
    log(f"EPOCH {ep} loss {run/nb:.4f} | gold AUC {auc:.4f} {per} | {ep_time:.0f}s")
    if ep >= total_epochs - SWA_LAST:
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if swa is None:
            swa = sd
        else:
            for k in swa:
                if swa[k].is_floating_point():
                    swa[k].mul_(n_swa / (n_swa + 1)).add_(sd[k] / (n_swa + 1))
        n_swa += 1
    ep += 1
    next_batch = 0; epoch_loss = 0.0; epoch_batches = 0
    save_resume()
    json.dump(history, open(f"{OUT}/{TAG}_history.json", "w"), indent=1)
    if time.time() - T0 >= TRAIN_BUDGET:
        paused = ep < total_epochs
        break

save_weights(f"{OUT}/{TAG}_last.pt")
if not paused and swa is not None:
    ref = model.state_dict()
    model.load_state_dict({k: v.to(ref[k].dtype) for k, v in swa.items()})
    bn_refresh(model)
pg = predict(model, gold_ids); auc, per = gold_auc(pg)
log(f"status={'PAUSED' if paused else 'COMPLETE'} SWA({n_swa}) gold AUC {auc:.4f} {per}")
if not paused:
    save_weights(f"{OUT}/{TAG}_swa.pt")
pd.DataFrame(pg, index=gold_ids, columns=LAB).to_csv(f"{OUT}/{TAG}_gold.csv")
if PREDICT_HELDOUT and not paused:
    held = [s for s in index if s not in set(train_ids) and s not in gold.index]
    log(f"predicting {len(held)} held-out studies")
    pd.DataFrame(predict(model, held), index=held, columns=LAB).to_csv(f"{OUT}/{TAG}_heldout.csv")
json.dump(dict(arch=ARCH, history=history, status="paused" if paused else "complete",
               epoch=ep, next_batch=next_batch, gold_auc=auc, per=per, n_train=len(train_ids)),
          open(f"{OUT}/{TAG}_log.json", "w"), indent=1)
log_memory()
log("SESSION_DONE; resume checkpoint is ready" if paused else "TRAINING_COMPLETE")

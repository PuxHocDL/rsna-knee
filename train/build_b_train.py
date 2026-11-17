"""Direction B training kernel (GPU, T4): heliosli's compact_train (v9.8, + [shards] multi-directory --px, + [fp16] T4 autocast)
on the 10 bloctuat px256 v4 shards, EfficientNetV2-S (timm ImageNet weights, downloaded), label v3 targets, arm E_prior.
10 hash folds, only fold 0 trained: 90 % of the pool, and 435 out-of-fold studies as an honest check besides gold.
MODE smoke: --forward-only on the shards that exist (shapes, memory, fp16 path). MODE full: training with --resume; a later
session attaches the previous session's output (RESUME_KERNEL) and continues from its fold checkpoint.

usage: python build_b_train.py <owner> smoke|full [resume_kernel] [suffix] [smoke_shards] [--fold K] [--targets owner/ds:file.csv]
       [--backbone timm_name] [--epochs N] [--max-hours H] [--shards 0,1,...]   (flags may follow the positional arguments; pass '' for unused positionals)"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, MODE = sys.argv[1], sys.argv[2]
FLAGS = sys.argv[3:]
def flag(name, default=None):
    return FLAGS[FLAGS.index(name) + 1] if name in FLAGS else default
POS = [a for i, a in enumerate(sys.argv[3:], 3) if not a.startswith("--") and not (i > 3 and sys.argv[i - 1].startswith("--"))]
RESUME = POS[0] if len(POS) > 0 and POS[0] else None
SUFFIX = POS[1] if len(POS) > 1 else ""
FOLD = flag("--fold", "0")
TARGETS = flag("--targets")          # owner/dataset:file.csv; default: targets_b_v3.csv inside the kit
BACKBONE = flag("--backbone", "tf_efficientnetv2_s.in21k_ft_in1k")
EPOCHS = flag("--epochs", "16")
SHARD_IDS = [int(x) for x in flag("--shards", ",".join(str(k) for k in range(10))).split(",")]   # px shard kernels to mount
NSH = len(SHARD_IDS)
MAX_HOURS = float(flag("--max-hours", "11.3"))
KIT = flag("--kit", "rsna-b-train-kit")
ZOOM = float(flag("--zoom", "1"))     # >1: specialist zoom of the local views (ids 1-5): centre 1/ZOOM, resized back to the cache size
EXTRA = (flag("--extra", "") or "").split()   # extra compact_train arguments, e.g. "--ema 0.999 --aug v2"   # dataset slug (owner's) holding compact_train_shards.py
ARGS = ["--arms", "E_prior", "--backbone", BACKBONE, "--init", "timm", "--folds", "10", "--only-folds", FOLD,
        "--epochs", EPOCHS, "--batch", "2", "--train-windows", "64", "--crop", "224", "--slices", "3", "--phys-mm", "3.5",
        "--spatial-attn", "--lr", "1e-4", "--head-lr", "1e-3", "--wd", "0.05", "--workers", "4", "--eval-workers", "2",
        "--eval-stride", "3", "--resume", "--ckpt-every", "1",
        "--i-have-user-authorisation-to-finetune"] + EXTRA   # the user asked for full fine-tuning (S6fall ran with the same flag)

ZOOM_DEF = """
ZOOM = %r


def zoom_locals(img, views, z):
    # specialist input: every local view (ids 1-5; 0 = full field kept) shows its centre 1/z, resized back to S
    sel = np.flatnonzero((views >= 1) & (views <= 5))
    if z <= 1 or not len(sel):
        return img
    S = img.shape[-1]
    c = int(round(S / z))
    o = (S - c) // 2
    x = torch.from_numpy(np.ascontiguousarray(img[sel, :, o:o + c, o:o + c])).float()
    x = F.interpolate(x, size=(S, S), mode='bilinear', align_corners=False)
    img = np.array(img, copy=True)
    img[sel] = x.round().clamp(0, 255).to(torch.uint8).numpy()
    return img
""" % ZOOM
code = r'''import os, sys, glob, json, shutil, subprocess, time
T0 = time.time()
MODE = %r
NSH = %d
MAX_HOURS = %r
ARGS = %r
TARGETS_FILE = %r
ZOOM_DEF = %r
def shallow(name):   # never recurse into the competition's DICOM tree
    hits = []
    for pat in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/'):
        hits += [h for h in glob.glob(pat + name) if 'abnormality-detection' not in h]
    return sorted(set(hits))
shards = sorted({os.path.dirname(p) for p in shallow('px_s*/index.json')})
print('shards found', len(shards), [os.path.basename(s) for s in shards], flush=True)
if MODE == 'full':
    assert len(shards) == NSH, 'need all %%d shards' %% NSH
kit = os.path.dirname(shallow('compact_train_shards.py')[0])
TARGETS_PATH = shallow(TARGETS_FILE)[0] if TARGETS_FILE else kit + '/targets_b_v3.csv'
print('targets', TARGETS_PATH, flush=True)
OUT = '/kaggle/working/b_out'
prev = shallow('b_out/ckpt')
if prev and not os.path.isdir(OUT):
    shutil.copytree(os.path.dirname(prev[0]), OUT)
    print('RESUME from', os.path.dirname(prev[0]), os.listdir(OUT + '/ckpt'), flush=True)
os.makedirs(OUT, exist_ok=True)
env = dict(os.environ, CT_FP16='1', PYTHONUNBUFFERED='1', PYTHONPATH=kit + os.pathsep + os.environ.get('PYTHONPATH', ''))
ZOOM = %r
TRAINER = kit + '/compact_train_shards.py'
if ZOOM > 1:
    src = open(TRAINER, encoding='utf-8').read()
    anchor = "        meta = np.stack([m[:, 1], m[:, 3], m[:, 4], m[:, 5], m[:, 6], m[:, 7]], 1)"
    assert src.count(anchor) == 1 and src.count(chr(10) + "def collate(batch):") == 1
    src = src.replace(anchor, "        if ZOOM > 1:" + chr(10) + "            img = zoom_locals(img, m[:, 3], ZOOM)" + chr(10) + anchor)
    src = src.replace(chr(10) + "def collate(batch):", ZOOM_DEF + chr(10) + chr(10) + "def collate(batch):")
    TRAINER = '/kaggle/working/compact_train_zoom.py'
    open(TRAINER, 'w', encoding='utf-8').write(src)
    print('ZOOM trainer written, zoom', ZOOM, flush=True)
cmd = [sys.executable, '-u', TRAINER, '--px', ','.join(shards), '--targets', TARGETS_PATH,
       '--out', OUT] + ARGS + (['--forward-only'] if MODE == 'smoke' else [])
print(' '.join(cmd), flush=True)
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], check=False)
MAX_SEC = MAX_HOURS * 3600   # stop before the account's GPU quota runs out; the fold checkpoint stays in b_out/ckpt
try:
    rc = subprocess.run(cmd, env=env, timeout=MAX_SEC).returncode
except subprocess.TimeoutExpired:
    rc = 0
    print('SESSION_TIMEBOX %%.1f h reached: resume from b_out/ckpt in the next session' %% MAX_HOURS, flush=True)
print('TRAIN_RC', rc, 'minutes %%.1f' %% ((time.time() - T0) / 60), flush=True)
for p in sorted(glob.glob(OUT + '/**', recursive=True))[:60]:
    if os.path.isfile(p):
        print('  out', p[len(OUT) + 1:], os.path.getsize(p), flush=True)
if rc != 0:
    raise SystemExit(rc)
''' % (MODE, NSH, MAX_HOURS, ARGS, TARGETS.split(':')[1] if TARGETS else '', ZOOM_DEF, ZOOM)
compile(code, "b_train", "exec")
slug = "rsna-b-train-%s%s" % (MODE, SUFFIX)
out = ROOT / "kaggle_push" / "b_train" / slug
out.mkdir(parents=True, exist_ok=True)
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
(out / (slug + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
SMOKE_SHARDS = [int(x) for x in POS[2].split(",")] if len(POS) > 2 else [0]
sources = ["%s/rsna-b-px256-s%02d" % (OWNER, k) for k in (SHARD_IDS if MODE == "full" else SMOKE_SHARDS)] + ([RESUME] if RESUME else [])
meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": sorted({f"{OWNER}/{KIT}"} | ({TARGETS.split(":")[0]} if TARGETS else set())), "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": sources, "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out)

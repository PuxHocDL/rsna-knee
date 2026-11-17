"""Train direction-B models on Modal GPUs with exactly the Kaggle recipe (compact_train_shards.py, EffV2-S, arm E_prior,
16 epochs, batch 2, fp16 autocast), one GPU per fold, in parallel.  Data: the 10 bloctuat px256 v4 shards (kernel outputs),
fetched from Kaggle straight into a Modal Volume; the Kaggle token is read from ~/.kaggle/zoo.json on this machine and
passed to the fetch calls as an argument (never printed).

  set MODAL_PROFILE=do-mi-xi
  modal run modal_btrain.py::setup --files <targets.csv>[,<more files>]     kit + label files -> volume, then fetch shards
  modal run --detach modal_btrain.py::train_many --targets targets_b_v10c.csv --tag v10c --folds 2,3,4,7
  modal volume get rsna-b-vol runs/v10c-f2 <local dir>
"""
import json, os, subprocess, sys, threading, time
import modal

APP = modal.App("rsna-b-train")
VOL = modal.Volume.from_name("rsna-b-vol", create_if_missing=True)
IMG = (modal.Image.debian_slim(python_version="3.11").apt_install("curl")
       .pip_install("torch==2.5.1", "timm==1.0.11", "huggingface_hub==0.26.2", "numpy==1.26.4", "pandas==2.2.3", "kaggle==2.2.4")
       .run_commands("python -c \"import timm; [timm.create_model(m, pretrained=True) for m in ('tf_efficientnetv2_s.in21k_ft_in1k', 'tf_efficientnetv2_m.in21k_ft_in1k', 'vit_small_patch14_dinov2.lvd142m')]; print('timm weights cached')\""))
OWNER = "bloctuat"
GPU = os.environ.get("B_GPU", "A100-40GB")
ARGS = ["--arms", "E_prior", "--backbone", "tf_efficientnetv2_s.in21k_ft_in1k", "--init", "timm", "--folds", "10",
        "--epochs", "16", "--batch", "2", "--train-windows", "64", "--crop", "224", "--slices", "3", "--phys-mm", "3.5",
        "--spatial-attn", "--lr", "1e-4", "--head-lr", "1e-3", "--wd", "0.05", "--workers", "4", "--eval-workers", "2",
        "--eval-stride", "3", "--resume", "--ckpt-every", "1", "--i-have-user-authorisation-to-finetune"]


def merge_args(base, over):
    args = list(base); toks = over.split()
    i = 0
    while i < len(toks):
        f = toks[i]; v = toks[i + 1] if i + 1 < len(toks) and not toks[i + 1].startswith("--") else None
        if f in args:
            j = args.index(f)
            if v is not None: args[j + 1] = v
        else:
            args += [f] + ([v] if v is not None else [])
        i += 2 if v is not None else 1
    return args


@APP.function(image=IMG, volumes={"/vol": VOL}, timeout=4 * 3600, cpu=2, memory=4096)
def fetch(k: int, token_env: dict):
    # resumable: list the kernel's output files (signed URLs), then curl -C - each one until it completes; a .complete
    # marker is written only after every file finished, so a half-downloaded shard is never mistaken for a full one
    os.environ.update(token_env)
    dst = "/vol/px/s%02d" % k
    if os.path.exists(dst + "/.complete"):
        return "s%02d complete" % k
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    api = KaggleApi(); api.authenticate()
    def listing():
        with api.build_kaggle_client() as kc:
            rq = ApiListKernelSessionOutputRequest(); rq.user_name = OWNER; rq.kernel_slug = "rsna-b-px256-s%02d" % k
            return [(f.file_name, f.url) for f in kc.kernels.kernels_api_client.list_kernel_session_output(rq).files or []]
    files = listing(); log = []
    for name, url in files:
        out = os.path.join(dst, name); os.makedirs(os.path.dirname(out), exist_ok=True)
        for attempt in range(60):
            r = subprocess.run(["curl", "-sS", "-L", "-f", "-C", "-", "--retry", "5", "--retry-delay", "3", "-o", out, url],
                               capture_output=True, text=True)
            if r.returncode == 0:
                break
            if r.returncode in (22, 403):          # expired signed URL: list again
                url = dict(listing())[name]
            if attempt % 5 == 4:
                VOL.commit()
            time.sleep(3)
        log.append("%s %.2fGB rc%d" % (name, os.path.getsize(out) / 1e9 if os.path.exists(out) else -1, r.returncode))
        if r.returncode != 0:
            VOL.commit()
            return "s%02d FAILED %s" % (k, log)
    open(dst + "/.complete", "w").write(json.dumps(log))
    VOL.commit()
    return "s%02d done %s" % (k, log)


@APP.function(image=IMG, volumes={"/vol": VOL}, gpu=GPU, timeout=10 * 3600, cpu=8, memory=49152, ephemeral_disk=512 * 1024)
def train(targets: str, tag: str, fold: str, over: str = ""):
    # over: flags that replace (or add to) the B4 recipe, e.g. "--backbone tf_efficientnetv2_m.in21k_ft_in1k --slices 5 --epochs 24"
    VOL.reload()
    assert all(os.path.exists("/vol/px/s%02d/.complete" % k) for k in range(10)), "shards incomplete"
    import shutil
    t1 = time.time()
    shutil.copytree("/vol/px", "/tmp/px", dirs_exist_ok=True)       # local NVMe: the trainer memory-maps px.u8.npy
    shards = sorted({r for r, _, fs in os.walk("/tmp/px") for f in fs if f == "index.json"})
    assert len(shards) == 10, shards
    print("shards copied to local disk in %.1f min" % ((time.time() - t1) / 60), flush=True)
    out = "/vol/runs/%s-f%s" % (tag, fold)
    os.makedirs(out, exist_ok=True)
    stop = threading.Event()
    def committer():
        while not stop.wait(300):
            try: VOL.commit()
            except Exception as e: print("commit failed", e, flush=True)
    threading.Thread(target=committer, daemon=True).start()
    cmd = [sys.executable, "-u", "/vol/kit/compact_train_shards.py", "--px", ",".join(shards), "--targets", "/vol/kit/" + targets,
           "--out", out, "--only-folds", fold] + merge_args(ARGS, over)
    print(" ".join(cmd), flush=True)
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    env = dict(os.environ, CT_FP16="1", PYTHONUNBUFFERED="1", PYTHONPATH="/vol/kit")
    t0 = time.time()
    with open(out + "/train_log.txt", "a") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    stop.set(); VOL.commit()
    tail = open(out + "/train_log.txt").read()[-1500:]
    return "%s-f%s rc %d minutes %.1f\n%s" % (tag, fold, rc, (time.time() - t0) / 60, tail)


@APP.local_entrypoint()
def setup(files: str = "", runs: str = ""):
    kit = os.path.join(os.path.dirname(os.path.abspath(__file__)), "b-kit-v5-bloctuat", "compact_train_shards.py")
    with VOL.batch_upload(force=True) as b:
        b.put_file(kit, "/kit/compact_train_shards.py")
        for f in [x for x in files.split(",") if x]:
            b.put_file(f, "/kit/" + os.path.basename(f))
            print("uploaded", os.path.basename(f))
        if runs:                                   # resume material: <runs>/<tag-fK>/{ckpt/*.pt, run_config.json, pool_uids.json, ...}
            for name in sorted(os.listdir(runs)):
                b.put_directory(os.path.join(runs, name), "/runs/" + name)
                print("uploaded run", name)
    z = json.load(open(os.path.expanduser("~/.kaggle/zoo.json"), encoding="utf-8"))
    tok = {"KAGGLE_API_TOKEN": z["accounts"][OWNER]["token"]}
    for line in fetch.map(range(10), kwargs={"token_env": tok}):
        print(line, flush=True)


@APP.local_entrypoint()
def train_many(targets: str, tag: str, folds: str = "2,3,4,7"):
    fl = folds.split(",")
    for res in train.starmap([(targets, tag, f) for f in fl], return_exceptions=True):
        print("=" * 30, "\n", res, flush=True)

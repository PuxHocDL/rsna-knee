"""Build Kaggle push folders for the training pipeline.
usage: python build_train.py <user> cache|train [tag arch lr_bb [checkpoint_kernel run_suffix [--single-gpu] [--train-budget-minutes N]]]
Training sessions save full resume state and pause after the time budget (default 90 minutes) or the RAM guard. To continue,
attach the previous completed session using checkpoint_kernel and a new run_suffix.
Legacy partial checkpoints provide weights-only warm starts, not exact resumes."""
import base64, json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PUSH = os.path.join(HERE, "push")
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
NSH = 3


def nb(src):
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                       "source": src.splitlines(keepends=True)}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def write(slug, title, src, meta):
    d = os.path.join(PUSH, USER, slug); os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, slug + ".ipynb"), "w", encoding="utf-8") as f:
        json.dump(nb(src), f, separators=(",", ":"))
    m = dict(id=f"{USER}/{slug}", title=title, code_file=slug + ".ipynb", language="python",
             kernel_type="notebook", is_private=True, docker_image=DOCKER,
             competition_sources=["rsna-knee-abnormality-detection"], dataset_sources=[],
             kernel_sources=[], model_sources=[])
    m.update(meta)
    with open(os.path.join(d, "kernel-metadata.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)
    print("built", d, f"{os.path.getsize(os.path.join(d, slug + '.ipynb'))/1e3:.0f}KB")


USER, MODE = sys.argv[1], sys.argv[2]
if MODE == "cache":
    r224 = open(os.path.join(os.environ["TEMP"], "r224.py"), encoding="utf-8").read()
    b64 = re.search(r'_FASTREAD_B64 = "([^"]+)"', r224).group(1)
    base64.b64decode(b64)
    tpl = open(os.path.join(HERE, "cache_kernel.py"), encoding="utf-8").read()
    for k in range(NSH):
        src = tpl.replace("__SHARD__", str(k)).replace("__NSH__", str(NSH)).replace("__FASTREAD_B64__", b64)
        write(f"rsna-tk-cache-{k}", f"rsna-tk-cache-{k}", src,
              dict(enable_gpu=False, enable_internet=False))
else:
    tag, arch, lr = sys.argv[3], sys.argv[4], sys.argv[5]
    checkpoint = sys.argv[6] if len(sys.argv) > 6 else None
    suffix = "-" + sys.argv[7] if len(sys.argv) > 7 else ""
    src = (open(os.path.join(HERE, "train_kernel.py"), encoding="utf-8").read()
           .replace("__ARCH__", arch).replace("__TAG__", tag).replace("__LR_BB__", lr))
    if "--single-gpu" in sys.argv[8:]:
        src = src.replace("USE_DATA_PARALLEL = True", "USE_DATA_PARALLEL = False", 1)
    if "--continue" in sys.argv[8:]:
        n = int(sys.argv[sys.argv.index("--continue") + 1])
        src = src.replace("CONTINUE_EPOCHS = 0 ", f"CONTINUE_EPOCHS = {n} ", 1)
        src = src.replace("TRAIN_BUDGET = 90 * 60", "TRAIN_BUDGET = 240 * 60", 1)   # single GPU: no RAM growth
    if "--train-budget-minutes" in sys.argv[8:]:
        budget_index = sys.argv.index("--train-budget-minutes")
        if budget_index + 1 >= len(sys.argv):
            raise SystemExit("--train-budget-minutes requires a positive integer")
        try:
            minutes = int(sys.argv[budget_index + 1])
        except ValueError:
            raise SystemExit("--train-budget-minutes requires a positive integer")
        if minutes <= 0:
            raise SystemExit("--train-budget-minutes requires a positive integer")
        src, count = re.subn(r"^TRAIN_BUDGET = \d+ \* 60$", f"TRAIN_BUDGET = {minutes} * 60", src, flags=re.M)
        if count != 1:
            raise RuntimeError("Expected exactly one TRAIN_BUDGET assignment")
    sources = [f"{USER}/rsna-tk-cache-{k}" for k in range(NSH)]
    datasets = ["pilkwang/rsna-knee-llm-labels", "stevenleehans/rsna-knee-llm-report-labels"]
    if "--mirror" in sys.argv[8:]:
        p_ = float(sys.argv[sys.argv.index("--mirror") + 1])
        src, n = re.subn(r"^MIRROR_P = 0\.0 ", f"MIRROR_P = {p_} ", src, flags=re.M)
        assert n == 1
    if "--predict-heldout" in sys.argv[8:]:
        src, n = re.subn(r"^PREDICT_HELDOUT = False", "PREDICT_HELDOUT = True", src, flags=re.M)
        assert n == 1
    if "--targets" in sys.argv[8:]:   # --targets <owner/dataset>:<file.csv>
        ds_slug, tfile = sys.argv[sys.argv.index("--targets") + 1].split(":")
        src, n = re.subn(r'^TARGETS_FILE = ""', f'TARGETS_FILE = "{tfile}"', src, flags=re.M)
        assert n == 1
        datasets.append(ds_slug)
    if checkpoint:
        src = src.replace("REQUIRE_CHECKPOINT = False", "REQUIRE_CHECKPOINT = True", 1)
        if checkpoint.startswith("dataset:"):
            datasets.append(checkpoint.removeprefix("dataset:"))
        else:
            sources.append(checkpoint)
    write(f"rsna-tk-train-{tag}{suffix}", f"rsna-tk-train-{tag}{suffix}", src,
          dict(enable_gpu=True, enable_internet=True, machine_shape="NvidiaTeslaT4",
               dataset_sources=datasets,
               kernel_sources=sources))

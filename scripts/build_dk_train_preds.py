"""DK (dkim54/rsna-knee-models-v11i, 10 checkpoints, no mirror TTA) on one part of the training pool, for label v8c
(0.4 lingyu07 + 0.3 R6 OOF + 0.3 DK).  The pool = train studies minus the 58 gold and the 392 reference studies (they stay
blank in the training targets).  Part k of NPART (studies sorted by md5, every NPART-th) is split in two halves, one
infer_lib.predict() process per T4.  Output: dk_train_part<k>.csv (probabilities, label columns).
usage: python build_dk_train_preds.py <owner> <slug> <holdout_ids_dataset> <k> <npart>"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, SLUG, HOLD, K, NPART = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
code = r'''import glob, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd
T0 = time.time()
K, NPART = %d, %d
COMP = Path(sorted(glob.glob('/kaggle/input/**/test.csv', recursive=True), key=len)[0]).parent
MODELS = Path(sorted(glob.glob('/kaggle/input/**/models.json', recursive=True), key=len)[0]).parent
CODE = Path(sorted(glob.glob(str(MODELS) + '/**/infer_lib.py', recursive=True), key=len)[0]).parent
tr = pd.read_csv(COMP / 'train.csv', dtype={'StudyInstanceUID': str})
LABELS = [c for c in tr.columns if c not in ('StudyInstanceUID', 'Report')]
gold = set(tr[tr[LABELS].notna().all(axis=1)]['StudyInstanceUID'])
hold = set(pd.read_csv(sorted(glob.glob('/kaggle/input/**/holdout_ref392_ids.csv', recursive=True))[0]).iloc[:, 0].astype(str))
pool = sorted((u for u in tr['StudyInstanceUID'] if u not in gold and u not in hold), key=lambda u: hashlib.md5(u.encode()).hexdigest())
mine = pool[K::NPART]
print('pool', len(pool), 'part', K, 'of', NPART, 'studies', len(mine), flush=True)
spec = json.load(open(MODELS / 'models.json'))
md = Path('/tmp/models_full10_notta'); md.mkdir(exist_ok=True)
models = [dict(m, tta=False) for m in spec['models']]
for m in models:
    if not (md / m['weights']).exists():
        os.symlink(MODELS / m['weights'], md / m['weights'])
json.dump(dict(spec, models=models), open(md / 'models.json', 'w'))
ser = pd.read_csv(COMP / 'train_series.csv', dtype=str)
procs = []
for h, part in enumerate([mine[0::2], mine[1::2]]):
    R = Path('/tmp/root%%d' %% h); R.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'StudyInstanceUID': part}).to_csv(R / 'train.csv', index=False)
    ser[ser['StudyInstanceUID'].isin(set(part))].to_csv(R / 'train_series.csv', index=False)
    (R / 'train_series').mkdir(exist_ok=True)
    for u in part:
        if (COMP / 'train_series' / u).exists() and not (R / 'train_series' / u).exists():
            os.symlink(COMP / 'train_series' / u, R / 'train_series' / u)
    script = ("import sys; sys.path.insert(0, %%r); import infer_lib; infer_lib.predict(%%r, %%r, %%r, split='train', workers=2)"
              %% (str(CODE), str(R), str(md), '/tmp/dk_%%d.csv' %% h))
    log = open('/kaggle/working/dk_%%d_log.txt' %% h, 'w')
    procs.append((subprocess.Popen([sys.executable, '-u', '-c', script], stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(h))), log))
rcs = []
for p, log in procs:
    rcs.append(p.wait()); log.close()
out = pd.concat([pd.read_csv('/tmp/dk_%%d.csv' %% h) for h in range(2) if os.path.exists('/tmp/dk_%%d.csv' %% h)])
out.to_csv('/kaggle/working/dk_train_part%%d.csv' %% K, index=False)
print('DKTRAIN_DONE part', K, 'rc', rcs, 'rows', len(out), 'minutes %%.1f' %% ((time.time() - T0) / 60), flush=True)
for h in range(2):
    print(open('/kaggle/working/dk_%%d_log.txt' %% h).read()[-300:], flush=True)
''' % (K, NPART)
compile(code, SLUG, "exec")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}
out = ROOT / "kaggle_push" / SLUG
out.mkdir(parents=True, exist_ok=True)
(out / (SLUG + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": SLUG + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["dkim54/rsna-knee-models-v11i", HOLD], "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out)

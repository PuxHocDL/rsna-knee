"""Build a Kaggle kernel that scores heliosli's R6 (4 fold models, flip TTA, stackers) on the 58 gold train studies.

Cell 6 of heliosli/blend-mine-in-private-lb-luck-to-all-upvoters (the R6 setup: code, wheels, bundle checks, worker args)
is reused verbatim; the Raptor stages and the blend are dropped. The controller runs with --split train on a scratch
data root that holds only the 58 gold studies (file-level symlinks), because the native reader globs the whole split.
R6 was trained with the gold studies excluded (manifest gold_in_training = 0), so the 4-fold mean is a clean gold read.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb"
OUT = ROOT / "kaggle_push" / "r6-gold58"
nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
setup = "".join(nb["cells"][6]["source"])
assert setup.startswith("ARM = 'E_lrprior'") and "FLIP_TTA = True" in setup and "T0 = NB_T0" in setup

cell0 = '''# R6 on the 58 gold train studies (gate 0 of the image-corrected-label plan). No submission is produced.
import time
NB_T0 = time.time()
'''

cell_root = '''# Scratch data root with only the gold studies: train.csv / train_series.csv filtered, DICOM files symlinked one by one
import csv, os
import pandas as pd
REAL = DATA
tr = pd.read_csv(REAL / 'train.csv', dtype={'StudyInstanceUID': str})
LABELS = [c for c in tr.columns if c not in ('StudyInstanceUID', 'Report')]
gold = tr[tr[LABELS].notna().all(axis=1)].reset_index(drop=True)
assert len(gold) == 58, len(gold)
GOLD_IDS = gold['StudyInstanceUID'].tolist()
FAKE = TMP / 'gold_root'
FAKE.mkdir(parents=True, exist_ok=True)
gold.to_csv(FAKE / 'train.csv', index=False)
ser = pd.read_csv(REAL / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
ser[ser['StudyInstanceUID'].isin(set(GOLD_IDS))].to_csv(FAKE / 'train_series.csv', index=False)
header = list(pd.read_csv(REAL / 'sample_submission.csv', nrows=1).columns)
with open(FAKE / 'sample_submission.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh); w.writerow(header)
    for u in GOLD_IDS:
        w.writerow([u] + [0.5] * (len(header) - 1))
nfiles = 0
for u in GOLD_IDS:
    for dp, dn, fn in os.walk(REAL / 'train_series' / u):
        rel = Path(dp).relative_to(REAL)
        (FAKE / rel).mkdir(parents=True, exist_ok=True)
        for f in fn:
            dst = FAKE / rel / f
            if not dst.exists():
                os.symlink(Path(dp) / f, dst)
            nfiles += 1
print('GOLD_ROOT', FAKE, 'studies', len(GOLD_IDS), 'files', nfiles, flush=True)
DATA = FAKE
'''

cell_run = '''deadline = T0 + DEADLINE_HOURS * 3600
env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
pred = TMP / 'prediction'
rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'train',
                     '--code', str(CODE), '--bundle', str(BUNDLE), '--arm', ARM, '--output', str(pred),
                     '--amp', 'auto', '--loader-workers', '2', '--deadline', str(deadline),
                     '--worker', WORKER, '--worker-args', json.dumps(WORKER_ARGS),
                     '--max-fallback-fraction', '1.0'], env=env).returncode
for p in sorted(pred.glob('worker*/*')):
    if p.suffix in ('.txt', '.csv', '.json'):
        shutil.copyfile(p, WORK / (p.parent.name + '_' + p.name))
if rc != 0:
    raise RuntimeError('controller failed with exit code %d' % rc)
shutil.copyfile(pred / 'submission.csv', WORK / 'r6_gold58.csv')
shutil.copyfile(pred / 'submission_receipt.json', WORK / 'r6_gold58_receipt.json')
'''

cell_eval = '''# Per-label AUC of R6 on gold (fallback rows = column mean are excluded)
import numpy as np
from sklearn.metrics import roc_auc_score
p = pd.read_csv(WORK / 'r6_gold58.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[GOLD_IDS]
rec = json.loads((WORK / 'r6_gold58_receipt.json').read_text())
bad = set(rec.get('fallback_reasons', {}))
ok = [u for u in GOLD_IDS if u not in bad]
g = gold.set_index('StudyInstanceUID').loc[ok]
res = {c: roc_auc_score(g[c].astype(int), p.loc[ok, c]) for c in LABELS}
print('scored', len(ok), 'of', len(GOLD_IDS), 'fallback', sorted(bad))
for c, v in res.items():
    print('%-18s %.4f' % (c, v))
print('MACRO %.4f' % np.mean(list(res.values())))
json.dump({'scored': len(ok), 'auc': res, 'macro': float(np.mean(list(res.values())))}, open(WORK / 'r6_gold58_auc.json', 'w'), indent=1)
'''

def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}

out_nb = {"cells": [code(cell0), code(setup), code(cell_root), code(cell_run), code(cell_eval)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "rsna-r6-gold58.ipynb").write_text(json.dumps(out_nb, indent=1), encoding="utf-8")
meta = {"id": "nguynxunphc662/rsna-r6-gold58", "title": "rsna-r6-gold58", "code_file": "rsna-r6-gold58.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", OUT)

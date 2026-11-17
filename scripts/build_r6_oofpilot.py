"""Dry run for R6 out-of-fold labels: 24 non-gold train studies (6 per hash fold), each scored by every fold model
separately, no flip TTA. Two reads: (1) fold-assignment check -- a fold model should sit closer to the report labels on
the studies it trained on than on its held-out fold; (2) seconds per study for the full OOF pass."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb"
OUT = ROOT / "kaggle_push" / "r6-oofpilot"
nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
setup = "".join(nb["cells"][6]["source"])
assert setup.startswith("ARM = 'E_lrprior'") and "T0 = NB_T0" in setup

cell0 = '''# R6 OOF dry run (no submission)
import time
NB_T0 = time.time()
N_PER_FOLD = 6
K = 4
'''

cell_root = '''import csv, os, hashlib
import pandas as pd
REAL = DATA
tr = pd.read_csv(REAL / 'train.csv', dtype={'StudyInstanceUID': str})
LABELS = [c for c in tr.columns if c not in ('StudyInstanceUID', 'Report')]
pool = tr[tr[LABELS].isna().all(axis=1)]['StudyInstanceUID'].tolist()
fold_of = {u: int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % K for u in pool}
pick = []
for k in range(K):
    us = sorted([u for u in pool if fold_of[u] == k], key=lambda u: hashlib.md5(u.encode()).hexdigest())
    pick += us[:N_PER_FOLD]
FAKE = TMP / 'pilot_root'
FAKE.mkdir(parents=True, exist_ok=True)
tr[tr['StudyInstanceUID'].isin(set(pick))].to_csv(FAKE / 'train.csv', index=False)
ser = pd.read_csv(REAL / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
ser[ser['StudyInstanceUID'].isin(set(pick))].to_csv(FAKE / 'train_series.csv', index=False)
header = list(pd.read_csv(REAL / 'sample_submission.csv', nrows=1).columns)
order = tr[tr['StudyInstanceUID'].isin(set(pick))]['StudyInstanceUID'].tolist()
with open(FAKE / 'sample_submission.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh); w.writerow(header)
    for u in order:
        w.writerow([u] + [0.5] * (len(header) - 1))
for u in pick:
    for dp, dn, fn in os.walk(REAL / 'train_series' / u):
        rel = Path(dp).relative_to(REAL)
        (FAKE / rel).mkdir(parents=True, exist_ok=True)
        for f in fn:
            if not (FAKE / rel / f).exists():
                os.symlink(Path(dp) / f, FAKE / rel / f)
pd.DataFrame({'StudyInstanceUID': pick, 'hash_fold': [fold_of[u] for u in pick]}).to_csv(WORK / 'pilot_ids.csv', index=False)
print('PILOT', len(pick), 'studies', flush=True)
DATA = FAKE
ARGS = [x for x in WORKER_ARGS if x != '--flip-tta']    # the full OOF pass would run without flip TTA
assert '--flip-tta' not in ARGS
'''

cell_run = '''deadline = T0 + DEADLINE_HOURS * 3600
env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
timing = {}
for k in range(K):
    bk = TMP / ('bundle_f%d' % k)
    bk.mkdir(exist_ok=True)
    name = 'E_lrprior_s0_f%d.pt' % k
    assert (BUNDLE / name).is_file(), name
    if not (bk / name).exists():
        os.symlink(BUNDLE / name, bk / name)
    pred = TMP / ('prediction_f%d' % k)
    t = time.time()
    rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'train',
                         '--code', str(CODE), '--bundle', str(bk), '--arm', ARM, '--output', str(pred),
                         '--amp', 'auto', '--loader-workers', '2', '--deadline', str(deadline),
                         '--worker', WORKER, '--worker-args', json.dumps(ARGS), '--max-fallback-fraction', '1.0'],
                        env=env).returncode
    timing[k] = time.time() - t
    for p in sorted(pred.glob('worker*/log.txt')):
        shutil.copyfile(p, WORK / ('f%d_%s_log.txt' % (k, p.parent.name)))
    if rc != 0:
        raise RuntimeError('fold %d controller failed with exit code %d' % (k, rc))
    shutil.copyfile(pred / 'submission.csv', WORK / ('pilot_f%d.csv' % k))
    shutil.copyfile(pred / 'submission_receipt.json', WORK / ('pilot_f%d_receipt.json' % k))
    print('FOLD_DONE', k, 'seconds %.0f' % timing[k], flush=True)
json.dump(timing, open(WORK / 'pilot_timing.json', 'w'), indent=1)
'''

def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}

out_nb = {"cells": [code(cell0), code(setup), code(cell_root), code(cell_run)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "rsna-r6-oofpilot.ipynb").write_text(json.dumps(out_nb, indent=1), encoding="utf-8")
meta = {"id": "nguynxunphc662/rsna-r6-oofpilot", "title": "rsna-r6-oofpilot", "code_file": "rsna-r6-oofpilot.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", OUT)

"""R6 out-of-fold predictions for every non-gold train study (input to the image-corrected labels).

Each study is scored only by the fold model that did not train on it: hash fold k = sha1(uid)[:8] % 4 -> E_lrprior_s0_f{k}.pt
(the R6 trainer's own split; dry run rsna-r6-oofpilot). No flip TTA, no torch.compile. Studies run in chunks of CHUNK with a
scratch data root each, because the native reader globs the whole split directory per study. Part A = folds 0, 1 plus a
check (40 hash-0 studies also scored by f3); part B = folds 2, 3 plus a check (40 hash-3 studies also scored by f0).
Outputs: oof_f{k}_c{i}.csv (+ fluid/state stacker dumps), check_*.csv, progress.json after every chunk."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_NB = ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb"
nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
setup = "".join(nb["cells"][6]["source"])
assert setup.startswith("ARM = 'E_lrprior'") and "T0 = NB_T0" in setup and "DEADLINE_HOURS = 8.3" in setup
setup = setup.replace("DEADLINE_HOURS = 8.3", "DEADLINE_HOURS = 11.3", 1)

PARTS = {"a": dict(folds=[0, 1], check=(0, 3)), "b": dict(folds=[2, 3], check=(3, 0))}

CELL_PLAN = '''import csv, os, hashlib
import pandas as pd
REAL = DATA
tr = pd.read_csv(REAL / 'train.csv', dtype={'StudyInstanceUID': str})
LABELS = [c for c in tr.columns if c not in ('StudyInstanceUID', 'Report')]
pool = tr[tr[LABELS].isna().all(axis=1)]['StudyInstanceUID'].tolist()
fold_of = {u: int(hashlib.sha1(u.encode()).hexdigest()[:8], 16) % K for u in pool}
ser = pd.read_csv(REAL / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
header = list(pd.read_csv(REAL / 'sample_submission.csv', nrows=1).columns)
ARGS = [x for x in WORKER_ARGS if x not in ('--flip-tta', '--compile')]
tasks = []
for k in FOLDS:
    us = [u for u in pool if fold_of[u] == k]
    tasks += [('oof_f%d_c%02d' % (k, i // CHUNK), k, us[i:i + CHUNK]) for i in range(0, len(us), CHUNK)]
h, m = CHECK
chk = sorted([u for u in pool if fold_of[u] == h], key=lambda u: hashlib.md5(u.encode()).hexdigest())[:N_CHECK]
tasks += [('check_h%d_f%d' % (h, m), m, chk), ('check_h%d_f%d' % (h, h), h, chk)]
print('PLAN', len(tasks), 'chunks,', sum(len(t[2]) for t in tasks), 'study passes', flush=True)
for k in range(K):
    bk = TMP / ('bundle_f%d' % k); bk.mkdir(exist_ok=True)
    name = 'E_lrprior_s0_f%d.pt' % k
    if not (bk / name).exists():
        os.symlink(BUNDLE / name, bk / name)
'''

CELL_RUN = '''deadline = NB_T0 + DEADLINE_HOURS * 3600
env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
progress = {'done': [], 'failed': [], 'skipped': []}


def make_root(ids, root):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    tr[tr['StudyInstanceUID'].isin(set(ids))].to_csv(root / 'train.csv', index=False)
    ser[ser['StudyInstanceUID'].isin(set(ids))].to_csv(root / 'train_series.csv', index=False)
    order = tr[tr['StudyInstanceUID'].isin(set(ids))]['StudyInstanceUID'].tolist()
    with open(root / 'sample_submission.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(header)
        for u in order:
            w.writerow([u] + [0.5] * (len(header) - 1))
    for u in ids:
        for dp, dn, fn in os.walk(REAL / 'train_series' / u):
            rel = Path(dp).relative_to(REAL)
            (root / rel).mkdir(parents=True, exist_ok=True)
            for f in fn:
                os.symlink(Path(dp) / f, root / rel / f)


for name, k, ids in tasks:
    if time.time() - NB_T0 > STOP_AFTER_HOURS * 3600:
        progress['skipped'].append(name); continue
    t = time.time()
    root = TMP / 'chunk_root'; pred = TMP / 'chunk_pred'
    make_root(ids, root)
    if pred.exists():
        shutil.rmtree(pred)
    rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(root), '--split', 'train',
                         '--code', str(CODE), '--bundle', str(TMP / ('bundle_f%d' % k)), '--arm', ARM, '--output', str(pred),
                         '--amp', 'auto', '--loader-workers', '2', '--deadline', str(deadline),
                         '--worker', WORKER, '--worker-args', json.dumps(ARGS), '--max-fallback-fraction', '1.0'],
                        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode
    if rc == 0 and (pred / 'submission.csv').is_file():
        shutil.copyfile(pred / 'submission.csv', WORK / (name + '.csv'))
        rec = json.loads((pred / 'submission_receipt.json').read_text())
        for wd in sorted(pred.glob('worker*')):
            for extra in ('fluid_stacker.csv', 'state_stacker.csv'):
                if (wd / extra).is_file():
                    shutil.copyfile(wd / extra, WORK / ('%s_%s_%s' % (name, wd.name, extra)))
        progress['done'].append({'chunk': name, 'model': k, 'n': len(ids), 'scored': rec['scored'],
                                 'fallback': list(rec['fallback_reasons']), 'seconds': round(time.time() - t, 1)})
        print('CHUNK_DONE', name, 'f%d' % k, '%d/%d' % (rec['scored'], len(ids)), '%.0fs' % (time.time() - t),
              'elapsed %.2fh' % ((time.time() - NB_T0) / 3600), flush=True)
    else:
        logs = sorted(pred.glob('worker*/log.txt'))
        tail = logs[0].read_text(errors='replace')[-1500:] if logs else ''
        progress['failed'].append({'chunk': name, 'rc': rc, 'tail': tail})
        print('CHUNK_FAILED', name, rc, tail[-600:], flush=True)
    shutil.rmtree(pred, ignore_errors=True); shutil.rmtree(root, ignore_errors=True)
    json.dump(progress, open(WORK / 'progress.json', 'w'), indent=1)
print('OOF_PART_DONE', json.dumps({k: len(v) for k, v in progress.items()}), flush=True)
'''


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


def build(part):
    cfg = PARTS[part]
    cell0 = ("# R6 OOF part %s: folds %s, check = hash %d studies also scored by f%d\n" % (part, cfg["folds"], *cfg["check"])
             + "import time\nNB_T0 = time.time()\n"
             + "FOLDS = %r\nCHECK = %r\nN_CHECK = 40\nCHUNK = 120\nK = 4\nSTOP_AFTER_HOURS = 11.0\n" % (cfg["folds"], cfg["check"]))
    out_nb = {"cells": [code(cell0), code(setup), code(CELL_PLAN), code(CELL_RUN)],
              "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                           "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    slug = "rsna-r6-oof-" + part
    out = ROOT / "kaggle_push" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / (slug + ".ipynb")).write_text(json.dumps(out_nb, indent=1), encoding="utf-8")
    meta = {"id": "nguynxunphc662/" + slug, "title": slug, "code_file": slug + ".ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
            "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("built", out)


for p in (sys.argv[1:] or PARTS):
    build(p)

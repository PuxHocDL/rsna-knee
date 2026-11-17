"""DK-lite speed + accuracy check on the 450 clean train studies (58 gold + 392 reference), run exactly as a submission would:
the studies split in two halves, one dkim54 infer_lib.predict() process per T4.  Configs (models.json rewritten in a scratch
copy of the weights dir; the .pt files are symlinked): lite5 = group v2128r3 folds 0-4 without mirror TTA; full10_notta = all 10
checkpoints without TTA.  Writes dk_<cfg>.csv and prints DKLITE_CONFIG <cfg> minutes ... per config.
usage: python build_dk_lite_eval.py <owner> <slug> <holdout_ids_dataset>"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, SLUG, HOLD = sys.argv[1:4]
code = r'''import csv, glob, json, os, subprocess, sys, time
from pathlib import Path
import pandas as pd
T0 = time.time()
COMP = Path(sorted(glob.glob('/kaggle/input/**/test.csv', recursive=True), key=len)[0]).parent
MODELS = Path(sorted(glob.glob('/kaggle/input/**/models.json', recursive=True), key=len)[0]).parent
CODE = Path(sorted(glob.glob(str(MODELS) + '/**/infer_lib.py', recursive=True), key=len)[0]).parent
tr = pd.read_csv(COMP / 'train.csv', dtype={'StudyInstanceUID': str})
LABELS = [c for c in tr.columns if c not in ('StudyInstanceUID', 'Report')]
gold_ids = tr[tr[LABELS].notna().all(axis=1)]['StudyInstanceUID'].tolist()
ref_ids = pd.read_csv(sorted(glob.glob('/kaggle/input/**/holdout_ref392_ids.csv', recursive=True))[0]).iloc[:, 0].astype(str).tolist()
IDS = sorted(set(gold_ids) | set(ref_ids))
ser = pd.read_csv(COMP / 'train_series.csv', dtype=str)
roots = []
for k, part in enumerate([IDS[0::2], IDS[1::2]]):
    R = Path('/tmp/eval_root%d' % k); R.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'StudyInstanceUID': part}).to_csv(R / 'train.csv', index=False)
    ser[ser['StudyInstanceUID'].isin(set(part))].to_csv(R / 'train_series.csv', index=False)
    (R / 'train_series').mkdir(exist_ok=True)
    for u in part:
        if (COMP / 'train_series' / u).exists() and not (R / 'train_series' / u).exists():
            os.symlink(COMP / 'train_series' / u, R / 'train_series' / u)
    roots.append(R)
spec = json.load(open(MODELS / 'models.json'))
CONFIGS = {'lite5': [dict(m, tta=False) for m in spec['models'] if m['weights'].startswith('v2128r3_f')],
           'full10_notta': [dict(m, tta=False) for m in spec['models']]}
print('studies', len(IDS), {k: [m['weights'] for m in v] for k, v in CONFIGS.items()}, flush=True)
for name, models in CONFIGS.items():
    md = Path('/tmp/models_' + name); md.mkdir(exist_ok=True)
    for m in models:
        if not (md / m['weights']).exists():
            os.symlink(MODELS / m['weights'], md / m['weights'])
    json.dump(dict(spec, models=models, note='dk-lite eval: ' + name), open(md / 'models.json', 'w'))
    t1 = time.time(); procs = []
    for k, R in enumerate(roots):
        script = ("import sys; sys.path.insert(0, %r); import infer_lib; infer_lib.predict(%r, %r, %r, split='train', workers=2)"
                  % (str(CODE), str(R), str(md), '/tmp/dk_%s_%d.csv' % (name, k)))
        log = open('/kaggle/working/dk_%s_%d_log.txt' % (name, k), 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', '-c', script], stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))), log))
    rcs = []
    for p, log in procs:
        rcs.append(p.wait()); log.close()
    mins = (time.time() - t1) / 60
    pd.concat([pd.read_csv('/tmp/dk_%s_%d.csv' % (name, k)) for k in range(2)]).to_csv('/kaggle/working/dk_%s.csv' % name, index=False)
    print('DKLITE_CONFIG', name, 'rc', rcs, 'minutes %.1f' % mins, 'sec/study %.2f (2 GPUs)' % (mins * 60 / len(IDS)),
          'projected 1300 studies %.2f h' % (mins * 60 / len(IDS) * 1300 / 3600), flush=True)
    for k in range(2):
        print(open('/kaggle/working/dk_%s_%d_log.txt' % (name, k)).read()[-300:], flush=True)
print('DKLITE_DONE minutes %.1f' % ((time.time() - T0) / 60), flush=True)
'''
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}
compile(code, SLUG, "exec")
out = ROOT / "kaggle_push" / SLUG
out.mkdir(parents=True, exist_ok=True)
(out / (SLUG + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": SLUG + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["dkim54/rsna-knee-models-v11i", HOLD], "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out)

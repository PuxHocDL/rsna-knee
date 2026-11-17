"""dkim54/rsna-knee-models-v11i (10 ConvNeXt-S 384, layout v2128, attention pool, mirror TTA; public dataset with its own
offline inference library): MODE eval = predictions on the 450 clean train studies (58 gold + 392 zoliseress reference) through
a scratch train root; MODE solo = the test-set submission (dkim54's predict(), unchanged).
usage: python build_dk_eval.py <owner> eval|solo <slug> [holdout_ids_dataset]"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, MODE, SLUG = sys.argv[1:4]
HOLD = sys.argv[4] if len(sys.argv) > 4 else None
MDS = "dkim54/rsna-knee-models-v11i"
head = r'''import csv, glob, os, sys, time
from pathlib import Path
T0 = time.time()
COMP = Path(sorted(glob.glob('/kaggle/input/**/test.csv', recursive=True), key=len)[0]).parent
MODELS = Path(sorted(glob.glob('/kaggle/input/**/models.json', recursive=True), key=len)[0]).parent
CODE = Path(sorted(glob.glob(str(MODELS) + '/**/infer_lib.py', recursive=True), key=len)[0]).parent
sys.path.insert(0, str(CODE))
import infer_lib
print('COMP', COMP, 'MODELS', MODELS, 'CODE', CODE, flush=True)
'''
eval_cell = r'''# scratch train root: gold + reference studies only
import pandas as pd
tr = pd.read_csv(COMP / 'train.csv', dtype={'StudyInstanceUID': str})
LABELS = [c for c in tr.columns if c not in ('StudyInstanceUID', 'Report')]
gold_ids = tr[tr[LABELS].notna().all(axis=1)]['StudyInstanceUID'].tolist()
hold = sorted(glob.glob('/kaggle/input/**/holdout_ref392_ids.csv', recursive=True))
ref_ids = pd.read_csv(hold[0]).iloc[:, 0].astype(str).tolist()
IDS = sorted(set(gold_ids) | set(ref_ids))
FAKE = Path('/tmp/eval_root'); FAKE.mkdir(parents=True, exist_ok=True)
tr[tr['StudyInstanceUID'].isin(set(IDS))][['StudyInstanceUID']].to_csv(FAKE / 'train.csv', index=False)
ser = pd.read_csv(COMP / 'train_series.csv', dtype=str)
ser[ser['StudyInstanceUID'].isin(set(IDS))].to_csv(FAKE / 'train_series.csv', index=False)
os.makedirs(FAKE / 'train_series', exist_ok=True)
for u in IDS:
    src = COMP / 'train_series' / u
    if src.exists() and not (FAKE / 'train_series' / u).exists():
        os.symlink(src, FAKE / 'train_series' / u)
print('studies', len(IDS), flush=True)
infer_lib.predict(str(FAKE), str(MODELS), '/kaggle/working/dk_ref450.csv', split='train')
print('DK_EVAL_DONE minutes %.1f' % ((time.time() - T0) / 60), flush=True)
'''
solo_cell = r'''infer_lib.predict(str(COMP), str(MODELS), '/kaggle/working/submission.csv', split='test')
import pandas as pd
s = pd.read_csv('/kaggle/working/submission.csv')
print('DK_SOLO_DONE rows', len(s), 'nan', int(s.isna().sum().sum()), 'minutes %.1f' % ((time.time() - T0) / 60), flush=True)
'''


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


cells = [code(head), code(eval_cell if MODE == "eval" else solo_cell)]
for c in cells:
    compile("".join(c["source"]), "cell", "exec")
out = ROOT / "kaggle_push" / SLUG
out.mkdir(parents=True, exist_ok=True)
(out / (SLUG + ".ipynb")).write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                 "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4},
                                                indent=1), encoding="utf-8")
meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": SLUG + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [MDS] + ([HOLD] if HOLD else []), "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out)

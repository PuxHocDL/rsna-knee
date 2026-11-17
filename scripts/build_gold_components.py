"""Score every component of the G2 notebook on the 58 gold train studies (no submission).

rsna-g2-s6fall-rerank-base runs unchanged (Raptor-384, Raptor-224, R7 flip, s6fall) on a scratch root laid out like the test
split (test.csv, test_series.csv, test_series/<uid>/... as file symlinks, sample_submission.csv) holding the gold studies.
Patches: both Raptor scripts' find_test_root(), the compact stage's DATA, and s6fall's find_mount for the competition slug
return that root (env GOLD_ROOT). Outputs: raptor_raw.csv, raptor224_sub.csv, ours_sub.csv (R7), s6fall_sub.csv, and the
G2 blend itself (submission.csv)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
g2 = json.loads((ROOT / "rsna-g2-s6fall-rerank-base.ipynb").read_text(encoding="utf-8"))
code = "".join(g2["cells"][1]["source"])

PRE = '''# [gold] scratch root shaped like the test split, holding the 58 gold train studies
import os, csv
from pathlib import Path
import pandas as pd
_comp = next(Path(p) for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection',
                                '/kaggle/input/rsna-knee-abnormality-detection') if Path(p, 'train.csv').is_file())
_tr = pd.read_csv(_comp / 'train.csv', dtype={'StudyInstanceUID': str})
_lab = [c for c in _tr.columns if c not in ('StudyInstanceUID', 'Report')]
_gold = _tr[_tr[_lab].notna().all(axis=1)]['StudyInstanceUID'].tolist()
assert len(_gold) == 58, len(_gold)
GOLD_ROOT = Path('/tmp/gold_root')
GOLD_ROOT.mkdir(parents=True, exist_ok=True)
pd.DataFrame({'StudyInstanceUID': _gold}).to_csv(GOLD_ROOT / 'test.csv', index=False)
_ser = pd.read_csv(_comp / 'train_series.csv', dtype=str)
_ser[_ser['StudyInstanceUID'].isin(set(_gold))].to_csv(GOLD_ROOT / 'test_series.csv', index=False)
_hdr = list(pd.read_csv(_comp / 'sample_submission.csv', nrows=1).columns)
with open(GOLD_ROOT / 'sample_submission.csv', 'w', newline='', encoding='utf-8') as _fh:
    _w = csv.writer(_fh); _w.writerow(_hdr)
    for _u in _gold:
        _w.writerow([_u] + [0.5] * (len(_hdr) - 1))
_n = 0
for _u in _gold:
    for _dp, _dn, _fn in os.walk(_comp / 'train_series' / _u):
        _rel = Path(_dp).relative_to(_comp / 'train_series')
        (GOLD_ROOT / 'test_series' / _rel).mkdir(parents=True, exist_ok=True)
        for _f in _fn:
            if not (GOLD_ROOT / 'test_series' / _rel / _f).exists():
                os.symlink(Path(_dp) / _f, GOLD_ROOT / 'test_series' / _rel / _f)
            _n += 1
os.environ['GOLD_ROOT'] = str(GOLD_ROOT)
print('GOLD_ROOT', GOLD_ROOT, 'studies', len(_gold), 'files', _n, flush=True)

'''
FTR = "def find_test_root():\n"
assert code.count(FTR) == 2
code = code.replace(FTR, FTR + "    if os.environ.get('GOLD_ROOT'):\n        return os.environ['GOLD_ROOT']   # [gold]\n")
D = "DATA = find_mount(COMPETITION, 'test.csv')\n"
assert code.count(D) == 1
code = code.replace(D, D + "DATA = Path(os.environ['GOLD_ROOT'])   # [gold]\n", 1)
X = "    exec(compile(_src, '/tmp/s6fall_solo.py', 'exec'), _mod.__dict__)\n"
assert code.count(X) == 1
code = code.replace(X, X + "    _fm = _mod.find_mount   # [gold] the competition slug resolves to the gold root\n"
                           "    _mod.find_mount = lambda slug, marker, _o=_fm: Path(os.environ['GOLD_ROOT']) if slug == _mod.COMP else _o(slug, marker)\n", 1)
code = PRE + code
compile(code, "gold", "exec")

slug = "rsna-gold-components"
out = ROOT / "kaggle_push" / slug
out.mkdir(parents=True, exist_ok=True)
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# G2 components on the 58 gold studies (no submission)\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 4}
(out / (slug + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
g2_meta = json.loads((ROOT / "kaggle_push" / "rsna-g2-s6fall-rerank-base" / "kernel-metadata.json").read_text(encoding="utf-8"))
meta = {"id": "nguynxunphc662/" + slug, "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": g2_meta["dataset_sources"], "competition_sources": ["rsna-knee-abnormality-detection"],
        "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out, meta["dataset_sources"])

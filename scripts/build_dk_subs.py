"""Submission variant: an existing notebook (G6RM) + the public dkim54/rsna-knee-models-v11i model (DK: 10 ConvNeXt-S 384,
own v2128 pipeline, mirror TTA; LB 0.952 alone per its team) as one more leg, run last.

The DK stage runs dkim54's own infer_lib.predict() unchanged in a subprocess on the test DICOMs, starts only if DK_NEED_MIN
minutes remain, is killed at 8.75 h after the notebook start, and is blended only if it scored >= 97 % of the studies (its
fallback rows are 0.5 in every column); any failure keeps the base notebook's submission unchanged (backed up first).
final = (1 - W_DK) x rank(base submission) + W_DK x rank(DK).

usage: python build_dk_subs.py <owner> <base_dir> <slug> <w_dk> <need_min> <title> [PROBE_COL]"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, BASE, SLUG, W_DK, NEED, TITLE = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
PROBE_COL = sys.argv[7] if len(sys.argv) > 7 else ""
MDS = "dkim54/rsna-knee-models-v11i"
src_dir = ROOT / "kaggle_push" / BASE
nb = json.loads((src_dir / (BASE + ".ipynb")).read_text(encoding="utf-8"))
md = "".join(nb["cells"][0]["source"])
code = "".join(nb["cells"][1]["source"])
for v in ("DATA = find_mount(COMPETITION, 'test.csv')", "NB_T0 = time.time()"):
    assert v in code, v

STAGE = r'''

# ==============================================================================
# Stage 6 [dk]: public dkim54 v11i model as one more leg (run last; any failure keeps the submission above)
# ==============================================================================
import glob as _dg, os as _do, shutil as _ds, subprocess as _dp, sys as _dsys, time as _dt
import numpy as _dn, pandas as _dd
W_DK = %r
DK_NEED_MIN = %r
PROBE_COL = %r   # runtime probe: DK skipped -> this column becomes a constant (visible as a lower LB score)
DK_USED = False
_d_left = 9 * 3600 - (_dt.time() - NB_T0)
print('DK_STAGE %%.0f min left' %% (_d_left / 60), flush=True)
_d_bak = WORK / '_before_dk_submission.csv'
_ds.copyfile(WORK / 'submission.csv', _d_bak)
try:
    if _d_left < DK_NEED_MIN * 60:
        raise RuntimeError('not enough time: %%.0f min left' %% (_d_left / 60))
    _d_models = sorted({_do.path.dirname(h) for h in _dg.glob('/kaggle/input/**/models.json', recursive=True)
                        if _do.path.exists(_do.path.join(_do.path.dirname(h), 'v2128r3_f0.pt'))})
    assert len(_d_models) == 1, ('DK models', _d_models)
    _d_code = sorted(_dg.glob(_d_models[0] + '/**/infer_lib.py', recursive=True), key=len)
    assert _d_code, 'DK infer_lib.py not found'
    _d_out = TMP / 'dk_submission.csv'
    _d_script = ("import sys; sys.path.insert(0, %%r); import infer_lib; infer_lib.predict(%%r, %%r, %%r, split='test')"
                 %% (_do.path.dirname(_d_code[0]), str(DATA), _d_models[0], str(_d_out)))
    _dt0 = _dt.time()
    with open(WORK / 'dk_log.txt', 'w') as _dlog:
        _d_rc = _dp.run([_dsys.executable, '-u', '-c', _d_script], stdout=_dlog, stderr=_dp.STDOUT,
                        timeout=max(60, NB_T0 + 8.75 * 3600 - _dt.time())).returncode
    print(open(WORK / 'dk_log.txt').read()[-1500:], flush=True)
    if _d_rc != 0:
        raise RuntimeError('DK exit code %%d' %% _d_rc)
    _base = _dd.read_csv(_d_bak, dtype={'StudyInstanceUID': str})
    _labels = [c for c in _base.columns if c != 'StudyInstanceUID']
    _dk = _dd.read_csv(_d_out, dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(_base['StudyInstanceUID'])[_labels]
    _ok = _dn.isfinite(_dk.values).all(axis=1) & ~(_dk.values == 0.5).all(axis=1)
    if _ok.sum() < 0.97 * len(_base):
        raise RuntimeError('DK scored only %%d of %%d studies' %% (_ok.sum(), len(_base)))
    _out = _base.copy()
    for _c in _labels:
        _r = _dn.full(len(_base), _dn.nan)
        _r[_ok] = _dd.Series(_dk[_c].values[_ok]).rank(method='average', pct=True).values
        _v = _dd.Series(_base[_c].values.astype(_dn.float64)).rank(method='average', pct=True).values
        _out[_c] = _dn.where(_ok, (1.0 - W_DK) * _v + W_DK * _r, _v)
    assert _dn.isfinite(_out[_labels].values).all() and ((_out[_labels].values >= 0) & (_out[_labels].values <= 1)).all()
    assert _out['StudyInstanceUID'].tolist() == _base['StudyInstanceUID'].tolist()
    _out.to_csv(WORK / 'submission.csv', index=False)
    DK_USED = True
    _rho = {c: round(float(_dd.Series(_base[c].values[_ok]).corr(_dd.Series(_dk[c].values[_ok]), method='spearman')), 3)
            for c in _labels} if _ok.sum() >= 3 else {}
    print('DK_BLEND_DONE scored %%d of %%d in %%.0f s, W_DK %%.3f; spearman(base, DK) %%s' %% (_ok.sum(), len(_base), _dt.time() - _dt0, W_DK, _rho), flush=True)
except BaseException as _e:
    _ds.copyfile(_d_bak, WORK / 'submission.csv')
    print('DK_BLEND_SKIPPED (base submission kept):', repr(_e)[:500], flush=True)
    if PROBE_COL:
        _pb = _dd.read_csv(WORK / 'submission.csv', dtype={'StudyInstanceUID': str})
        _pb[PROBE_COL] = 0.5
        _pb.to_csv(WORK / 'submission.csv', index=False)
        print('DK_PROBE: DK skipped, column', PROBE_COL, 'set to 0.5', flush=True)
print('DK_FINAL', DK_USED, 'total minutes %%.1f' %% ((_dt.time() - NB_T0) / 60), flush=True)
''' % (W_DK, NEED, PROBE_COL)
new = code + STAGE
compile(new, SLUG, "exec")
out = ROOT / "kaggle_push" / SLUG
out.mkdir(parents=True, exist_ok=True)
nb2 = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# " + TITLE + "\n\n" + md]},
                 {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": new.splitlines(keepends=True)}],
       "metadata": nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 4}
(out / (SLUG + ".ipynb")).write_text(json.dumps(nb2, indent=1), encoding="utf-8")
meta0 = json.loads((src_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
meta = dict(meta0, id=f"{OWNER}/{SLUG}", title=SLUG, code_file=SLUG + ".ipynb", dataset_sources=meta0["dataset_sources"] + [MDS])
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out, meta["dataset_sources"])

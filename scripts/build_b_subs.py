"""Submission variants: an existing notebook + our compact model(s) B as one more equal leg, run last.

The B stage reuses the notebook's R6 setup (heliosli's code, wheels and worker sources: SRC, CODE, CODE_SRC, DATA, TMP) and runs
compact_infer on the test DICOMs with the B training configuration (px256 v4 views, crop 224, 3 slices, 3.5 mm). It starts only
if B_NEED_MIN minutes remain, stops at B_DEADLINE_H after the notebook start, and is blended only if it scored >= 97 % of the
studies; any failure keeps the base notebook's submission unchanged (backed up first).
final = (1 - W_B) x rank(base submission) + W_B x rank(B), W_B fixed by the leg count (G6RM: 3 legs -> B is the 4th, 1/4;
G7: 2 legs -> 1/3). Studies B did not score keep their base rank.

usage: python build_b_subs.py <owner> <models_dataset> <arm> <base_dir> <slug> <w_b> <title>"""
import json, os, sys
from zoom_patch import worker_cell
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, MDS, ARM, BASE, SLUG, W_B, TITLE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6]), sys.argv[7]
PROBE_COL = sys.argv[8] if len(sys.argv) > 8 else ""   # runtime probe: if the B stage is skipped, this column is set to 0.5
ZOOM = float(os.environ.get("ZOOM", "1"))                      # >1: zoom-specialist B worker (zoom_patch.py)
B_COLS = [c for c in os.environ.get("B_COLS", "").split(",") if c]   # blend only these findings (empty = all)
ZW = worker_cell(ZOOM, indent="    ") if ZOOM > 1 else ""
src_dir = ROOT / "kaggle_push" / BASE
nb = json.loads((src_dir / (BASE + ".ipynb")).read_text(encoding="utf-8"))
md = "".join(nb["cells"][0]["source"])
code = "".join(nb["cells"][1]["source"])
for v in ("SRC = TMP / 'probe_src'", "CODE_SRC = find_mount(CODE_SLUG, 'receipt.json')", "DATA = find_mount(COMPETITION, 'test.csv')", "NB_T0 = time.time()"):
    assert v in code, v

STAGE = r'''

# ==============================================================================
# Stage 6 [b]: our compact model(s) B as one more equal leg (run last; any failure keeps the submission above)
# ==============================================================================
import glob as _bg, json as _bj, os as _bo, shutil as _bs, subprocess as _bp, sys as _bsys, time as _bt
import numpy as _bn, pandas as _bd
B_ARM = %r
W_B = %r
PROBE_COL = %r   # runtime probe: B skipped -> this column becomes a constant (visible as a lower LB score)
B_NEED_MIN = 75            # minutes that must remain before the stage starts (measured ~70-80 min for ~1,300 studies)
B_DEADLINE_H = 8.6         # the worker stops here (notebook clock); studies not reached are left unscored
B_USED = False
B_COLS = %r   # empty = every finding
_b_left = 9 * 3600 - (_bt.time() - NB_T0)
print('B_STAGE %%.0f min left' %% (_b_left / 60), flush=True)
_b_bak = WORK / '_before_b_submission.csv'
_bs.copyfile(WORK / 'submission.csv', _b_bak)
try:
    if _b_left < B_NEED_MIN * 60:
        raise RuntimeError('not enough time: %%.0f min left' %% (_b_left / 60))
    _b_hits = []
    for _pat in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/'):
        _b_hits += [h for h in _bg.glob(_pat + B_ARM + '_s*_f*.pt') if 'abnormality-detection' not in h]
    _b_src = sorted({_bo.path.dirname(h) for h in _b_hits})
    assert len(_b_src) == 1, ('B weights', _b_src)
    _b_bundle = TMP / 'bundle_b'
    _b_bundle.mkdir(parents=True, exist_ok=True)
    for _h in sorted(_bg.glob(_b_src[0] + '/' + B_ARM + '_s*_f*.pt')):
        _bs.copy(_h, _b_bundle / _bo.path.basename(_h))
    print('B_MODELS', sorted(p.name for p in _b_bundle.iterdir()), flush=True)
    _b_args = ['--backbone', 'tf_efficientnetv2_s.in21k_ft_in1k', '--size', '256', '--crop', '224', '--eval-stride', '3', '--views', 'v4',
               '--slices', '3', '--phys-mm', '3.5', '--anchor-weights', str(CODE_SRC / 'anatomy.pt')]
    _b_env = dict(_bo.environ, PYTHONPATH=str(CODE) + _bo.pathsep + _bo.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
    _B_WORKER = 'compact_infer.py'
%s    _b_pred = TMP / 'prediction_b'
    _bt0 = _bt.time()
    _b_rc = _bp.run([_bsys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'test',
                     '--code', str(CODE), '--bundle', str(_b_bundle), '--arm', B_ARM, '--output', str(_b_pred),
                     '--amp', 'auto', '--loader-workers', '2', '--deadline', str(NB_T0 + B_DEADLINE_H * 3600),
                     '--worker', _B_WORKER, '--worker-args', _bj.dumps(_b_args),
                     '--max-fallback-fraction', '1.0'], env=_b_env, timeout=max(60, NB_T0 + 8.75 * 3600 - _bt.time())).returncode
    for _p in sorted(_b_pred.glob('worker*/log.txt')):
        _bs.copyfile(_p, WORK / ('b_' + _p.parent.name + '_log.txt'))
    if _b_rc != 0:
        raise RuntimeError('B controller exit code %%d' %% _b_rc)
    _b_rec = _bj.loads((_b_pred / 'submission_receipt.json').read_text())
    _b_fail = set(_b_rec.get('fallback_reasons', {}))
    _base = _bd.read_csv(_b_bak, dtype={'StudyInstanceUID': str})
    _labels = [c for c in _base.columns if c != 'StudyInstanceUID']
    _bb = _bd.read_csv(_b_pred / 'submission.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(_base['StudyInstanceUID'])[_labels]
    _ok = _bn.isfinite(_bb.values).all(axis=1) & ~_base['StudyInstanceUID'].isin(_b_fail).values
    if _ok.sum() < 0.97 * len(_base):
        raise RuntimeError('B scored only %%d of %%d studies' %% (_ok.sum(), len(_base)))
    _out = _base.copy()
    for _c in _labels:
        if B_COLS and _c not in B_COLS:
            continue                                   # findings outside the specialist's set keep the base values
        _r = _bn.full(len(_base), _bn.nan)
        _r[_ok] = _bd.Series(_bb[_c].values[_ok]).rank(method='average', pct=True).values
        _v = _bd.Series(_base[_c].values.astype(_bn.float64)).rank(method='average', pct=True).values
        _out[_c] = _bn.where(_ok, (1.0 - W_B) * _v + W_B * _r, _v)
    assert _bn.isfinite(_out[_labels].values).all() and ((_out[_labels].values >= 0) & (_out[_labels].values <= 1)).all()
    assert _out['StudyInstanceUID'].tolist() == _base['StudyInstanceUID'].tolist()
    _out.to_csv(WORK / 'submission.csv', index=False)
    B_USED = True
    _rho = {c: round(float(_bd.Series(_base[c].values[_ok]).corr(_bd.Series(_bb[c].values[_ok]), method='spearman')), 3)
            for c in _labels} if _ok.sum() >= 3 else {}
    print('B_BLEND_DONE scored %%d of %%d in %%.0f s, W_B %%.3f; spearman(base, B) %%s' %% (_ok.sum(), len(_base), _bt.time() - _bt0, W_B, _rho), flush=True)
except BaseException as _e:
    _bs.copyfile(_b_bak, WORK / 'submission.csv')
    print('B_BLEND_SKIPPED (base submission kept):', repr(_e)[:500], flush=True)
    if PROBE_COL:
        _pb = _bd.read_csv(WORK / 'submission.csv', dtype={'StudyInstanceUID': str})
        _pb[PROBE_COL] = 0.5
        _pb.to_csv(WORK / 'submission.csv', index=False)
        print('B_PROBE: B skipped, column', PROBE_COL, 'set to 0.5', flush=True)
try:
    _br = _bj.loads((WORK / 'blend_receipt.json').read_text())
    _br.update({'b_used': B_USED, 'b_arm': B_ARM, 'w_b': W_B if B_USED else 0.0})
    (WORK / 'blend_receipt.json').write_text(_bj.dumps(_br, indent=1))
except Exception as _e:
    print('B receipt unavailable:', repr(_e)[:200], flush=True)
print('B_FINAL', B_USED, 'total minutes %%.1f' %% ((_bt.time() - NB_T0) / 60), flush=True)
''' % (ARM, W_B, PROBE_COL, B_COLS, ZW)
# ZOOM env (>1: zoom-specialist worker) and B_COLS env (comma list: blend only these findings)
# B_STRIDE env (default 3): window stride of the B worker; 1 = every slice (TTA sweep 10-18: +0.0007 ref392, same wall time)
B_STRIDE = os.environ.get("B_STRIDE", "3")
assert STAGE.count("'--eval-stride', '3'") == 1
STAGE = STAGE.replace("'--eval-stride', '3'", "'--eval-stride', '%s'" % B_STRIDE)
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

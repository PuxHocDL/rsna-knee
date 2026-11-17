"""Append a DK stage to an existing submission notebook: the public dkim54/rsna-knee-models-v11i model (dkim54's own
infer_lib.predict(), unchanged) on two halves of the test studies, one process per T4, then a per-finding rank blend of ONLY
the listed columns:  col = (1 - W) x rank(current value) + W x rank(DK);  every other column is left exactly as it was.
The stage starts only if NEED_MIN minutes remain, the DK processes are killed at 8.75 h after NB_T0, and any failure keeps the
submission unchanged (PROBE_COL, if given, is then set to 0.5 so the LB shows the skip).
Evidence for the column choice is offline only (ref392 + gold-58), never the LB.

usage: python build_dk_col.py <base.ipynb> <owner> <slug> <cols comma> <w> <need_min> <title> [PROBE_COL]"""
import json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE, OWNER, SLUG, COLS, W, NEED, TITLE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), float(sys.argv[5]), int(sys.argv[6]), sys.argv[7]
PROBE_COL = sys.argv[8] if len(sys.argv) > 8 else ""
DK_CFG = os.environ.get("DK_CFG", "full")   # full | full10_notta | lite5 (models.json rewritten in a scratch dir)
base_path = Path(BASE)
nb = json.loads(base_path.read_text(encoding="utf-8"))
meta_path = base_path.parent / "kernel-metadata.json"
base_meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else None

STAGE = r'''# ==============================================================================
# Stage DK [dkcol]: public dkim54 v11i on the listed findings only (two halves, one process per T4)
# ==============================================================================
import glob as _kg, os as _ko, shutil as _ks, subprocess as _kp, sys as _ksys, time as _kt
import numpy as _kn, pandas as _kd
DK_COLS = %r
DK_W = %r
DK_NEED_MIN = %r
DK_PROBE_COL = %r
DK_USED = False
_k_left = NB_T0 + 9 * 3600 - _kt.time()
print('DKCOL_STAGE %%.0f min left' %% (_k_left / 60), flush=True)
_k_bak = WORK / '_before_dkcol_submission.csv'
_ks.copyfile(WORK / 'submission.csv', _k_bak)
try:
    if _k_left < DK_NEED_MIN * 60:
        raise RuntimeError('not enough time: %%.0f min left' %% (_k_left / 60))
    _k_mdirs = sorted({_ko.path.dirname(h) for h in _kg.glob('/kaggle/input/**/models.json', recursive=True)
                       if _ko.path.exists(_ko.path.join(_ko.path.dirname(h), 'v2128r3_f0.pt'))})
    assert len(_k_mdirs) == 1, ('DK models', _k_mdirs)
    _k_code = sorted(_kg.glob(_k_mdirs[0] + '/**/infer_lib.py', recursive=True), key=len)[0]
    DK_CFG = %r
    if DK_CFG != 'full':
        import json as _kj
        _spec = _kj.load(open(_ko.path.join(_k_mdirs[0], 'models.json')))
        _ms = [dict(m, tta=False) for m in _spec['models'] if DK_CFG != 'lite5' or m['weights'].startswith('v2128r3_f')]
        _md = TMP / ('dk_models_' + DK_CFG)
        _md.mkdir(parents=True, exist_ok=True)
        for _m in _ms:
            if not (_md / _m['weights']).exists():
                _ko.symlink(_ko.path.join(_k_mdirs[0], _m['weights']), _md / _m['weights'])
        _kj.dump(dict(_spec, models=_ms), open(_md / 'models.json', 'w'))
        print('DK_CFG', DK_CFG, [m['weights'] for m in _ms], 'tta', any(m.get('tta') for m in _ms), flush=True)
        _k_mdirs = [str(_md)]
    _k_test = _kd.read_csv(DATA / 'test.csv', dtype={'StudyInstanceUID': str})
    _k_ids = _k_test['StudyInstanceUID'].tolist()
    _k_procs = []
    _kt0 = _kt.time()
    for _k, _part in enumerate([_k_ids[0::2], _k_ids[1::2]]):
        _root = TMP / ('dkcol_root%%d' %% _k)
        _root.mkdir(parents=True, exist_ok=True)
        _k_test[_k_test['StudyInstanceUID'].isin(set(_part))].to_csv(_root / 'test.csv', index=False)
        _ks.copyfile(DATA / 'test_series.csv', _root / 'test_series.csv')
        if (DATA / 'sample_submission.csv').exists():
            _ks.copyfile(DATA / 'sample_submission.csv', _root / 'sample_submission.csv')
        if not (_root / 'test_series').exists():
            _ko.symlink(DATA / 'test_series', _root / 'test_series')
        _script = ("import sys; sys.path.insert(0, %%r); import infer_lib; infer_lib.predict(%%r, %%r, %%r, split='test', workers=2)"
                   %% (_ko.path.dirname(_k_code), str(_root), _k_mdirs[0], str(TMP / ('dkcol_%%d.csv' %% _k))))
        _log = open(WORK / ('dkcol_%%d_log.txt' %% _k), 'w')
        _k_procs.append((_kp.Popen([_ksys.executable, '-u', '-c', _script], stdout=_log, stderr=_kp.STDOUT,
                                   env=dict(_ko.environ, CUDA_VISIBLE_DEVICES=str(_k))), _log))
    _k_deadline = NB_T0 + 8.75 * 3600
    for _p, _log in _k_procs:
        try:
            _p.wait(timeout=max(1, _k_deadline - _kt.time()))
        except _kp.TimeoutExpired:
            pass
    for _p, _log in _k_procs:
        if _p.poll() is None:
            _p.kill()
        _log.close()
    _rcs = [_p.returncode for _p, _ in _k_procs]
    for _k in range(2):
        print(open(WORK / ('dkcol_%%d_log.txt' %% _k)).read()[-400:], flush=True)
    if any(_r != 0 for _r in _rcs):
        raise RuntimeError('DK processes did not finish: rc %%s after %%.0f min' %% (_rcs, (_kt.time() - _kt0) / 60))
    _base = _kd.read_csv(_k_bak, dtype={'StudyInstanceUID': str})
    _labels = [c for c in _base.columns if c != 'StudyInstanceUID']
    _dk = _kd.concat([_kd.read_csv(TMP / ('dkcol_%%d.csv' %% _k), dtype={'StudyInstanceUID': str}) for _k in range(2)])
    _dk = _dk.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').reindex(_base['StudyInstanceUID'])[_labels]
    _ok = _kn.isfinite(_dk.values).all(axis=1) & ~(_dk.values == 0.5).all(axis=1)
    if _ok.sum() < 0.97 * len(_base):
        raise RuntimeError('DK scored only %%d of %%d studies' %% (_ok.sum(), len(_base)))
    _out = _base.copy()
    for _c in DK_COLS:
        _v = _kd.Series(_base[_c].values.astype(_kn.float64)).rank(method='average', pct=True).values
        _r = _kn.full(len(_base), _kn.nan)
        _r[_ok] = _kd.Series(_dk[_c].values[_ok]).rank(method='average', pct=True).values
        _out[_c] = _kn.where(_ok, (1.0 - DK_W) * _v + DK_W * _r, _v)
    assert _kn.isfinite(_out[_labels].values).all() and ((_out[_labels].values >= 0) & (_out[_labels].values <= 1)).all()
    assert all((_out[c].values == _base[c].values).all() for c in _labels if c not in DK_COLS), 'a column outside DK_COLS changed'
    _out.to_csv(WORK / 'submission.csv', index=False)
    DK_USED = True
    print('DKCOL_DONE cfg', DK_CFG, 'cols %%s W %%.2f, DK scored %%d of %%d, DK minutes %%.1f' %% (DK_COLS, DK_W, _ok.sum(), len(_base), (_kt.time() - _kt0) / 60), flush=True)
except BaseException as _e:
    _ks.copyfile(_k_bak, WORK / 'submission.csv')
    print('DKCOL_SKIPPED (previous submission kept):', repr(_e)[:500], flush=True)
    if DK_PROBE_COL:
        _pb = _kd.read_csv(WORK / 'submission.csv', dtype={'StudyInstanceUID': str})
        _pb[DK_PROBE_COL] = 0.5
        _pb.to_csv(WORK / 'submission.csv', index=False)
        print('DKCOL_PROBE: DK skipped, column', DK_PROBE_COL, 'set to 0.5', flush=True)
print('DKCOL_FINAL', DK_USED, 'total minutes %%.1f' %% ((_kt.time() - NB_T0) / 60), flush=True)
''' % (COLS, W, NEED, PROBE_COL, DK_CFG)

cells = [c for c in nb["cells"]]
stage = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": STAGE.splitlines(keepends=True)}
code_all = "".join("".join(c["source"]) for c in cells if c["cell_type"] == "code") + "\n" + STAGE
compile(code_all.replace("%%writefile", "#%%writefile"), SLUG, "exec") if "%%writefile" not in code_all else compile(STAGE, SLUG, "exec")
for c in cells:
    c["outputs"] = [] if c["cell_type"] == "code" else c.get("outputs")
    if c["cell_type"] == "markdown":
        c.pop("outputs", None)
nb2 = dict(nb, cells=cells + [stage])
out = ROOT / "kaggle_push" / SLUG
out.mkdir(parents=True, exist_ok=True)
(out / (SLUG + ".ipynb")).write_text(json.dumps(nb2, indent=1), encoding="utf-8")
meta = dict(base_meta) if base_meta else {"language": "python", "kernel_type": "notebook", "enable_gpu": True, "enable_tpu": False,
                                          "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
                                          "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
for k in ("id_no", "keywords"):
    meta.pop(k, None)
meta.update(id=f"{OWNER}/{SLUG}", title=SLUG, code_file=SLUG + ".ipynb", is_private=True)
ds = [d for d in meta.get("dataset_sources", [])]
if "dkim54/rsna-knee-models-v11i" not in ds:
    ds.append("dkim54/rsna-knee-models-v11i")
meta["dataset_sources"] = ds
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out, meta["dataset_sources"])

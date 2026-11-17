"""Submission B4 + DK: our compact B models (B4, 4 v5h folds) and the public dkim54/rsna-knee-models-v11i model (DK), rank mean
with weight W_DK (0.5 = equal halves; ref392: B4 0.9289, DK 0.9264, equal-half blend 0.9330, P(win vs B4) 100 %).

Order: heliosli's R6 worker setup (cell 6 of blend-mine-in-private-lb-luck-to-all-upvoters, verbatim) -> B4 on the test DICOMs
(compact_infer, B training configuration, eval stride 1) -> DK = dkim54's infer_lib.predict() unchanged, the test studies split
in two halves, one process per T4 (CUDA_VISIBLE_DEVICES), killed at DK_DEADLINE_H -> per-finding rank blend.
If DK fails or is not done in time, the submission is B4 alone and PROBE_COL (if set) becomes 0.5, so the LB shows it.
usage: python build_b4dk.py <owner> <b_models_dataset> <arm> <slug> <w_dk[;Finding=w,...]> <title> [PROBE_COL]
       e.g. "0.5;Fracture=0.7": DK weight 0.5 for every finding except Fracture (0.7)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, MDS, ARM, SLUG, TITLE = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[6]
_w = sys.argv[5].split(";")
W_DK = float(_w[0])
W_COL = {k.strip(): float(v) for k, v in (x.split("=") for x in _w[1:] if x.strip())}
PROBE_COL = sys.argv[7] if len(sys.argv) > 7 else ""
DK_DS = "dkim54/rsna-knee-models-v11i"
nb = json.loads((ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb").read_text(encoding="utf-8"))
setup = "".join(nb["cells"][6]["source"])
assert setup.startswith("ARM = 'E_lrprior'") and "T0 = NB_T0" in setup

cell0 = "# %s\nimport time\nNB_T0 = time.time()\n" % TITLE
cell_b = r'''# 1) B4 on the test studies (eval stride 1); its submission.csv (fallback rows = column means) is kept as b_sub.csv
import glob
B_ARM = %r
hits = []
for pat in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/'):
    hits += [h for h in glob.glob(pat + B_ARM + '_s*_f*.pt') if 'abnormality-detection' not in h]
src = sorted({os.path.dirname(h) for h in hits})
assert len(src) == 1, src
bundle = TMP / 'bundle_b'
bundle.mkdir(parents=True, exist_ok=True)
for h in sorted(glob.glob(src[0] + '/' + B_ARM + '_s*_f*.pt')):
    shutil.copy(h, bundle / os.path.basename(h))
print('B_MODELS', sorted(p.name for p in bundle.iterdir()), flush=True)
b_args = ['--backbone', 'tf_efficientnetv2_s.in21k_ft_in1k', '--size', '256', '--crop', '224', '--eval-stride', '1', '--views', 'v4',
          '--slices', '3', '--phys-mm', '3.5', '--anchor-weights', str(CODE_SRC / 'anatomy.pt')]
env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
pred = TMP / 'prediction_b'
rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'test',
                     '--code', str(CODE), '--bundle', str(bundle), '--arm', B_ARM, '--output', str(pred),
                     '--amp', 'auto', '--loader-workers', '2', '--deadline', str(NB_T0 + 4.0 * 3600),
                     '--worker', 'compact_infer.py', '--worker-args', json.dumps(b_args),
                     '--max-fallback-fraction', '1.0'], env=env).returncode
for p in sorted(pred.glob('worker*/log.txt')):
    shutil.copyfile(p, WORK / ('b_' + p.parent.name + '_log.txt'))
if rc != 0:
    raise RuntimeError('B controller exit code %%d' %% rc)
shutil.copyfile(pred / 'submission.csv', WORK / 'b_sub.csv')
shutil.copyfile(pred / 'submission.csv', WORK / 'submission.csv')      # B4 alone until the blend below replaces it
rec = json.loads((pred / 'submission_receipt.json').read_text())
B_FAIL = set(rec.get('fallback_reasons', {}))
print('B_DONE', json.dumps({k: rec.get(k) for k in ('status', 'rows', 'scored', 'fallback_studies', 'seconds')}),
      'minutes %%.1f' %% ((time.time() - NB_T0) / 60), flush=True)
''' % (ARM,)
cell_dk = r'''# 2) DK (dkim54 infer_lib.predict, unchanged) on two halves of the test studies, one process per GPU; 3) rank blend
import pandas as pd, numpy as np
W_DK = %r
W_COL = %r   # per-finding DK weight overrides
PROBE_COL = %r
DK_DEADLINE_H = 8.7
DK_USED = False
try:
    mdirs = sorted({os.path.dirname(h) for h in glob.glob('/kaggle/input/**/models.json', recursive=True)
                    if os.path.exists(os.path.join(os.path.dirname(h), 'v2128r3_f0.pt'))})
    assert len(mdirs) == 1, ('DK models', mdirs)
    dcode = sorted(glob.glob(mdirs[0] + '/**/infer_lib.py', recursive=True), key=len)[0]
    test = pd.read_csv(DATA / 'test.csv', dtype={'StudyInstanceUID': str})
    ids = test['StudyInstanceUID'].tolist()
    halves = [ids[0::2], ids[1::2]]
    procs = []
    t1 = time.time()
    for k, part in enumerate(halves):
        root = TMP / ('dk_root%%d' %% k)
        root.mkdir(parents=True, exist_ok=True)
        test[test['StudyInstanceUID'].isin(set(part))].to_csv(root / 'test.csv', index=False)
        shutil.copyfile(DATA / 'test_series.csv', root / 'test_series.csv')
        if (DATA / 'sample_submission.csv').exists():
            shutil.copyfile(DATA / 'sample_submission.csv', root / 'sample_submission.csv')
        if not (root / 'test_series').exists():
            os.symlink(DATA / 'test_series', root / 'test_series')
        script = ("import sys; sys.path.insert(0, %%r); import infer_lib; infer_lib.predict(%%r, %%r, %%r, split='test', workers=2)"
                  %% (os.path.dirname(dcode), str(root), mdirs[0], str(TMP / ('dk_%%d.csv' %% k))))
        log = open(WORK / ('dk_%%d_log.txt' %% k), 'w')
        procs.append((subprocess.Popen([sys.executable, '-u', '-c', script], stdout=log, stderr=subprocess.STDOUT,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))), log))
    deadline = NB_T0 + DK_DEADLINE_H * 3600
    for p, log in procs:
        try:
            p.wait(timeout=max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            pass
    for p, log in procs:
        if p.poll() is None:
            p.kill()
        log.close()
    rcs = [p.returncode for p, _ in procs]
    for k in range(2):
        print(open(WORK / ('dk_%%d_log.txt' %% k)).read()[-600:], flush=True)
    if any(r != 0 for r in rcs):
        raise RuntimeError('DK processes did not finish: rc %%s after %%.0f min' %% (rcs, (time.time() - t1) / 60))
    base = pd.read_csv(WORK / 'b_sub.csv', dtype={'StudyInstanceUID': str})
    labels = [c for c in base.columns if c != 'StudyInstanceUID']
    dk = pd.concat([pd.read_csv(TMP / ('dk_%%d.csv' %% k), dtype={'StudyInstanceUID': str}) for k in range(2)])
    dk = dk.drop_duplicates('StudyInstanceUID').set_index('StudyInstanceUID').reindex(base['StudyInstanceUID'])[labels]
    ok_dk = np.isfinite(dk.values).all(axis=1) & ~(dk.values == 0.5).all(axis=1)
    ok_b = ~base['StudyInstanceUID'].isin(B_FAIL).values
    if ok_dk.sum() < 0.97 * len(base):
        raise RuntimeError('DK scored only %%d of %%d studies' %% (ok_dk.sum(), len(base)))
    both = ok_dk & ok_b
    out = base.copy()
    for c in labels:
        rb = pd.Series(base[c].values.astype(np.float64)).rank(method='average', pct=True).values
        rd = np.full(len(base), np.nan)
        rd[ok_dk] = pd.Series(dk[c].values[ok_dk]).rank(method='average', pct=True).values
        w = W_COL.get(c, W_DK)
        out[c] = np.where(both, (1.0 - w) * rb + w * rd, np.where(ok_dk & ~ok_b, rd, rb))
    assert np.isfinite(out[labels].values).all() and ((out[labels].values >= 0) & (out[labels].values <= 1)).all()
    out.to_csv(WORK / 'submission.csv', index=False)
    DK_USED = True
    rho = {c: round(float(pd.Series(base[c].values[both]).corr(pd.Series(dk[c].values[both]), method='spearman')), 3)
           for c in labels} if both.sum() >= 3 else {}
    print('DK_BLEND_DONE DK %%d / B %%d / both %%d of %%d, DK minutes %%.1f, W_DK %%.2f %%s; spearman(B, DK) %%s'
          %% (ok_dk.sum(), ok_b.sum(), both.sum(), len(base), (time.time() - t1) / 60, W_DK, W_COL, rho), flush=True)
except BaseException as e:
    shutil.copyfile(WORK / 'b_sub.csv', WORK / 'submission.csv')
    print('DK_BLEND_SKIPPED (B4 alone kept):', repr(e)[:500], flush=True)
    if PROBE_COL:
        pb = pd.read_csv(WORK / 'submission.csv', dtype={'StudyInstanceUID': str})
        pb[PROBE_COL] = 0.5
        pb.to_csv(WORK / 'submission.csv', index=False)
        print('DK_PROBE: DK skipped, column', PROBE_COL, 'set to 0.5', flush=True)
print('B4DK_FINAL', DK_USED, 'total minutes %%.1f' %% ((time.time() - NB_T0) / 60), flush=True)
''' % (W_DK, W_COL, PROBE_COL)


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


cells = [code(cell0), code(setup), code(cell_b), code(cell_dk)]
for c in cells:
    compile("".join(c["source"]), "cell", "exec")
out = ROOT / "kaggle_push" / SLUG
out.mkdir(parents=True, exist_ok=True)
(out / (SLUG + ".ipynb")).write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                 "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4},
                                                indent=1), encoding="utf-8")
meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": SLUG + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004", MDS, DK_DS],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out)

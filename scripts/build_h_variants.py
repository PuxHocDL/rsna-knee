"""Two submission notebooks that add our EffV2-S (24 ep, SWA) + ConvNeXt-S (C8) leg.

H1 (OAI): rsna-g2-s6fall-rerank-base (0.957: Raptor-384 [OAI] + R7 flip + s6fall, rerank) with USE_R224 = False (G3 read: the
          224 checkpoint adds nothing) and a stage 6/7 appended: our two models, then final = (1 - B_OWN) x rank(stage-5 file)
          + B_OWN x rank(own). The own stage runs only with >= 45 min left and never fails the notebook: on any problem the
          stage-5 submission stays.
H2 (no OAI): R6 flip (0.6) + R7 flip (0.4) as the compact leg, our two models as the second leg at W_OWN; each leg ranked over
          its own scored studies, a study one leg missed takes the other's rank.

usage: python build_h_variants.py <owner> <weights_dataset> <cnx_file>"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, WDS, CNX_FILE = sys.argv[1], sys.argv[2], sys.argv[3]
B_OWN_H1 = 0.10
W_OWN_H2 = 0.20
A_R7_H2 = 0.40
ARMS = [{"file": "effv2s_swa.pt", "arch": "tf_efficientnetv2_s.in21k_ft_in1k", "res": 224, "w": 0.5},
        {"file": CNX_FILE, "arch": "convnext_small.in12k_ft_in1k", "res": 224, "w": 0.5}]
SEP = "# " + "=" * 78 + "\n"

g2 = json.loads((ROOT / "rsna-g2-s6fall-rerank-base.ipynb").read_text(encoding="utf-8"))
g2_md, g2_code = ("".join(c["source"]) for c in g2["cells"])
assert g2_code.count("USE_R224 = True ") == 1 and "_v = pd.Series(_base[_c].values.astype(np.float64)).rank(" in g2_code

# ---- our models' script: nartaa's Raptor-224 inference script with our ARMS (same edit as build_own_r6_blend.py) ----
m = re.search(r"SCRIPT_RAPTOR224_RUN_PY = r'''(.*?)'''\nPath\('/tmp/raptor224_run.py'\)", g2_code, re.S)
script = m.group(1)
script = script[script.index("# %% [code]"):]
script = script.replace(re.search(r"ARMS = \[.*?\n\]\n", script, re.S).group(0), "ARMS = " + json.dumps(ARMS, indent=1) + "\n", 1)
assert script.count('out = "/kaggle/working/raptor224_sub.csv"') == 1
script = script.replace('out = "/kaggle/working/raptor224_sub.csv"', 'out = "/kaggle/working/own_sub.csv"', 1)
anchor = "    _w = np.array([float(a.get(\"w\", 1.0)) for a in ARMS], dtype=np.float64)\n"
assert script.count(anchor) == 1
script = script.replace(anchor, "    for _a, _p in enumerate(arm_probs):   # [own] raw probabilities per arm, for the failure check in the blend\n"
                                "        _d = pd.DataFrame(_p, columns=LAB); _d.insert(0, 'StudyInstanceUID', test_ids)\n"
                                "        _d.to_csv(f'/kaggle/working/own_raw_{_a}.csv', index=False)\n" + anchor, 1)
compile(script, "own_run.py", "exec")
OWN_WRITE = "SCRIPT_OWN = " + repr(script) + "\nPath('/tmp/own_run.py').write_text(SCRIPT_OWN, encoding='utf-8')\n"

# helper used by both blends: percentile rank over a mask, NaN elsewhere
PCT = '''
def _pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = pd.Series(np.asarray(v, dtype=np.float64)[ok]).rank(method='average', pct=True).values
    return r


def _own_ranks(test_ids, labels, n_arms):
    raws = [pd.read_csv(WORK / ('own_raw_%d.csv' % a), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID')
            .reindex(test_ids)[labels] for a in range(n_arms)]
    fail = np.all([(~np.isfinite(r.values)).any(axis=1) | (r.values == 0.5).all(axis=1) for r in raws], axis=0)
    ranks = {c: np.mean([_pct_on(r[c].values, ~fail) for r in raws], axis=0) for c in labels}
    return ranks, fail
'''

# ================================ H1 ================================
h1 = g2_code.replace("USE_R224 = True ", "USE_R224 = False", 1)
h1 += "\n\n" + SEP + "# Stage 6 [h1]: our EffV2-S + ConvNeXt-S (Raptor-224 script, own process); never fatal\n" + SEP + '''import os, sys, time, json, subprocess
import numpy as np, pandas as pd
from pathlib import Path
B_OWN = %r
N_ARMS = %d
OWN_OK = False
''' % (B_OWN_H1, len(ARMS)) + OWN_WRITE + '''_left = 9 * 3600 - (time.time() - NB_T0) - 15 * 60
if _left < 45 * 60:
    print('OWN_SKIPPED only %%.0f min left' %% (_left / 60), flush=True)
else:
    _t = time.time()
    try:
        _rc = subprocess.run([sys.executable, '-u', '/tmp/own_run.py'], timeout=_left).returncode
        OWN_OK = _rc == 0 and all(os.path.isfile('/kaggle/working/own_raw_%%d.csv' %% a) for a in range(N_ARMS))
        print('OWN_STAGE rc=%%d ok=%%s seconds %%.0f' %% (_rc, OWN_OK, time.time() - _t), flush=True)
    except subprocess.TimeoutExpired:
        print('OWN_TIMEOUT after %%.0fs - stage-5 submission kept' %% (time.time() - _t), flush=True)
''' % () + "\n" + SEP + "# Stage 7 [h1]: final = (1 - B_OWN) x rank(stage-5 file) + B_OWN x rank(own), fixed weight\n" + SEP + PCT + '''
OWN_USED = False
try:
    if not OWN_OK:
        raise RuntimeError('own stage did not complete')
    _base = pd.read_csv(WORK / 'submission.csv', dtype={'StudyInstanceUID': str})
    _labels = [c for c in _base.columns if c != 'StudyInstanceUID']
    _ids = _base['StudyInstanceUID'].tolist()
    _ro, _ofail = _own_ranks(_ids, _labels, N_ARMS)
    if _ofail.sum() > 0.1 * len(_ids):
        raise RuntimeError('own stage failed on %d of %d studies' % (_ofail.sum(), len(_ids)))
    _out = _base.copy()
    for _c in _labels:
        _v = pd.Series(_base[_c].values.astype(np.float64)).rank(method='average', pct=True).values
        _out[_c] = np.where(_ofail, _v, (1.0 - B_OWN) * _v + B_OWN * np.nan_to_num(_ro[_c], nan=0.5))
    assert np.isfinite(_out[_labels].values).all() and ((_out[_labels].values >= 0) & (_out[_labels].values <= 1)).all()
    _out.to_csv(WORK / 'submission.csv', index=False)
    OWN_USED = True
    print('OWN_BLEND_DONE own failed %d of %d' % (_ofail.sum(), len(_ids)), flush=True)
except BaseException as _e:
    print('OWN_BLEND_SKIPPED (stage-5 submission kept):', repr(_e)[:400], flush=True)
try:
    _br = json.loads((WORK / 'blend_receipt.json').read_text())
    _br.update({'own_used': OWN_USED, 'b_own': B_OWN if OWN_USED else 0.0})
    (WORK / 'blend_receipt.json').write_text(json.dumps(_br, indent=1))
    print('FINAL_RECEIPT_H1', json.dumps({k: _br.get(k) for k in ('r224_used', 'r7_flip_tta', 'r7_only', 's6fall_used', 'own_used', 'b_own')}), flush=True)
except Exception as _e:
    print('FINAL_RECEIPT unavailable:', repr(_e)[:200], flush=True)
display(pd.read_csv(WORK / 'submission.csv').head())
'''
compile(h1, "h1", "exec")

# ================================ H2 ================================
h958 = json.loads((ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb").read_text(encoding="utf-8"))
r6_setup = "".join(h958["cells"][6]["source"])
r6_run = "".join(h958["cells"][7]["source"])
assert "C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'" in r6_setup and "FLIP_TTA = True" in r6_setup
c2 = g2_code.index(SEP + "# Notebook code cell 2\n" + SEP) + len(SEP + "# Notebook code cell 2\n" + SEP)
c3h = g2_code.index(SEP + "# Notebook code cell 3\n" + SEP)
r7_setup = g2_code[c2:c3h]
c3 = c3h + len(SEP + "# Notebook code cell 3\n" + SEP)
r7_run = g2_code[c3:g2_code.index(SEP + "# Write /tmp/raptor224_run.py\n")]
assert "C_SLUG = 'rsna-knee-compact-r7cnxfsss'" in r7_setup and "rc = _run_r7(WORKER_ARGS)" in r7_run

h2_cells = []
h2_cells.append('''# Stage 1 [h2]: our two models (Raptor-224 script, own process)
import os, sys, time, json, shutil, subprocess
from pathlib import Path
NB_T0 = time.time()
R7_FLIP_TTA = True
W_OWN = %r
A_R7 = %r
N_ARMS = %d
''' % (W_OWN_H2, A_R7_H2, len(ARMS)) + OWN_WRITE + '''rc = subprocess.run([sys.executable, '-u', '/tmp/own_run.py'], timeout=3 * 3600).returncode
if rc != 0:
    raise RuntimeError('own stage failed with exit code %d' % rc)
OWN_SECONDS = time.time() - NB_T0
print('OWN_STAGE_DONE seconds %.0f' % OWN_SECONDS, flush=True)
''')
h2_cells.append("# Stage 2 [h2]: R6 (heliosli's cells, unchanged)\n" + r6_setup)
h2_cells.append(r6_run)
h2_cells.append('''# keep R6's outputs before R7 reuses the same work directories
import csv
def _scored(pred_dir, labels):
    s = set()
    for f in pred_dir.glob('worker*/predictions.csv'):
        with open(f, encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                try:
                    v = [float(r[L]) for L in labels]
                except (KeyError, ValueError, TypeError):
                    continue
                if all(0. <= x <= 1. for x in v):
                    s.add(r['StudyInstanceUID'])
    return s
LABELS = list(pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)[1:]
R6_SCORED = _scored(pred, LABELS)
shutil.copyfile(WORK / 'ours_sub.csv', WORK / 'r6_sub.csv')
shutil.rmtree(pred, ignore_errors=True); shutil.rmtree(BUNDLE, ignore_errors=True)
print('R6_KEPT scored', len(R6_SCORED), flush=True)
''')
h2_cells.append("# Stage 3 [h2]: R7 with flip TTA (cells of rsna-g2-s6fall-rerank-base, unchanged)\n" + r7_setup)
h2_cells.append(r7_run)
h2_cells.append('''# Stage 4 [h2]: compact leg = (1 - A_R7) R6 + A_R7 R7, final = (1 - W_OWN) compact + W_OWN own (ranks, fixed weights)
import numpy as np
''' + PCT + '''
test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
cols = list(pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
labels = cols[1:]
n = len(test_ids)
R7_SCORED = _scored(pred, labels)
r6 = pd.read_csv(WORK / 'r6_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
r7 = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
r6_fail = np.array([u not in R6_SCORED for u in test_ids]); r7_fail = np.array([u not in R7_SCORED for u in test_ids])
ro, own_fail = _own_ranks(test_ids, labels, N_ARMS)
if (r6_fail & r7_fail).sum() > 0.5 * n:
    raise RuntimeError('compact leg missing on %d of %d studies' % ((r6_fail & r7_fail).sum(), n))
if own_fail.sum() > 0.1 * n:
    raise RuntimeError('own stage failed on %d of %d studies' % (own_fail.sum(), n))
out = np.zeros((n, len(labels)))
for j, c in enumerate(labels):
    a6, a7 = _pct_on(r6[c].values, ~r6_fail), _pct_on(r7[c].values, ~r7_fail)
    comp = (1.0 - A_R7) * a6 + A_R7 * a7
    comp = np.where(r6_fail, a7, np.where(r7_fail, a6, comp))
    cfail = r6_fail & r7_fail
    b = (1.0 - W_OWN) * comp + W_OWN * ro[c]
    b[cfail] = ro[c][cfail]
    b[own_fail] = comp[own_fail]
    b[cfail & own_fail] = 0.5
    out[:, j] = b
sub = pd.DataFrame(out.astype(np.float32), columns=labels)
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)
receipt = {'w_own': W_OWN, 'a_r7': A_R7, 'rows': n, 'r6_failed': int(r6_fail.sum()), 'r7_failed': int(r7_fail.sum()),
           'own_failed': int(own_fail.sum()), 'r7_flip_tta': bool(globals().get('R7_FLIP_ADDED', False)),
           'own_seconds': round(OWN_SECONDS, 1), 'total_seconds': round(time.time() - NB_T0, 1)}
(WORK / 'blend_receipt.json').write_text(json.dumps(receipt, indent=1))
print('NOTEBOOK_COMPLETE', json.dumps(receipt), flush=True)
display(sub.head())
''')
for c in h2_cells:
    compile(c, "h2", "exec")


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}


def write(slug, cells, datasets):
    out = ROOT / "kaggle_push" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / (slug + ".ipynb")).write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
        "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4},
        indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": datasets, "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [],
            "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("built", out)


g2_meta = json.loads((ROOT / "kaggle_push" / "rsna-g2-s6fall-rerank-base" / "kernel-metadata.json").read_text(encoding="utf-8"))
write("rsna-h1-g2-nor224-own", [md("# H1: G2 (rerank) without R224, + our EffV2-S / ConvNeXt-S leg at %.2f\n\n" % B_OWN_H1 + g2_md), code(h1)],
      sorted(set(g2_meta["dataset_sources"]) | {WDS}))
write("rsna-h2-r6-r7-own", [md("# H2 (no OAI): R6 flip %.1f + R7 flip %.1f, + our EffV2-S / ConvNeXt-S leg at %.2f\n\n"
                               "R6 / R7: heliosli's compact ConvNeXt-S bundles and notebook cells, unchanged. Our models run through "
                               "nartaa's public Raptor-224 inference script (dreaddevelopment's Raptor recipe).\n"
                               % (1 - A_R7_H2, A_R7_H2, W_OWN_H2))] + [code(c) for c in h2_cells],
      [WDS, "heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-knee-compact-r7cnxfsss",
       "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"])

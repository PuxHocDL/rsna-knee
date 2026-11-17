"""Submission notebook: rank blend 0.8 x R6 (heliosli, 4 folds + flip TTA + stackers) + 0.1 x EffV2-S + 0.1 x ConvNeXt-S
(our Raptor-geometry models). No OAI anywhere: R6 = report targets (+ fastMRI auxiliary), ours = pilkwang + stevenleehans labels.

Stage 1  our two models through nartaa's Raptor-224 inference script taken verbatim from rsna-g2-s6fall-rerank-base.ipynb,
         with the checkpoint-hash header removed, ARMS replaced by our two checkpoints (arch / res read from the file) and the
         outputs renamed (own_sub.csv + per-arm raw probabilities own_raw_<a>.csv). Same windows as training: 0.8 centre crop of
         the 384 px field, resized to 224, 94 windows, ImageNet normalisation.
Stage 2  R6 = cells 6-7 of heliosli/blend-mine-in-private-lb-luck-to-all-upvoters unchanged (writes ours_sub.csv).
Stage 3  per-finding percentile-rank blend, each model ranked over its own scored studies; a study one side failed on takes
         the other side's rank. Fixed weights (gold-58 read 10-16: 0.9385 vs R6 alone 0.9366, inside the noise).

usage: python build_own_r6_blend.py <owner> <weights_dataset_slug> <cnx_file>"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, WDS, CNX_FILE = sys.argv[1], sys.argv[2], sys.argv[3]
W_OWN = 0.2                     # our two models together; R6 gets 1 - W_OWN
ARMS = [{"file": "effv2s_swa.pt", "arch": "tf_efficientnetv2_s.in21k_ft_in1k", "res": 224, "w": 0.5},
        {"file": CNX_FILE, "arch": "convnext_small.in12k_ft_in1k", "res": 224, "w": 0.5}]

g2 = json.loads((ROOT / "rsna-g2-s6fall-rerank-base.ipynb").read_text(encoding="utf-8"))
g2src = "".join("".join(c["source"]) for c in g2["cells"] if c["cell_type"] == "code")
m = re.search(r"SCRIPT_RAPTOR224_RUN_PY = r'''(.*?)'''\nPath\('/tmp/raptor224_run.py'\)", g2src, re.S)
script = m.group(1)
head_end = script.index("# %% [code]")
assert "Checkpoint hash differs" in script[:head_end]
script = script[head_end:]
old_arms = re.search(r"ARMS = \[.*?\n\]\n", script, re.S).group(0)
script = script.replace(old_arms, "ARMS = " + json.dumps(ARMS, indent=1) + "\n", 1)
assert script.count('out = "/kaggle/working/raptor224_sub.csv"') == 1
script = script.replace('out = "/kaggle/working/raptor224_sub.csv"', 'out = "/kaggle/working/own_sub.csv"', 1)
anchor = "    _w = np.array([float(a.get(\"w\", 1.0)) for a in ARMS], dtype=np.float64)\n"
assert script.count(anchor) == 1
script = script.replace(anchor, "    for _a, _p in enumerate(arm_probs):   # [own] raw probabilities per arm, for the failure check in the blend\n"
                                "        _d = pd.DataFrame(_p, columns=LAB); _d.insert(0, 'StudyInstanceUID', test_ids)\n"
                                "        _d.to_csv(f'/kaggle/working/own_raw_{_a}.csv', index=False)\n" + anchor, 1)
compile(script, "own_run.py", "exec")

h958 = json.loads((ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb").read_text(encoding="utf-8"))
r6_setup = "".join(h958["cells"][6]["source"])
r6_run = "".join(h958["cells"][7]["source"])
assert r6_setup.startswith("ARM = 'E_lrprior'") and "FLIP_TTA = True" in r6_setup and "C_SLUG = 'rsna-knee-r6-frozen-fe055406-w1'" in r6_setup
assert "shutil.copyfile(pred / 'submission.csv', WORK / 'ours_sub.csv')" in r6_run

cell_md = ("# R6 0.8 + own EffV2-S / ConvNeXt-S 0.2 (rank blend, OAI-free)\n\n"
           "R6: heliosli's compact ConvNeXt-S (public bundle rsna-knee-r6-frozen-fe055406-w1), its notebook cells unchanged. "
           "Our two models run through nartaa's public Raptor-224 inference script (dreaddevelopment's Raptor recipe). "
           "Labels behind our models: pilkwang and stevenleehans public LLM report labels.\n")
cell_own = ("# Stage 1: our two models (Raptor-224 inference script, own process so its GPU memory is freed before R6)\n"
            "import os, sys, time, subprocess\nfrom pathlib import Path\nNB_T0 = time.time()\n"
            "SCRIPT_OWN = " + repr(script) + "\n"
            "Path('/tmp/own_run.py').write_text(SCRIPT_OWN, encoding='utf-8')\n"
            "rc = subprocess.run([sys.executable, '-u', '/tmp/own_run.py'], timeout=3 * 3600).returncode\n"
            "if rc != 0:\n    raise RuntimeError('own stage failed with exit code %d' % rc)\n"
            "for f in ['own_sub.csv'] + ['own_raw_%d.csv' % a for a in range(" + str(len(ARMS)) + ")]:\n"
            "    assert os.path.isfile('/kaggle/working/' + f), f + ' missing'\n"
            "OWN_SECONDS = time.time() - NB_T0\nprint('OWN_STAGE_DONE seconds %.0f' % OWN_SECONDS, flush=True)\n")
cell_blend = '''# Stage 3: per-finding percentile-rank blend, fixed weights
import csv
import numpy as np
import pandas as pd
W_OWN = %r
N_ARMS = %d
test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
cols = list(pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
labels = cols[1:]
r6 = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
raws = [pd.read_csv(WORK / ('own_raw_%%d.csv' %% a), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[test_ids, labels]
        for a in range(N_ARMS)]
n = len(test_ids)
own_fail = np.all([(r.values == 0.5).all(axis=1) for r in raws], axis=0)    # the script leaves a failed study at 0.5
scored = set()
for f in pred.glob('worker*/predictions.csv'):
    with open(f, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            try:
                v = [float(r[L]) for L in labels]
            except (KeyError, ValueError, TypeError):
                continue
            if all(0. <= x <= 1. for x in v):
                scored.add(r['StudyInstanceUID'])
r6_fail = np.array([u not in scored for u in test_ids])
if r6_fail.sum() > 0.5 * n:
    raise RuntimeError('R6 did not score %%d of %%d studies' %% (r6_fail.sum(), n))
if own_fail.sum() > 0.1 * n:
    raise RuntimeError('own stage failed on %%d of %%d studies' %% (own_fail.sum(), n))


def pct_on(v, ok):
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = pd.Series(np.asarray(v, dtype=np.float64)[ok]).rank(method='average', pct=True).values
    return r


out = np.zeros((n, len(labels)))
rho = {}
for j, c in enumerate(labels):
    rr = pct_on(r6[c].values, ~r6_fail)
    ro = np.mean([pct_on(r[c].values, ~own_fail) for r in raws], axis=0)
    b = (1.0 - W_OWN) * rr + W_OWN * ro
    b[r6_fail] = ro[r6_fail]
    b[own_fail] = rr[own_fail]
    b[r6_fail & own_fail] = 0.5
    out[:, j] = b
    if n >= 3:
        rho[c] = round(float(pd.Series(r6[c].values).corr(pd.Series(raws[0][c].values), method='spearman')), 3)
sub = pd.DataFrame(out.astype(np.float32), columns=labels)
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)
receipt = {'w_own': W_OWN, 'arms': N_ARMS, 'rows': n, 'r6_failed': int(r6_fail.sum()), 'own_failed': int(own_fail.sum()),
           'own_seconds': round(OWN_SECONDS, 1), 'total_seconds': round(time.time() - NB_T0, 1), 'spearman_r6_vs_effv2': rho}
(WORK / 'blend_receipt.json').write_text(json.dumps(receipt, indent=1))
print('NOTEBOOK_COMPLETE', json.dumps(receipt), flush=True)
display(sub.head())
''' % (W_OWN, len(ARMS))


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


cells = [{"cell_type": "markdown", "metadata": {}, "source": cell_md.splitlines(keepends=True)},
         code(cell_own), code(r6_setup), code(r6_run), code(cell_blend)]
for c in cells[1:]:
    compile("".join(c["source"]), "cell", "exec")
slug = "rsna-r6-own-blend"
out = ROOT / "kaggle_push" / slug
out.mkdir(parents=True, exist_ok=True)
(out / (slug + ".ipynb")).write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
    "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4},
    indent=1), encoding="utf-8")
meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [WDS, "heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out)

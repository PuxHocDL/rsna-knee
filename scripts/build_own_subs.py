"""Three submission notebooks with our self-trained Raptor-geometry models (weights: bloctuat/rsna-tk-own-weights-v2).

O3   : equal rank blend of E24 (EffV2-S, old labels, 24 ep) + E288v3 (EffV2-S 288 px, label v3) + COAT2v3 (CoAtNet-2, label v3)
O5   : equal rank blend of all five (+ C16 ConvNeXt-S old labels, A16v3 EffV2-S label v3)
G6O3 : G6 (0.958: Raptor-384 + R6 flip + s6fall, rerank, no R224) + the O3 trio at B_OWN = 0.10 on the final file; the own stage
       runs only with >= 45 min left and never fails the notebook (on any problem the G6 file stays).
Our models run through nartaa's Raptor-224 inference script (from rsna-g2-s6fall-rerank-base) with ARMS replaced; arch / res are
read from each checkpoint (E288v3 at 288 px). usage: python build_own_subs.py <owner>"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER = sys.argv[1]
WDS = OWNER + "/rsna-tk-own-weights-v2"
SEP = "# " + "=" * 78 + "\n"
ARCH = {"effv2s_swa.pt": ("tf_efficientnetv2_s.in21k_ft_in1k", 224), "e2r288_swa.pt": ("tf_efficientnetv2_s.in21k_ft_in1k", 288),
        "coat2_swa.pt": ("coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", 224), "cnxs_swa.pt": ("convnext_small.in12k_ft_in1k", 224),
        "e2v3_swa.pt": ("tf_efficientnetv2_s.in21k_ft_in1k", 224)}
O3 = ["effv2s_swa.pt", "e2r288_swa.pt", "coat2_swa.pt"]
O5 = O3 + ["cnxs_swa.pt", "e2v3_swa.pt"]

g2 = json.loads((ROOT / "rsna-g2-s6fall-rerank-base.ipynb").read_text(encoding="utf-8"))
g2_code = "".join(g2["cells"][1]["source"])
base_script = re.search(r"SCRIPT_RAPTOR224_RUN_PY = r'''(.*?)'''\nPath\('/tmp/raptor224_run.py'\)", g2_code, re.S).group(1)
base_script = base_script[base_script.index("# %% [code]"):]


def own_script(files):
    arms = [{"file": f, "arch": ARCH[f][0], "res": ARCH[f][1], "w": 1.0} for f in files]
    s = base_script.replace(re.search(r"ARMS = \[.*?\n\]\n", base_script, re.S).group(0), "ARMS = " + json.dumps(arms, indent=1) + "\n", 1)
    assert s.count('out = "/kaggle/working/raptor224_sub.csv"') == 1
    s = s.replace('out = "/kaggle/working/raptor224_sub.csv"', 'out = "/kaggle/working/own_sub.csv"', 1)
    anchor = "    _w = np.array([float(a.get(\"w\", 1.0)) for a in ARMS], dtype=np.float64)\n"
    assert s.count(anchor) == 1
    s = s.replace(anchor, "    for _a, _p in enumerate(arm_probs):   # [own] raw probabilities per arm (failure check)\n"
                          "        _d = pd.DataFrame(_p, columns=LAB); _d.insert(0, 'StudyInstanceUID', test_ids)\n"
                          "        _d.to_csv(f'/kaggle/working/own_raw_{_a}.csv', index=False)\n" + anchor, 1)
    compile(s, "own_run.py", "exec")
    return "SCRIPT_OWN = " + repr(s) + "\nPath('/tmp/own_run.py').write_text(SCRIPT_OWN, encoding='utf-8')\n"


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


def own_only(files):
    return ('''# Our self-trained models only: equal per-finding rank blend (fixed weights)
import os, sys, time, json, subprocess
import numpy as np, pandas as pd
from pathlib import Path
NB_T0 = time.time()
WORK = Path('/kaggle/working')
N_ARMS = %d
''' % len(files)) + own_script(files) + '''rc = subprocess.run([sys.executable, '-u', '/tmp/own_run.py'], timeout=8 * 3600).returncode
if rc != 0:
    raise RuntimeError('own stage failed with exit code %d' % rc)
''' + PCT + '''
DATA = next(Path(p) for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection')
            if Path(p, 'test.csv').is_file())
test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
cols = list(pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
labels = cols[1:]
ro, fail = _own_ranks(test_ids, labels, N_ARMS)
if fail.sum() > 0.1 * len(test_ids):
    raise RuntimeError('own models failed on %d of %d studies' % (fail.sum(), len(test_ids)))
sub = pd.DataFrame({c: np.nan_to_num(ro[c], nan=0.5) for c in labels})
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert np.isfinite(sub[labels].values).all() and ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)
print('NOTEBOOK_COMPLETE', json.dumps({'arms': N_ARMS, 'rows': len(test_ids), 'failed': int(fail.sum()),
      'seconds': round(time.time() - NB_T0, 1)}), flush=True)
'''


g6 = json.loads((ROOT / "kaggle_push" / "rsna-g6-raptor-r6-s6fall" / "rsna-g6-raptor-r6-s6fall.ipynb").read_text(encoding="utf-8"))
g6_code = "".join(g6["cells"][1]["source"])
g6o3 = g6_code + "\n\n" + SEP + "# Stage 6 [g6o3]: our E24 + E288v3 + COAT2v3 (Raptor-224 script, own process); never fatal\n" + SEP + '''import os, sys, time, json, subprocess
import numpy as np, pandas as pd
from pathlib import Path
B_OWN = 0.10
N_ARMS = 3
OWN_OK = False
''' + own_script(O3) + '''_left = 9 * 3600 - (time.time() - NB_T0) - 15 * 60
if _left < 45 * 60:
    print('OWN_SKIPPED only %.0f min left' % (_left / 60), flush=True)
else:
    _t = time.time()
    try:
        _rc = subprocess.run([sys.executable, '-u', '/tmp/own_run.py'], timeout=_left).returncode
        OWN_OK = _rc == 0 and all(os.path.isfile('/kaggle/working/own_raw_%d.csv' % a) for a in range(N_ARMS))
        print('OWN_STAGE rc=%d ok=%s seconds %.0f' % (_rc, OWN_OK, time.time() - _t), flush=True)
    except subprocess.TimeoutExpired:
        print('OWN_TIMEOUT after %.0fs - G6 submission kept' % (time.time() - _t), flush=True)
''' + "\n" + SEP + "# Stage 7 [g6o3]: final = (1 - B_OWN) x rank(G6 file) + B_OWN x rank(own trio)\n" + SEP + PCT + '''
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
    print('OWN_BLEND_SKIPPED (G6 submission kept):', repr(_e)[:400], flush=True)
try:
    _br = json.loads((WORK / 'blend_receipt.json').read_text())
    _br.update({'own_used': OWN_USED, 'b_own': B_OWN if OWN_USED else 0.0})
    (WORK / 'blend_receipt.json').write_text(json.dumps(_br, indent=1))
    print('FINAL_RECEIPT_G6O3', json.dumps({k: _br.get(k) for k in ('r224_used', 's6fall_used', 'own_used', 'b_own')}), flush=True)
except Exception as _e:
    print('FINAL_RECEIPT unavailable:', repr(_e)[:200], flush=True)
'''
compile(g6o3, "g6o3", "exec")
g6_meta = json.loads((ROOT / "kaggle_push" / "rsna-g6-raptor-r6-s6fall" / "kernel-metadata.json").read_text(encoding="utf-8"))


def write(slug, title_md, code, datasets):
    out = ROOT / "kaggle_push" / slug
    out.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [title_md]},
                    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    compile(code, slug, "exec")
    (out / (slug + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": datasets, "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("built", slug, datasets)


CRED = ("Our models run through nartaa's public Raptor-224 inference script (dreaddevelopment's Raptor recipe). Labels: pilkwang and "
        "stevenleehans public LLM report labels; label v3 mixes them with out-of-fold predictions of heliosli's public R6 bundle.\n")
write("rsna-o3-own-trio", "# O3: E24 + E288v3 + COAT2v3 (our models only, equal rank blend)\n\n" + CRED, own_only(O3), [WDS])
write("rsna-o5-own-all", "# O5: all five of our models (equal rank blend)\n\n" + CRED, own_only(O5), [WDS])
write("rsna-g6o3-g6-own-trio", "# G6O3: G6 + our trio at 0.10\n\n" + CRED, g6o3, sorted(set(g6_meta["dataset_sources"]) | {WDS}))

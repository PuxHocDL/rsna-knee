"""Two LB experiments from discussion 746792 (nartaa's 0.949 write-up and its comments), each one change against a scored entry.

O3M : O3 (E24 + E288v3 + COAT2v3, equal rank) + anatomical mirror TTA on every own model: the plain and the mirrored pass are
      averaged per study (sigmoid probabilities, as nartaa's 0.949 notebook does). Mirror = the 0.949 notebook's _mirror_tri: the
      slot of each window's centre slice decides the axis (sagittal slots 0-1: reverse the 3 slice channels; coronal / axial:
      flip the width), on the uint8 triplet before cropping and normalisation. Compare against O3.
G6R : G6 with nartaa's 224 px checkpoint back in the Raptor slot at 35 % (varshithakolupuri's comment: it moved two blends up
      inside their ties; our G3 read the other way). Compare against G6 (0.958).
usage: python build_disc_subs.py <owner>"""
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER = sys.argv[1]
WDS = OWNER + "/rsna-c2v4-weights"
ARCH = {"effv2s_swa.pt": ("tf_efficientnetv2_s.in21k_ft_in1k", 224), "e2r288_swa.pt": ("tf_efficientnetv2_s.in21k_ft_in1k", 288),
        "coat2_swa.pt": ("coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", 224),
        "coat2v4_swa.pt": ("coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", 224), "coat2v4m_swa.pt": ("coatnet_rmlp_2_rw_224.sw_in12k_ft_in1k", 224)}
O3 = ["coat2v4_swa.pt", "coat2v4m_swa.pt"]

g2 = json.loads((ROOT / "rsna-g2-s6fall-rerank-base.ipynb").read_text(encoding="utf-8"))
g2_code = "".join(g2["cells"][1]["source"])
s = re.search(r"SCRIPT_RAPTOR224_RUN_PY = r'''(.*?)'''\nPath\('/tmp/raptor224_run.py'\)", g2_code, re.S).group(1)
s = s[s.index("# %% [code]"):]
arms = [{"file": f, "arch": ARCH[f][0], "res": ARCH[f][1], "w": 1.0} for f in O3]
s = s.replace(re.search(r"ARMS = \[.*?\n\]\n", s, re.S).group(0), "ARMS = " + json.dumps(arms, indent=1) + "\n", 1)
s = s.replace('out = "/kaggle/working/raptor224_sub.csv"', 'out = "/kaggle/working/own_sub.csv"', 1)
anchor = "    _w = np.array([float(a.get(\"w\", 1.0)) for a in ARMS], dtype=np.float64)\n"
assert s.count(anchor) == 1
s = s.replace(anchor, "    for _a, _p in enumerate(arm_probs):   # [own] raw probabilities per arm (failure check)\n"
                      "        _d = pd.DataFrame(_p, columns=LAB); _d.insert(0, 'StudyInstanceUID', test_ids)\n"
                      "        _d.to_csv(f'/kaggle/working/own_raw_{_a}.csv', index=False)\n" + anchor, 1)
# ---- [mirror] the 0.949 notebook's anatomical mirror, verbatim, + a mirror switch in gpu_windows + the paired average ----
MIRROR = '''
SLOT_SIZES = [26, 22, 18, 12, 18]
SAGITTAL_SLOTS = (0, 1)

def _slot_index(D):
    b = np.cumsum([0] + SLOT_SIZES)
    if b[-1] != D: raise RuntimeError(f'slot sizes sum to {b[-1]}, volume has {D} slices')
    s_ = np.searchsorted(b, np.arange(D), side='right') - 1
    rel = (np.arange(D) - b[s_]) / np.maximum(1, np.array(SLOT_SIZES)[s_] - 1)
    return s_, rel

def _mirror_tri(tri, cs, D):
    s_, _ = _slot_index(D)
    sag = torch.from_numpy(np.isin(s_[np.asarray(cs, dtype=np.int64)], SAGITTAL_SLOTS)).to(tri.device).view(-1, 1, 1, 1)
    return torch.where(sag, tri.flip(1), tri.flip(-1))


'''
gw_def = "def gpu_windows(vol, mask, k, res, dev, norm=NORM):\n"
assert s.count(gw_def) == 1
s = s.replace("_WLUT = {}\n" + gw_def, MIRROR + "_WLUT = {}\ndef gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):\n", 1)
tri_line = "    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8\n"
assert s.count(tri_line) == 1
s = s.replace(tri_line, tri_line + "    if mirror:\n        tri = _mirror_tri(tri, cs, D)\n", 1)
inf_line = "                        arm_probs[a][i] = infer_probs(model, xw, d)\n"
assert s.count(inf_line) == 1
s = s.replace(inf_line, "                        _xm = gpu_windows(vol, mask, K_EVAL, res, d, mirror=True)   # [mirror] paired pass\n"
                        "                        arm_probs[a][i] = 0.5 * (infer_probs(model, xw, d) + infer_probs(model, _xm, d))\n", 1)
compile(s, "own_run_mirror.py", "exec")

o3m = '''# C2V4 solo: our CoAtNet-2 v4h pair (plain + mirror-trained), equal rank blend, each with anatomical mirror TTA
import os, sys, time, json, subprocess
import numpy as np, pandas as pd
from pathlib import Path
NB_T0 = time.time()
WORK = Path('/kaggle/working')
N_ARMS = 2
SCRIPT_OWN = ''' + repr(s) + '''
Path('/tmp/own_run.py').write_text(SCRIPT_OWN, encoding='utf-8')
rc = subprocess.run([sys.executable, '-u', '/tmp/own_run.py'], timeout=8 * 3600).returncode
if rc != 0:
    raise RuntimeError('own stage failed with exit code %d' % rc)
DATA = next(Path(p) for p in ('/kaggle/input/competitions/rsna-knee-abnormality-detection', '/kaggle/input/rsna-knee-abnormality-detection')
            if Path(p, 'test.csv').is_file())
test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
cols = list(pd.read_csv(DATA / 'sample_submission.csv', nrows=1).columns)
labels = cols[1:]
raws = [pd.read_csv(WORK / ('own_raw_%d.csv' % a), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').reindex(test_ids)[labels]
        for a in range(N_ARMS)]
fail = np.all([(~np.isfinite(r.values)).any(axis=1) | (r.values == 0.5).all(axis=1) for r in raws], axis=0)
if fail.sum() > 0.1 * len(test_ids):
    raise RuntimeError('own models failed on %d of %d studies' % (fail.sum(), len(test_ids)))
def pct(v, ok):
    r = np.full(len(v), 0.5)
    r[ok] = pd.Series(np.asarray(v, dtype=np.float64)[ok]).rank(method='average', pct=True).values
    return r
sub = pd.DataFrame({c: np.mean([pct(r[c].values, ~fail) for r in raws], axis=0) for c in labels})
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert np.isfinite(sub[labels].values).all() and ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)
print('NOTEBOOK_COMPLETE', json.dumps({'arms': N_ARMS, 'mirror_tta': True, 'rows': len(test_ids), 'failed': int(fail.sum()),
      'seconds': round(time.time() - NB_T0, 1)}), flush=True)
'''
compile(o3m, "o3m", "exec")

def write(slug, md, code, datasets):
    out = ROOT / "kaggle_push" / slug
    out.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [md]},
                    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    (out / (slug + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": datasets, "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("built", slug)


write('rsna-c2v4-solo', '# C2V4 solo: our CoAtNet-2 v4h pair with anatomical mirror TTA', o3m, [WDS])

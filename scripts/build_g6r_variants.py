"""Two one-change variants of G6R (0.958: Raptor-384 + 224 at 35 % of the Raptor slot + R6 flip + s6fall 1/3, rerank).

G6RM : the 224 px checkpoint also gets nartaa's anatomical mirror TTA (the 384 stage already has it): plain and mirrored passes
       averaged per study, _mirror_tri copied from the 384 script of the same notebook.
G6RW : heliosli's per-finding Raptor weights from blend-mine-in (0.957 -> 0.958 on the same Raptor-slot + R6 structure):
       Synovitis 0.8, Lateral OA 0.2, PF OA 0.6, Baker's 0.35, Medial OA 0.5, the other findings 0.5. s6fall stays at 1/3.
usage: python build_g6r_variants.py <owner>"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER = sys.argv[1]
src_dir = ROOT / "kaggle_push" / "rsna-g6r-g6-with-r224"
nb = json.loads((src_dir / "rsna-g6r-g6-with-r224.ipynb").read_text(encoding="utf-8"))
md = "".join(nb["cells"][0]["source"])
code = "".join(nb["cells"][1]["source"])
assert "USE_R224 = True " in code

# ---------------- G6RM: patch only inside the 224 script string ----------------
head = "SCRIPT_RAPTOR224_RUN_PY = r'''"
a = code.index(head) + len(head)
b = code.index("'''\nPath('/tmp/raptor224_run.py')", a)
s224 = code[a:b]
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
gw = "_WLUT = {}\ndef gpu_windows(vol, mask, k, res, dev, norm=NORM):\n"
assert s224.count(gw) == 1
s224 = s224.replace(gw, MIRROR + "_WLUT = {}\ndef gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):\n", 1)
tri = "    tri = torch.from_numpy(np.ascontiguousarray(vol[np.stack([cs - 1, cs, cs + 1], 1)])).to(dev)   # (k,3,IMG,IMG) uint8\n"
assert s224.count(tri) == 1
s224 = s224.replace(tri, tri + "    if mirror:\n        tri = _mirror_tri(tri, cs, D)   # [g6rm]\n", 1)
inf = "                        arm_probs[a][i] = infer_probs(model, xw, d)\n"
assert s224.count(inf) == 1
s224 = s224.replace(inf, "                        _xm = gpu_windows(vol, mask, K_EVAL, res, d, mirror=True)   # [g6rm] anatomical mirror pass\n"
                         "                        arm_probs[a][i] = 0.5 * (infer_probs(model, xw, d) + infer_probs(model, _xm, d))\n", 1)
compile(s224, "raptor224_run_mirror.py", "exec")
g6rm = code[:a] + s224 + code[b:]
compile(g6rm, "g6rm", "exec")

# ---------------- G6RW: per-finding Raptor weights in stage 3 ----------------
old_w = "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.5, 'PF OA': 0.5}"
assert code.count(old_w) == 1
g6rw = code.replace(old_w, "W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, \"Baker's\": 0.35}   # [g6rw] heliosli blend-mine-in", 1)
compile(g6rw, "g6rw", "exec")

meta0 = json.loads((src_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
for slug, src, title in (("rsna-g6rm-r224-mirror", g6rm, "G6RM: G6R + anatomical mirror TTA on the 224 px checkpoint"),
                         ("rsna-g6rw-perfinding", g6rw, "G6RW: G6R + heliosli's per-finding Raptor weights")):
    out = ROOT / "kaggle_push" / slug
    out.mkdir(parents=True, exist_ok=True)
    nb2 = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["# " + title + "\n\n" + md]},
                     {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}],
           "metadata": nb.get("metadata", {}), "nbformat": 4, "nbformat_minor": 4}
    (out / (slug + ".ipynb")).write_text(json.dumps(nb2, indent=1), encoding="utf-8")
    meta = dict(meta0, id=f"{OWNER}/{slug}", title=slug, code_file=slug + ".ipynb")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("built", slug, meta["dataset_sources"])

"""G6 / G7 from rsna-g2-s6fall-rerank-base (0.957): the compact stage R7 (cells 2-3) is replaced by heliosli's R6 cells (flip TTA,
stackers; bundle rsna-knee-r6-frozen-fe055406-w1), unchanged.  Gold-58 read 10-16: R6 in place of R7 +0.004 (90 % CI > 0).
G6 (OAI): Raptor-384 + R6 + s6fall, USE_R224 = False (G3 read), rerank kept.
G7 (no OAI): R7_ONLY path (no Raptor checkpoints) with R6 as the compact model, s6fall at an equal half (as G4).
usage: python build_g67.py <owner>"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER = sys.argv[1]
SEP = "# " + "=" * 78 + "\n"
g2 = json.loads((ROOT / "rsna-g2-s6fall-rerank-base.ipynb").read_text(encoding="utf-8"))
g2_md, code = ("".join(c["source"]) for c in g2["cells"])
h958 = json.loads((ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb").read_text(encoding="utf-8"))
r6_setup, r6_run = "".join(h958["cells"][6]["source"]), "".join(h958["cells"][7]["source"])
h2 = SEP + "# Notebook code cell 2\n" + SEP
h3 = SEP + "# Notebook code cell 3\n" + SEP
a, b = code.index(h2) + len(h2), code.index(h3)
c, d = b + len(h3), code.index(SEP + "# Write /tmp/raptor224_run.py\n")
assert "C_SLUG = 'rsna-knee-compact-r7cnxfsss'" in code[a:b] and "rc = _run_r7(WORKER_ARGS)" in code[c:d]
base = code[:a] + "# [g6] heliosli's R6 cells in place of the R7 stage (cells unchanged)\n" + r6_setup + "\n\n" + h3 + r6_run + "\n\n" + code[d:]
base = base.replace("USE_R224 = True ", "USE_R224 = False", 1)
g6 = base
g7 = base.replace("R7_ONLY = False", "R7_ONLY = True", 1)
for old, new in (("(2.0 / 3.0) * _v + (1.0 / 3.0) * _r", "0.5 * _v + 0.5 * _r"),
                 ("'s6fall_share': (1.0 / 3.0) if S6F_USED else 0.0", "'s6fall_share': 0.5 if S6F_USED else 0.0")):
    assert g7.count(old) == 1, old
    g7 = g7.replace(old, new, 1)
for s in (g6, g7):
    compile(s, "g", "exec")
    assert "rsna-knee-r6-frozen-fe055406-w1" in s and "C_SLUG = 'rsna-knee-compact-r7cnxfsss'" not in s
g2_meta = json.loads((ROOT / "kaggle_push" / "rsna-g2-s6fall-rerank-base" / "kernel-metadata.json").read_text(encoding="utf-8"))
ds = [x for x in g2_meta["dataset_sources"] if not x.endswith("rsna-knee-compact-r7cnxfsss")] + ["heliosli/rsna-knee-r6-frozen-fe055406-w1"]
for tag, src, desc, dsx in (("g6", g6, "G6: Raptor-384 + R6 flip + s6fall (rerank), no R224", ds),
                            ("g7", g7, "G7 (no OAI): R6 flip + s6fall, equal halves (Raptor stage skipped by R7_ONLY)", ds)):
    slug = "rsna-%s-%s" % (tag, "r6-s6fall" if tag == "g7" else "raptor-r6-s6fall")
    out = ROOT / "kaggle_push" / slug
    out.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [desc + "\n\n" + g2_md]},
                    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    (out / (slug + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": dsx, "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("built", slug, dsx)

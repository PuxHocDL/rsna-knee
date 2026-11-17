import ast
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "rsna-s6-convnext.ipynb"
notebook = json.loads(BASE.read_text(encoding="utf-8"))
source = "".join(notebook["cells"][0]["source"])


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new, 1)


variants = [
    ("G1", "bloctuat", "rsna-g1-s6fall-no-r7flip",
     "Disable R7 flip only; preserve direct blend and S6fall share 1/3."),
    ("G2", "bloctuat", "rsna-g2-s6fall-rerank-base",
     "Rank the base ensemble before S6fall blending; preserve R7 flip."),
    ("G3", "phatdennibba", "rsna-g3-s6fall-no-r224",
     "Disable Raptor224 only; retain Raptor384, R7 flip and S6fall share 1/3."),
    ("G4", "phatdennibba", "rsna-g4-r7-s6fall-half",
     "Run R7 with flip and S6fall at equal shares; skip both Raptor checkpoints."),
]
manifest = []
for tag, owner, slug, description in variants:
    text = source
    if tag == "G1":
        text = replace_once(text, "R7_FLIP_TTA = True", "R7_FLIP_TTA = False")
    elif tag == "G2":
        text = replace_once(
            text, "_v = _base[_c].values.astype(np.float64)",
            "_v = pd.Series(_base[_c].values.astype(np.float64)).rank(method='average', pct=True).values",
        )
    elif tag == "G3":
        text = replace_once(text, "USE_R224 = True", "USE_R224 = False")
    elif tag == "G4":
        text = replace_once(text, "USE_R224 = True", "USE_R224 = False")
        text = replace_once(text, "R7_ONLY = False", "R7_ONLY = True")
        text = replace_once(
            text, "(2.0 / 3.0) * _v + (1.0 / 3.0) * _r",
            "0.5 * _v + 0.5 * _r",
        )
        text = replace_once(
            text, "'s6fall_share': (1.0 / 3.0) if S6F_USED else 0.0",
            "'s6fall_share': 0.5 if S6F_USED else 0.0",
        )
    ast.parse(text)
    nb = copy.deepcopy(notebook)
    nb["cells"] = [
        {"cell_type": "markdown", "metadata": {},
         "source": [f"# {tag}\n{description}\n\n"
                    "Derived from rsna-s6-convnext.ipynb; original embedded credits retained. "
                    "S6fall reference: https://www.kaggle.com/competitions/"
                    "rsna-knee-abnormality-detection/discussion/749359\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {},
         "outputs": [], "source": text.splitlines(keepends=True)},
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    filename = slug + ".ipynb"
    payload = json.dumps(nb, separators=(",", ":"), ensure_ascii=True)
    assert len(payload.encode("utf-8")) < 1_000_000
    (ROOT / filename).write_text(payload, encoding="utf-8")
    folder = ROOT / "kaggle_push" / slug
    folder.mkdir(exist_ok=True)
    (folder / filename).write_text(payload, encoding="utf-8")
    meta = json.loads((ROOT / "kaggle_push" / "rsna-f1-9573-s6fall-r9" /
                       "kernel-metadata.json").read_text(encoding="utf-8-sig"))
    meta.update(id=f"{owner}/{slug}", title=slug, code_file=filename)
    meta["dataset_sources"] = [d for d in meta["dataset_sources"] if "leg-r9all" not in d]
    if tag == "G4":
        meta["dataset_sources"] = [
            d for d in meta["dataset_sources"] if "publication-swa-weights" not in d
        ]
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    item = dict(tag=tag, owner=owner, slug=slug, folder=str(folder),
                kernel=meta["id"], description=description,
                bytes=len(payload.encode("utf-8")),
                sha256=hashlib.sha256(text.encode()).hexdigest())
    manifest.append(item)
    print(tag, item["kernel"], item["bytes"], description)

(ROOT / "kaggle_push" / "g-manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8"
)

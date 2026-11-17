"""Direction B dry run (CPU): build a px_cache-format training cache for N train studies with heliosli's own inference-time
path (compact_infer.CStudies.rows: native render -> geometry bank -> SKM-TEA anchors -> v4 views), to measure seconds and rows
per study and validate the format compact_train reads (px.u8.npy, meta.f32.npy, index.json with done / blocks / cursor).
Environment = cell 6 of heliosli/blend-mine-in-private-lb-luck-to-all-upvoters (mainline code, pinned wheels, sources)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER = sys.argv[1] if len(sys.argv) > 1 else "nguynxunphc662"
N_STUDIES = int(sys.argv[2]) if len(sys.argv) > 2 else 24
h958 = json.loads((ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb").read_text(encoding="utf-8"))
setup = "".join(h958["cells"][6]["source"])
assert "T0 = NB_T0" in setup
NVS = "subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], check=False)"
assert setup.count(NVS) == 1
setup = setup.replace(NVS, "shutil.which('nvidia-smi') and " + NVS, 1)   # CPU kernel: no nvidia-smi binary

cell0 = "import time\nNB_T0 = time.time()\nN_STUDIES = %d\nSIZE = 320\nVIEWS = 'v4'\n" % N_STUDIES
cell_run = r'''import os, csv, hashlib, shutil, json, time, multiprocessing
import numpy as np, pandas as pd, torch
print('cpus', os.cpu_count(), 'cuda', torch.cuda.is_available(), flush=True)
REAL = DATA
tr = pd.read_csv(REAL / 'train.csv', dtype={'StudyInstanceUID': str})
LABELS = [c for c in tr.columns if c not in ('StudyInstanceUID', 'Report')]
pool = tr[tr[LABELS].isna().all(axis=1)]['StudyInstanceUID'].tolist()
pick = sorted(pool, key=lambda u: hashlib.md5(u.encode()).hexdigest())[:N_STUDIES]
FAKE = TMP / 'b_root'
FAKE.mkdir(parents=True, exist_ok=True)
tr[tr['StudyInstanceUID'].isin(set(pick))].to_csv(FAKE / 'train.csv', index=False)
ser = pd.read_csv(REAL / 'train_series.csv', dtype=str)
ser[ser['StudyInstanceUID'].isin(set(pick))].to_csv(FAKE / 'train_series.csv', index=False)
for u in pick:
    for dp, dn, fn in os.walk(REAL / 'train_series' / u):
        rel = Path(dp).relative_to(REAL)
        (FAKE / rel).mkdir(parents=True, exist_ok=True)
        for f in fn:
            if not (FAKE / rel / f).exists():
                os.symlink(Path(dp) / f, FAKE / rel / f)
for p in (str(CODE), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ['PYTHONPATH'] = str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', '')
import compact_infer as CI
ds = CI.CStudies(FAKE, 'train', pick, TMP / 'b_tmp', CODE, SIZE, 3, VIEWS, 3, CODE_SRC / 'anatomy.pt', 3.5)
(TMP / 'b_tmp').mkdir(parents=True, exist_ok=True)
pxs, metas, index = [], [], {'order': pick, 'done': {}, 'blocks': {}, 'cursor': 0, 'errors': {}, 'removed_series': 0,
                             'config': {'size': SIZE, 'views': VIEWS, 'builder': 'compact_infer.CStudies.rows (b dry run)'}}
times = []
for i, u in enumerate(pick):
    t = time.time()
    try:
        item, sides = ds.rows(u)
    except Exception as e:
        index['errors'][u] = repr(e)[:300]; print('ERROR', u[:20], repr(e)[:300], flush=True); continue
    px = item['px'].numpy() if hasattr(item['px'], 'numpy') else np.asarray(item['px'])
    meta = item['meta'].numpy() if hasattr(item['meta'], 'numpy') else np.asarray(item['meta'])
    cur, n = index['cursor'], len(px)
    blocks = item['blocks']
    for b in blocks:
        b['views'] = [[v, cur + off] for v, off in b['views']]
    index['done'][u] = [cur, n, len(blocks)]
    index['blocks'][u] = blocks
    index['cursor'] = cur + n
    index['removed_series'] += int(item.get('removed', 0))
    pxs.append(px.astype(np.uint8)); metas.append(meta.astype(np.float32))
    dt = time.time() - t; times.append(dt)
    print('STUDY %d/%d rows %d seconds %.1f px %s' % (i + 1, len(pick), n, dt, px.shape), flush=True)
out = WORK / 'b_px_dry'
out.mkdir(exist_ok=True)
np.save(out / 'px.u8.npy', np.concatenate(pxs)); np.save(out / 'meta.f32.npy', np.concatenate(metas))
json.dump(index, open(out / 'index.json', 'w'))
rows = [v[1] for v in index['done'].values()]
summary = {'studies': len(rows), 'errors': len(index['errors']), 'rows_mean': float(np.mean(rows)), 'rows_max': int(np.max(rows)),
           'seconds_mean': float(np.mean(times)), 'seconds_median': float(np.median(times)), 'cpus': os.cpu_count(),
           'bytes_per_study_320': float(np.mean(rows)) * 320 * 320, 'GB_4407_at_320': float(np.mean(rows)) * 320 * 320 * 4407 / 2**30,
           'GB_4407_at_256': float(np.mean(rows)) * 256 * 256 * 4407 / 2**30, 'GB_4407_at_224': float(np.mean(rows)) * 224 * 224 * 4407 / 2**30,
           'cpu_hours_4407_single_process': float(np.mean(times)) * 4407 / 3600}
json.dump(summary, open(WORK / 'b_dry_summary.json', 'w'), indent=1)
print('B_DRY_SUMMARY', json.dumps(summary), flush=True)
'''


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


slug = "rsna-b-px-dryrun"
out = ROOT / "kaggle_push" / slug
out.mkdir(parents=True, exist_ok=True)
nb = {"cells": [code(cell0), code(setup), code(cell_run)], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
      "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
for c in nb["cells"]:
    compile("".join(c["source"]), "c", "exec")
(out / (slug + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out)

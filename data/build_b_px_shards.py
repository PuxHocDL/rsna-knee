"""Direction B: px_cache-format training cache shards (CPU kernels), built with heliosli's inference-time path
(compact_infer.CStudies.rows: native render -> geometry bank -> SKM-TEA anchors -> v4 views), size SIZE.
Shard k holds the train studies whose md5 rank % NSH == k (gold included: compact_train monitors them).
Each study is rendered from its own one-study scratch root (the native reader globs the whole split per study).
Two worker processes; rows stream into px.u8.npy through a fixed-length .npy header rewritten at the end.
Outputs per shard: px.u8.npy, meta.f32.npy, index.json (px_cache format, row offsets local to the shard), progress.json.

usage: python build_b_px_shards.py <owner> <nsh> <size> [shards...]"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, NSH, SIZE = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
SHARDS = [int(x) for x in sys.argv[4:]] or list(range(NSH))
h958 = json.loads((ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb").read_text(encoding="utf-8"))
setup = "".join(h958["cells"][6]["source"])
NVS = "subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], check=False)"
assert "T0 = NB_T0" in setup and setup.count(NVS) == 1
setup = setup.replace(NVS, "shutil.which('nvidia-smi') and " + NVS, 1)

RUN = r'''import os, csv, hashlib, shutil, json, time, multiprocessing as mp
import numpy as np, pandas as pd, torch
REAL = DATA
tr = pd.read_csv(REAL / 'train.csv', dtype={'StudyInstanceUID': str})
order = sorted(tr['StudyInstanceUID'].tolist(), key=lambda u: hashlib.md5(u.encode()).hexdigest())
mine = [u for i, u in enumerate(order) if i % NSH == SHARD]
ser = pd.read_csv(REAL / 'train_series.csv', dtype=str)
SER = {u: g for u, g in ser.groupby('StudyInstanceUID')}
print('SHARD', SHARD, 'of', NSH, 'studies', len(mine), 'cpus', os.cpu_count(), flush=True)
for p in (str(CODE), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ['PYTHONPATH'] = str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', '')
import compact_infer as CI
_DS = None


def _work(u):
    global _DS
    pid = os.getpid()
    root = TMP / ('one_%d' % pid)
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    tr[tr['StudyInstanceUID'] == u].to_csv(root / 'train.csv', index=False)
    SER[u].to_csv(root / 'train_series.csv', index=False)
    for dp, dn, fn in os.walk(REAL / 'train_series' / u):
        rel = Path(dp).relative_to(REAL)
        (root / rel).mkdir(parents=True, exist_ok=True)
        for f in fn:
            os.symlink(Path(dp) / f, root / rel / f)
    t = time.time()
    try:
        if _DS is None:
            torch.set_num_threads(2)
            _DS = CI.CStudies(root, 'train', [u], TMP / ('w_%d' % pid), CODE, SIZE, 3, 'v4', 3, CODE_SRC / 'anatomy.pt', 3.5)
            (TMP / ('w_%d' % pid)).mkdir(parents=True, exist_ok=True)
        _DS.data = str(root)
        item, sides = _DS.rows(u)
        px = item['px'].numpy() if hasattr(item['px'], 'numpy') else np.asarray(item['px'])
        meta = item['meta'].numpy() if hasattr(item['meta'], 'numpy') else np.asarray(item['meta'])
        return u, px.astype(np.uint8), meta.astype(np.float32), item['blocks'], int(item.get('removed', 0)), None, time.time() - t
    except Exception as e:
        return u, None, None, None, 0, repr(e)[:300], time.time() - t
    finally:
        shutil.rmtree(root, ignore_errors=True)


OUT = WORK / ('px_s%02d' % SHARD)
OUT.mkdir(exist_ok=True)
HDR = 256                      # fixed .npy header length; rewritten with the final row count at the end


def npy_header(n):
    d = "{'descr': '|u1', 'fortran_order': False, 'shape': (%d, %d, %d), }" % (n, SIZE, SIZE)
    body = d + ' ' * (HDR - 10 - len(d) - 1) + '\n'
    return b'\x93NUMPY\x01\x00' + len(body).to_bytes(2, 'little') + body.encode('latin1')


fh = open(OUT / 'px.u8.npy', 'wb'); fh.write(npy_header(0))
metas, index = [], {'order': mine, 'done': {}, 'blocks': {}, 'cursor': 0, 'errors': {}, 'removed_series': 0,
                    'config': {'size': SIZE, 'views': 'v4', 'local_mm': 100.0, 'shard': SHARD, 'nsh': NSH,
                               'builder': 'compact_infer.CStudies.rows (heliosli inference path), one-study roots'}}
t0 = time.time(); times = []
with mp.get_context('fork').Pool(N_PROC) as pool:
    for k, (u, px, meta, blocks, removed, err, dt) in enumerate(pool.imap_unordered(_work, mine, chunksize=1)):
        times.append(dt)
        if err:
            index['errors'][u] = err; print('ERROR', u[:24], err, flush=True)
        else:
            cur, n = index['cursor'], len(px)
            fh.write(np.ascontiguousarray(px).tobytes()); metas.append(meta)
            for b in blocks:
                b['views'] = [[v, cur + off] for v, off in b['views']]
            index['done'][u] = [cur, n, len(blocks)]; index['blocks'][u] = blocks
            index['cursor'] = cur + n; index['removed_series'] += removed
        if (k + 1) % 20 == 0 or k + 1 == len(mine):
            el = time.time() - t0
            prog = {'done': len(index['done']), 'errors': len(index['errors']), 'total': len(mine), 'rows': index['cursor'],
                    'elapsed_min': round(el / 60, 1), 'eta_min': round(el / (k + 1) * (len(mine) - k - 1) / 60, 1),
                    'mean_s': round(float(np.mean(times)), 1)}
            json.dump(prog, open(OUT / 'progress.json', 'w')); print('PROGRESS', json.dumps(prog), flush=True)
fh.seek(0); fh.write(npy_header(index['cursor'])); fh.close()
np.save(OUT / 'meta.f32.npy', np.concatenate(metas) if metas else np.zeros((0, 12), np.float32))
json.dump(index, open(OUT / 'index.json', 'w'))
chk = np.load(OUT / 'px.u8.npy', mmap_mode='r'); assert chk.shape == (index['cursor'], SIZE, SIZE)
print('SHARD_DONE', SHARD, 'studies', len(index['done']), 'errors', len(index['errors']), 'rows', index['cursor'],
      'GB %.2f' % (index['cursor'] * SIZE * SIZE / 2 ** 30), 'minutes %.1f' % ((time.time() - t0) / 60), flush=True)
'''


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


for k in SHARDS:
    cell0 = "import time\nNB_T0 = time.time()\nSHARD = %d\nNSH = %d\nSIZE = %d\nN_PROC = 2\n" % (k, NSH, SIZE)
    nb = {"cells": [code(cell0), code(setup), code(RUN)], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
          "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4}
    for c in nb["cells"]:
        compile("".join(c["source"]), "c", "exec")
    slug = "rsna-b-px%d-s%02d" % (SIZE, k)
    out = ROOT / "kaggle_push" / "b_shards" / slug
    out.mkdir(parents=True, exist_ok=True)
    (out / (slug + ".ipynb")).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    meta = {"id": f"{OWNER}/{slug}", "title": slug, "code_file": slug + ".ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "dataset_sources": ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004"],
            "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("built", slug)

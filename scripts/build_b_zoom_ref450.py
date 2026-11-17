"""Evaluate every B fold model separately on the 450 clean studies (58 gold + 392 zoliseress reference studies; none of them
was trained on by the B v5h folds; folds 0/1 never saw gold) in ONE decode pass: one worker on one GPU with --dump-logits,
which records each fold model's logits per study. The submission-time path (heliosli's R6 worker setup + compact_infer).
usage: python build_b_zoom_ref450.py <owner> <models_dataset> <labels_dataset_with_holdout_ids> <arm[,arm...]> <slug> <zoom>
Zoom-specialist variant: the worker copy shows every local view (ids 1-5) at its centre 1/zoom, resized back to the cache
size, before the centre crop (the trainer's --zoom transform, applied at the same point).
Fold files of every listed arm are copied into one bundle as arm EV (fold numbers must not collide). Each worker writes
its own --dump-logits file through a small shim (the controller always starts one worker per visible GPU)."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, MDS, LDS, ARM, SLUG = sys.argv[1:6]
ZOOM = float(sys.argv[6])
nb = json.loads((ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb").read_text(encoding="utf-8"))
setup = "".join(nb["cells"][6]["source"])
assert setup.startswith("ARM = 'E_lrprior'") and "T0 = NB_T0" in setup
cell0 = "# B fold models on the 450 clean train studies (gold + reference); per-model logits. No submission is produced.\nimport time\nNB_T0 = time.time()\n"
cell_root = r'''# Scratch data root with the gold studies + the 392 reference studies (train.csv / train_series.csv filtered, DICOM files symlinked)
import csv, glob, os
import pandas as pd
REAL = DATA
tr = pd.read_csv(REAL / 'train.csv', dtype={'StudyInstanceUID': str})
LABELS = [c for c in tr.columns if c not in ('StudyInstanceUID', 'Report')]
gold_ids = tr[tr[LABELS].notna().all(axis=1)]['StudyInstanceUID'].tolist()
hold = []
for pat in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/'):
    hold += glob.glob(pat + 'holdout_ref392_ids.csv')
ref_ids = pd.read_csv(sorted(hold)[0]).iloc[:, 0].astype(str).tolist()
IDS = sorted(set(gold_ids) | set(ref_ids))
print('studies', len(IDS), 'gold', len(gold_ids), 'ref', len(ref_ids), flush=True)
FAKE = TMP / 'eval_root'
FAKE.mkdir(parents=True, exist_ok=True)
sub = tr[tr['StudyInstanceUID'].isin(set(IDS))]
sub.to_csv(FAKE / 'train.csv', index=False)
ser = pd.read_csv(REAL / 'train_series.csv', dtype={'StudyInstanceUID': str, 'SeriesInstanceUID': str})
ser[ser['StudyInstanceUID'].isin(set(IDS))].to_csv(FAKE / 'train_series.csv', index=False)
header = list(pd.read_csv(REAL / 'sample_submission.csv', nrows=1).columns)
with open(FAKE / 'sample_submission.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.writer(fh); w.writerow(header)
    for u in IDS:
        w.writerow([u] + [0.5] * (len(header) - 1))
for u in IDS:
    for dp, dn, fn in os.walk(REAL / 'train_series' / u):
        rel = Path(dp).relative_to(REAL)
        (FAKE / rel).mkdir(parents=True, exist_ok=True)
        for f in fn:
            dst = FAKE / rel / f
            if not dst.exists():
                os.symlink(Path(dp) / f, dst)
DATA = FAKE
'''
cell_b = r'''# All fold models of the listed arms in one bundle (arm EV); a shim gives every worker its own --dump-logits file
ARMS = %r
bundle = TMP / 'bundle_b'
bundle.mkdir(parents=True, exist_ok=True)
for arm in ARMS:
    hits = []
    for pat in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/'):
        hits += [h for h in glob.glob(pat + arm + '_s*_f*.pt') if 'abnormality-detection' not in h]
    for h in sorted(set(hits)):
        dst = bundle / ('EV_' + os.path.basename(h).split('_', 1)[1])
        assert not dst.exists(), ('fold collision', h)
        shutil.copy(h, dst)
models = sorted(p.name for p in bundle.glob('EV_s*_f*.pt'))
print('B_MODELS', models, flush=True)
ZOOM = %r
w = (SRC / 'compact_infer.py').read_text(encoding='utf-8')
ZDEF = (chr(10) + 'ZOOM = ' + repr(ZOOM) + chr(10) + chr(10) + chr(10) + 'def zoom_locals(img, views, z):' + chr(10)
        + '    import torch.nn.functional as F' + chr(10)
        + '    sel = np.flatnonzero((views >= 1) & (views <= 5))' + chr(10)
        + '    if z <= 1 or not len(sel):' + chr(10) + '        return img' + chr(10)
        + '    S = img.shape[-1]' + chr(10) + '    c = int(round(S / z))' + chr(10) + '    o = (S - c) // 2' + chr(10)
        + '    x = torch.from_numpy(np.ascontiguousarray(img[sel, :, o:o + c, o:o + c])).float()' + chr(10)
        + "    x = F.interpolate(x, size=(S, S), mode='bilinear', align_corners=False)" + chr(10)
        + '    img = np.array(img, copy=True)' + chr(10)
        + '    img[sel] = x.round().clamp(0, 255).to(torch.uint8).numpy()' + chr(10) + '    return img' + chr(10))
A1 = chr(10) + 'import torch' + chr(10)
A2 = '            m = meta[rows[:, 1]]' + chr(10)
assert w.count(A1) == 1 and w.count(A2) == 1
w = w.replace(A1, A1 + ZDEF, 1).replace(A2, A2 + '            img = zoom_locals(img, m[:, 3], ZOOM)' + chr(10), 1)
(SRC / 'compact_infer_zoom.py').write_text(w, encoding='utf-8')
print('ZOOM worker written, zoom', ZOOM, flush=True)
(SRC / 'compact_infer_dump.py').write_text(
    "import os, runpy, sys\n"
    "a = sys.argv[1:]\n"
    "out = a[a.index('--output') + 1]\n"
    "i = a.index('--dump-logits')\n"
    "a[i + 1] = a[i + 1] + '.' + os.path.basename(os.path.normpath(out)) + '.json'\n"
    "sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compact_infer_zoom.py')] + a\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n", encoding='utf-8')
dump = WORK / 'b_dump'
b_args = ['--backbone', 'tf_efficientnetv2_s.in21k_ft_in1k', '--size', '256', '--crop', '224', '--eval-stride', '3', '--views', 'v4',
          '--slices', '3', '--phys-mm', '3.5', '--anchor-weights', str(CODE_SRC / 'anatomy.pt'), '--dump-logits', str(dump)]
env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
pred = TMP / 'prediction_b'
rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'train',
                     '--code', str(CODE), '--bundle', str(bundle), '--arm', 'EV', '--output', str(pred),
                     '--amp', 'auto', '--loader-workers', '2', '--deadline', str(NB_T0 + 8.5 * 3600),
                     '--worker', 'compact_infer_dump.py', '--worker-args', json.dumps(b_args),
                     '--max-fallback-fraction', '1.0'], env=env).returncode
for p in sorted(pred.glob('worker*/log.txt')):
    shutil.copyfile(p, WORK / ('b_' + p.parent.name + '_log.txt'))
merged = {}
for f in sorted(WORK.glob('b_dump.worker*.json')):
    merged.update(json.loads(f.read_text()))
json.dump(merged, open(WORK / 'b_dump_all.json', 'w'))
json.dump({'models': models}, open(WORK / 'b_models.json', 'w'))
print('BZ_REF450_DONE rc', rc, 'studies', len(merged), 'minutes %%.1f' %% ((time.time() - NB_T0) / 60), flush=True)
''' % (ARM.split(','), ZOOM)


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


cells = [code(cell0), code(setup), code(cell_root), code(cell_b)]
for c in cells:
    compile("".join(c["source"]), "cell", "exec")
out = ROOT / "kaggle_push" / SLUG
out.mkdir(parents=True, exist_ok=True)
(out / (SLUG + ".ipynb")).write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                 "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4},
                                                indent=1), encoding="utf-8")
meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": SLUG + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004", MDS, LDS],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out)

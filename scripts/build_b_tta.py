"""Inference-only TTA sweep for one B arm on the 450 clean studies (58 gold + 392 zoliseress reference): the extras R6's own
submission path uses and our B runs did not -- every-slice windows (--eval-stride 1), the R6 bundle's side classifier
(side_cls.pt) and v4c flip TTA. B fold models have no extent head, so the worker never adds the plane column the flip needs;
the trainer's v9.4 note says the model ignores the extra meta columns (only the extent head reads columns 6-8), so a patched
copy of the worker adds them whenever --flip-tta is on. Each config dumps per-model plain + mirrored logits.
usage: python build_b_tta.py <owner> <models_dataset> <labels_dataset_with_holdout_ids> <arm> <slug> [config,config...]"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, MDS, LDS, ARM, SLUG = sys.argv[1:6]
ONLY = sys.argv[6].split(",") if len(sys.argv) > 6 else None
nb = json.loads((ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb").read_text(encoding="utf-8"))
setup = "".join(nb["cells"][6]["source"])
assert setup.startswith("ARM = 'E_lrprior'") and "T0 = NB_T0" in setup
src450 = (ROOT / "kaggle_push" / "build_b_ref450.py").read_text(encoding="utf-8")
cell_root = src450.split("cell_root = r'''", 1)[1].split("'''", 1)[0]
cell0 = "# B TTA sweep on the 450 clean train studies; per-model logits per config. No submission is produced.\nimport time\nNB_T0 = time.time()\n"
cell_b = r'''# Patched worker copy: plane columns whenever --flip-tta is on; one shim per config gives every worker its own dump file
ARM_B = %r
bundle = TMP / 'bundle_b'
bundle.mkdir(parents=True, exist_ok=True)
hits = []
for pat in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/'):
    hits += [h for h in glob.glob(pat + ARM_B + '_s*_f*.pt') if 'abnormality-detection' not in h]
for h in sorted(set(hits)):
    shutil.copy(h, bundle / ('EV_' + os.path.basename(h).split('_', 1)[1]))
models = sorted(p.name for p in bundle.glob('EV_s*_f*.pt'))
print('B_MODELS', models, flush=True)
w = (SRC / 'compact_infer.py').read_text(encoding='utf-8')
for old, new in [('extent_meta=N_EXTENT[0] > 0,', 'extent_meta=N_EXTENT[0] > 0 or a.flip_tta,'),
                 ('if N_EXTENT[0] == 0:\n            raise SystemExit("--flip-tta needs', 'if False:\n            raise SystemExit("--flip-tta needs')]:
    assert w.count(old) == 1, old
    w = w.replace(old, new)
(SRC / 'compact_infer_tta.py').write_text(w, encoding='utf-8')
(SRC / 'compact_infer_dump.py').write_text(
    "import os, runpy, sys\n"
    "a = sys.argv[1:]\n"
    "out = a[a.index('--output') + 1]\n"
    "i = a.index('--dump-logits')\n"
    "a[i + 1] = a[i + 1] + '.' + os.path.basename(os.path.normpath(out)) + '.json'\n"
    "sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compact_infer_tta.py')] + a\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n", encoding='utf-8')
SIDE = ['--side-model', str(BUNDLE / manifest['side_model']), '--side-threshold', str(manifest.get('side_threshold', 2.0)),
        '--side-position-mm', str(manifest.get('side_position_mm', 20.0)), '--side-max-x', str(manifest.get('side_max_x', 0.0))]
CONFIGS = [('s3_flip', ['--eval-stride', '3', '--flip-tta']),
           ('s1_flip', ['--eval-stride', '1', '--flip-tta']),
           ('s1_side_flip', ['--eval-stride', '1', '--flip-tta'] + SIDE)]
ONLY = %r
CONFIGS = [c for c in CONFIGS if ONLY is None or c[0] in ONLY]
env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
for name, extra in CONFIGS:
    t1 = time.time()
    dump = WORK / ('b_dump_' + name)
    b_args = ['--backbone', 'tf_efficientnetv2_s.in21k_ft_in1k', '--size', '256', '--crop', '224', '--views', 'v4', '--slices', '3',
              '--phys-mm', '3.5', '--anchor-weights', str(CODE_SRC / 'anatomy.pt'), '--dump-logits', str(dump)] + extra
    pred = TMP / ('prediction_' + name)
    rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'train',
                         '--code', str(CODE), '--bundle', str(bundle), '--arm', 'EV', '--output', str(pred),
                         '--amp', 'auto', '--loader-workers', '2', '--deadline', str(NB_T0 + 8.6 * 3600),
                         '--worker', 'compact_infer_dump.py', '--worker-args', json.dumps(b_args),
                         '--max-fallback-fraction', '1.0'], env=env).returncode
    for p in sorted(pred.glob('worker*/log.txt')):
        shutil.copyfile(p, WORK / ('b_' + name + '_' + p.parent.name + '_log.txt'))
    merged = {}
    for f in sorted(WORK.glob('b_dump_' + name + '.worker*.json')):
        merged.update(json.loads(f.read_text()))
    json.dump(merged, open(WORK / ('b_dump_all_' + name + '.json'), 'w'))
    print('B_TTA_CONFIG', name, 'rc', rc, 'studies', len(merged), 'minutes %%.1f' %% ((time.time() - t1) / 60), flush=True)
json.dump({'models': models, 'configs': [c[0] for c in CONFIGS]}, open(WORK / 'b_models.json', 'w'))
print('B_TTA_DONE minutes %%.1f' %% ((time.time() - NB_T0) / 60), flush=True)
''' % (ARM, ONLY)


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

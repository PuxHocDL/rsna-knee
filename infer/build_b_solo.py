"""Submission of our compact B models alone: heliosli's R6 worker setup (code, wheels, anatomy.pt; cell 6 of
blend-mine-in-private-lb-luck-to-all-upvoters, verbatim) + compact_infer on the test DICOMs with the B training configuration
(px256 v4 views, crop 224, 3 slices, 3.5 mm). The fold models of one arm are averaged in logit space by the worker.
usage: python build_b_solo.py <owner> <models_dataset> <arm> <slug> <title>"""
import json, os, sys
from zoom_patch import worker_cell
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, MDS, ARM, SLUG, TITLE = sys.argv[1:6]
nb = json.loads((ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb").read_text(encoding="utf-8"))
setup = "".join(nb["cells"][6]["source"])
assert setup.startswith("ARM = 'E_lrprior'") and "T0 = NB_T0" in setup

cell0 = "# %s\nimport time\nNB_T0 = time.time()\n" % TITLE
ZOOM = float(os.environ.get('ZOOM', '1'))   # >1: zoom-specialist B models (see zoom_patch.py)
cell_b = r'''# Our B models on the test studies; the worker's submission.csv (fallback rows = column means) is the submission
import glob
B_ARM = %r
hits = []
for pat in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/'):
    hits += [h for h in glob.glob(pat + B_ARM + '_s*_f*.pt') if 'abnormality-detection' not in h]
src = sorted({os.path.dirname(h) for h in hits})
assert len(src) == 1, src
bundle = TMP / 'bundle_b'
bundle.mkdir(parents=True, exist_ok=True)
for h in sorted(glob.glob(src[0] + '/' + B_ARM + '_s*_f*.pt')):
    shutil.copy(h, bundle / os.path.basename(h))
print('B_MODELS', sorted(p.name for p in bundle.iterdir()), flush=True)
b_args = ['--backbone', 'tf_efficientnetv2_s.in21k_ft_in1k', '--size', '256', '--crop', '224', '--eval-stride', '3', '--views', 'v4',
          '--slices', '3', '--phys-mm', '3.5', '--anchor-weights', str(CODE_SRC / 'anatomy.pt')]
ZOOM_WORKER = 'compact_infer.py'
%senv = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
pred = TMP / 'prediction_b'
rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'test',
                     '--code', str(CODE), '--bundle', str(bundle), '--arm', B_ARM, '--output', str(pred),
                     '--amp', 'auto', '--loader-workers', '2', '--deadline', str(NB_T0 + 8.5 * 3600),
                     '--worker', ZOOM_WORKER, '--worker-args', json.dumps(b_args),
                     '--max-fallback-fraction', '1.0'], env=env).returncode
for p in sorted(pred.glob('worker*/log.txt')):
    shutil.copyfile(p, WORK / ('b_' + p.parent.name + '_log.txt'))
if rc != 0:
    raise RuntimeError('B controller exit code %%d' %% rc)
shutil.copyfile(pred / 'submission.csv', WORK / 'submission.csv')
rec = json.loads((pred / 'submission_receipt.json').read_text())
print('B_SOLO_DONE', json.dumps({k: rec.get(k) for k in ('status', 'rows', 'scored', 'fallback_studies', 'seconds')}),
      'minutes %%.1f' %% ((time.time() - NB_T0) / 60), flush=True)
''' % (ARM, worker_cell(ZOOM) if ZOOM > 1 else '')


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


cells = [code(cell0), code(setup), code(cell_b)]
for c in cells:
    compile("".join(c["source"]), "cell", "exec")
out = ROOT / "kaggle_push" / SLUG
out.mkdir(parents=True, exist_ok=True)
(out / (SLUG + ".ipynb")).write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                 "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 4},
                                                indent=1), encoding="utf-8")
meta = {"id": f"{OWNER}/{SLUG}", "title": SLUG, "code_file": SLUG + ".ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004", MDS],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", out)

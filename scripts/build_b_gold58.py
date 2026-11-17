"""Score our compact-pipeline B models through the submission-time path (heliosli's R6 worker setup, compact_infer from DICOM)
on the 58 gold train studies, plain and with --flip-tta. Checks that inference reproduces the training-time gold read
(B1 fold 0: gold macro 0.9266 from the trainer's own evaluation) before any submission uses it.

Cell 6 of heliosli/blend-mine-in-private-lb-luck-to-all-upvoters (R6 setup: code, wheels, worker sources) is reused verbatim;
then a second bundle folder holds our fold models and the worker arguments are replaced by the B training configuration.
usage: python build_b_gold58.py <owner> <models_dataset> <arm>[,<arm>...]"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OWNER, MDS, ARMS = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
SRC_NB = ROOT / "pulled" / "h958" / "blend-mine-in-private-lb-luck-to-all-upvoters.ipynb"
OUT = ROOT / "kaggle_push" / "b-gold58"
nb = json.loads(SRC_NB.read_text(encoding="utf-8"))
setup = "".join(nb["cells"][6]["source"])
assert setup.startswith("ARM = 'E_lrprior'") and "T0 = NB_T0" in setup

cell0 = '''# B models on the 58 gold train studies through the submission-time worker. No submission is produced.
import time
NB_T0 = time.time()
'''
gold_src = (ROOT / "kaggle_push" / "build_r6_gold58.py").read_text(encoding="utf-8")
cell_root = gold_src.split("cell_root = '''", 1)[1].split("'''", 1)[0]

cell_b = '''# Our B bundle and worker arguments (the B training configuration: px256 v4 views, crop 224, 3 slices, 3.5 mm)
import glob
ARMS = %r
B_SRC = sorted({os.path.dirname(p) for pat in ('/kaggle/input/*/', '/kaggle/input/*/*/', '/kaggle/input/*/*/*/', '/kaggle/input/*/*/*/*/')
                for p in glob.glob(pat + ARMS[0] + '_s*_f*.pt') if 'abnormality-detection' not in p})
assert len(B_SRC) == 1, B_SRC
B_SRC = Path(B_SRC[0])
B_ARGS = ['--backbone', 'tf_efficientnetv2_s.in21k_ft_in1k', '--size', '256', '--crop', '224', '--eval-stride', '3', '--views', 'v4',
          '--slices', '3', '--phys-mm', '3.5', '--anchor-weights', str(CODE_SRC / 'anatomy.pt')]
env = dict(os.environ, PYTHONPATH=str(CODE) + os.pathsep + os.environ.get('PYTHONPATH', ''), PYTHONDONTWRITEBYTECODE='1')
import numpy as np
from sklearn.metrics import roc_auc_score
RES = {}
for arm in ARMS:
    bundle = TMP / ('bundle_' + arm)
    bundle.mkdir(exist_ok=True)
    for p in B_SRC.glob(arm + '_s*_f*.pt'):
        shutil.copy(p, bundle / p.name)
    print(arm, sorted(x.name for x in bundle.iterdir()), flush=True)
    for tag, extra in (('plain', []), ('flip', ['--flip-tta'])):
        pred = TMP / ('pred_%%s_%%s' %% (arm, tag))
        t = time.time()
        rc = subprocess.run([sys.executable, '-u', str(SRC / 'probe_dual.py'), '--data', str(DATA), '--split', 'train',
                             '--code', str(CODE), '--bundle', str(bundle), '--arm', arm, '--output', str(pred),
                             '--amp', 'auto', '--loader-workers', '2', '--deadline', str(T0 + 8.5 * 3600),
                             '--worker', 'compact_infer.py', '--worker-args', json.dumps(B_ARGS + extra),
                             '--max-fallback-fraction', '1.0'], env=env).returncode
        for p in sorted(pred.glob('worker*/log.txt')):
            shutil.copyfile(p, WORK / ('%%s_%%s_%%s_log.txt' %% (arm, tag, p.parent.name)))
        if rc != 0:
            print('FAILED', arm, tag, rc, flush=True)
            continue
        shutil.copyfile(pred / 'submission.csv', WORK / ('%%s_%%s_gold58.csv' %% (arm, tag)))
        rec = json.loads((pred / 'submission_receipt.json').read_text())
        bad = set(rec.get('fallback_reasons', {}))
        p_ = pd.read_csv(WORK / ('%%s_%%s_gold58.csv' %% (arm, tag)), dtype={'StudyInstanceUID': str}).set_index('StudyInstanceUID').loc[GOLD_IDS]
        ok = [u for u in GOLD_IDS if u not in bad]
        g = gold.set_index('StudyInstanceUID').loc[ok]
        auc = {c: roc_auc_score(g[c].astype(int), p_.loc[ok, c]) for c in LABELS}
        RES['%%s_%%s' %% (arm, tag)] = {'scored': len(ok), 'macro': float(np.mean(list(auc.values()))), 'auc': auc, 'seconds': time.time() - t}
        print('B_GOLD', arm, tag, 'scored', len(ok), 'macro %%.4f' %% np.mean(list(auc.values())), 'seconds %%.0f' %% (time.time() - t), flush=True)
json.dump(RES, open(WORK / 'b_gold58_auc.json', 'w'), indent=1)
''' % (ARMS,)


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}


out_nb = {"cells": [code(cell0), code(setup), code(cell_root), code(cell_b)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
for c in out_nb["cells"]:
    compile("".join(c["source"]), "cell", "exec")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "rsna-b-gold58.ipynb").write_text(json.dumps(out_nb, indent=1), encoding="utf-8")
meta = {"id": f"{OWNER}/rsna-b-gold58", "title": "rsna-b-gold58", "code_file": "rsna-b-gold58.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["heliosli/rsna-knee-r6-frozen-fe055406-w1", "heliosli/rsna-orthofoundation-fold0-epoch7-20261004", MDS],
        "competition_sources": ["rsna-knee-abnormality-detection"], "kernel_sources": [], "model_sources": []}
(OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("built", OUT)

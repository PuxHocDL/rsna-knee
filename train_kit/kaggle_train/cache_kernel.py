# RSNA knee -- training cache shard (CPU). Re-uses the Raptor inference preprocessing
# (fastread 'study_lut', byte-identical to build_study) at IMG 384 / 96 slices, then stores
# the centre 0.8 crop (307 px) losslessly as uint8 so train windows == inference windows.
import os, sys, time, glob, base64, importlib.util
import numpy as np, pandas as pd, multiprocessing as mp

SHARD, NSH = __SHARD__, __NSH__
IMG = 384
SLOTS = [("Sagittal", 1, 26), ("Sagittal", 0, 22), ("Coronal", 1, 18),
         ("Coronal", 0, 12), ("Axial", -1, 18)]
MAXS = 96
CS = int(round(IMG * 0.8)); C0 = (IMG - CS) // 2          # 307, 38 (same as r224 _crop)
OUT = "/kaggle/working"
_FASTREAD_B64 = "__FASTREAD_B64__"

FR_PATH = "/tmp/_fastread.py"
open(FR_PATH, "wb").write(base64.b64decode(_FASTREAD_B64))


def _load_fastread():
    spec = importlib.util.spec_from_file_location("_fastread", FR_PATH)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def find_root():
    for b in ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
              "/kaggle/input/rsna-knee-abnormality-detection"]:
        if os.path.exists(b + "/train_series.csv"):
            return b
    for d in glob.glob("/kaggle/input/*") + glob.glob("/kaggle/input/*/*"):
        if os.path.exists(d + "/train_series.csv"):
            return d
    raise RuntimeError("competition root not found")


_SER = None; _TSDIR = None; _B = None
def _init():
    global _B
    _B = _load_fastread().make_builder("study_lut", IMG, SLOTS, hdr_threads=32, px_threads=16)


def _one(arg):
    i, sid = arg
    try:
        vol, mask = _B(sid, _SER, _TSDIR)
        ok = 1
    except Exception as e:
        print(f"  FAIL {sid[:20]} {type(e).__name__}: {e}", flush=True)
        vol = np.zeros((MAXS, IMG, IMG), np.uint8); mask = np.zeros(MAXS, np.uint8); ok = 0
    return i, np.ascontiguousarray(vol[:, C0:C0 + CS, C0:C0 + CS]), mask.astype(np.uint8), ok


def main():
    global _SER, _TSDIR
    t0 = time.time()
    ROOT = find_root()
    _TSDIR = ROOT + "/train_series" if os.path.isdir(ROOT + "/train_series") else ROOT + "/train_images"
    print("root", ROOT, "| series dir", _TSDIR, "| exists", os.path.isdir(_TSDIR), flush=True)
    ser = pd.read_csv(ROOT + "/train_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    _SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    ids = sorted(_SER.keys())[SHARD::NSH]
    N = len(ids)
    print(f"shard {SHARD}/{NSH}: {N} studies", flush=True)
    vols = np.lib.format.open_memmap(f"{OUT}/vols.npy", mode="w+", dtype=np.uint8, shape=(N, MAXS, CS, CS))
    masks = np.zeros((N, MAXS), np.uint8); oks = np.zeros(N, np.uint8)
    nproc = max(4, (os.cpu_count() or 4))
    with mp.get_context("fork").Pool(nproc, initializer=_init) as pool:
        for k, (i, v, m, ok) in enumerate(pool.imap_unordered(_one, list(enumerate(ids)), chunksize=1)):
            vols[i] = v; masks[i] = m; oks[i] = ok
            if (k + 1) % 100 == 0 or k + 1 == N:
                el = time.time() - t0
                print(f"  {k+1}/{N} | {el:.0f}s | eta {el/(k+1)*(N-k-1):.0f}s", flush=True)
    vols.flush(); del vols
    np.save(f"{OUT}/masks.npy", masks)
    open(f"{OUT}/ids.txt", "w").write("\n".join(ids))
    print(f"done {N} studies, failed {int((oks == 0).sum())}, empty-mask {int((masks.sum(1) == 0).sum())}, "
          f"{time.time()-t0:.0f}s", flush=True)
    os.remove(FR_PATH)


if __name__ == "__main__":
    main()

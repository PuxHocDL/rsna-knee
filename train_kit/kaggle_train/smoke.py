"""Local smoke test of the target build + windowing + gpu_aug shapes (CPU, tiny fake shard)."""
import ast, os, re, sys, tempfile, types
import numpy as np, pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..", "labels")
src = open(os.path.join(HERE, "train_kernel.py"), encoding="utf-8").read()
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# --- targets exactly as in the kernel
tr = pd.read_csv(f"{D}/train.csv")
gold = tr[tr[LAB].notna().any(axis=1)].set_index("StudyInstanceUID")[LAB].astype(float)
st = pd.read_csv(f"{D}/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")[LAB]
pk = pd.read_csv(f"{D}/report_labels_v2.csv").set_index("StudyInstanceUID")[LAB]
st, pk = st.align(pk, join="outer")
tgt = pd.DataFrame(np.where(np.isclose(st.values, 0.25) | st.isna().values, pk.values,
                            np.where(pk.isna().values, st.values, (st.values + pk.values) / 2)),
                   index=st.index, columns=LAB).dropna()
print("targets", tgt.shape, "nan", int(tgt.isna().sum().sum()))
print(tgt.describe().T[["mean", "min", "max"]].round(3))
# agreement of targets with gold (sanity: AUC of label vs gold on the 58)
def roc_auc_score(y, p):
    y = np.asarray(y); r = pd.Series(np.asarray(p)).rank().values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
g = gold.index.intersection(tgt.index)
print("gold in tgt", len(g), "label-vs-gold AUC",
      np.mean([roc_auc_score(gold.loc[g, c], tgt.loc[g, c]) for c in LAB]).round(4))
compile(src, "train_kernel.py", "exec")
print("train_kernel compiles")
tree = ast.parse(src)
names = {"valid_centers", "eval_centers", "read_windows", "TrainDS", "RaptorClassifier"}
subset = ast.Module(body=[node for node in tree.body if getattr(node, "name", None) in names],
                    type_ignores=[])
namespace = dict(np=np, torch=torch, nn=torch.nn, WIN=3, SEED=42, tgt=tgt)
exec(compile(subset, "train_kernel_helpers", "exec"), namespace)
with tempfile.TemporaryDirectory() as tmp:
    data = np.arange(2 * 8 * 5 * 5, dtype=np.uint8).reshape(2, 8, 5, 5)
    np.save(os.path.join(tmp, "vols.npy"), data)
    sid = tgt.index[0]
    namespace.update(shards=[dict(dir=tmp, masks=np.ones((2, 8), np.uint8))],
                     index={sid: (0, 1)})
    windows = namespace["read_windows"](0, 1, [1, 4])
    np.testing.assert_array_equal(windows, data[1][[[0, 1, 2], [3, 4, 5]]])
    x, y = namespace["TrainDS"]([sid])[0]
    assert x.shape == (3, 3, 5, 5) and y.shape == (12,)
    for centers in ([1, 1, 6], [3, 2, 4], [6]):
        actual = namespace["read_windows"](0, 1, centers)
        indices = np.stack([np.array(centers) - 1, centers, np.array(centers) + 1], 1)
        np.testing.assert_array_equal(actual, data[1][indices])
    for _ in range(1000):
        namespace["read_windows"](0, 1, [1, 4])
    with open(os.path.join(tmp, "vols.npy"), "r+b") as stream:
        stream.truncate(namespace["shards"][0]["layout"][0] + 10)
    try:
        namespace["read_windows"](0, 1, [1])
    except EOFError:
        pass
    else:
        raise AssertionError("truncated cache was not rejected")
    os.remove(os.path.join(tmp, "vols.npy"))
    assert windows.flags.owndata
backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 5 * 5, 8))
model = namespace["RaptorClassifier"](backbone, F_dim=8).eval()
x = torch.randn(1, 19, 3, 5, 5)
with torch.no_grad():
    feats = torch.cat([model.encode(chunk) for chunk in x.split(4, dim=1)], dim=1)
    torch.testing.assert_close(model(x), model.head(feats))
print("bounded cache reads and chunked validation pass")
helpers = ast.Module(
    body=[node for node in tree.body if getattr(node, "name", None) in {"cpu_state", "save_resume"}],
    type_ignores=[],
)
import random
net = torch.nn.Linear(3, 2)
optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)
scaler = torch.amp.GradScaler("cpu", enabled=False)
sample = torch.randn(4, 3)
net(sample).square().mean().backward()
optimizer.step(); optimizer.zero_grad(set_to_none=True)
with tempfile.TemporaryDirectory() as tmp:
    scope = dict(torch=torch, os=os, random=random, np=np, model=net, opt=optimizer,
                 scaler=scaler, ARCH="test", RES=224, ep=0, next_batch=4, step=1,
                 history=[], swa=None, n_swa=0, epoch_loss=0.5, epoch_batches=4,
                 train_ids=["test"], EPOCHS=16, BS=4, ACCUM=4, OUT=tmp, TAG="test",
                 log=lambda *args: None)
    exec(compile(helpers, "resume_helpers", "exec"), scope)
    scope["save_resume"]()
    state = torch.load(os.path.join(tmp, "test_resume.pt"), weights_only=False)
    assert state["next_batch"] == 4 and state["step"] == 1
    restored = torch.nn.Linear(3, 2)
    restored.load_state_dict(state["model"])
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.01)
    restored_opt.load_state_dict(state["optimizer"])
    for m, opt in [(net, optimizer), (restored, restored_opt)]:
        m(sample).square().mean().backward()
        opt.step(); opt.zero_grad(set_to_none=True)
    for a, b in zip(net.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b)
    assert not os.path.exists(os.path.join(tmp, "test_resume.pt.tmp"))
    epoch_order = torch.randperm(29, generator=torch.Generator().manual_seed(42)).tolist()
    batches = [epoch_order[i:i + 4] for i in range(0, 28, 4)]
    assert batches[:4] + batches[4:] == batches
print("atomic resume checkpoint and restored AdamW update pass")
c = open(os.path.join(HERE, "cache_kernel.py"), encoding="utf-8").read().replace("__SHARD__", "0").replace("__NSH__", "3")
compile(c, "cache_kernel.py", "exec")
print("cache_kernel compiles")

"""Spawn B training calls on the DEPLOYED rsna-b-train app (modal deploy modal_btrain.py first).  A spawned call runs to
completion on Modal even if this machine shuts down (unlike `modal run --detach` with a local entrypoint).
usage: MODAL_PROFILE=<ws> python spawn_btrain.py <targets.csv> <tag> <fold,fold,...> ["<override flags>"]"""
import sys
import modal
targets, tag, folds = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
over = sys.argv[4] if len(sys.argv) > 4 else ""   # e.g. "--backbone tf_efficientnetv2_m.in21k_ft_in1k --slices 5 --epochs 24"
train = modal.Function.from_name("rsna-b-train", "train")
for f in folds:
    call = train.spawn(targets, tag, f, over)
    print("spawned", tag, "fold", f, call.object_id, flush=True)

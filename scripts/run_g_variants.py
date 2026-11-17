import json
import os
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
items = json.loads((HERE / "g-manifest.json").read_text())
state_path = HERE / "g-run-state.json"
if state_path.exists():
    raise RuntimeError("Existing G run state found; inspect it before starting another run.")
state = {}
deadline = time.monotonic() + 45 * 60


def save():
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def command(item, args):
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = env["G_TOKEN_" + item["owner"].upper()]
    result = subprocess.run(["kaggle", *args], env=env, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=180)
    return result.returncode, result.stdout + result.stderr


def validate(item, folder):
    log = (folder / (item["slug"] + ".log")).read_text(encoding="utf-8")
    assert "S6F_BLEND_DONE" in log, "S6fall blend did not complete"
    lines = [line for line in log.splitlines() if "FINAL_RECEIPT" in line]
    assert lines, "Missing final receipt"
    decoded = json.loads(lines[-1].lstrip(","))
    receipt = json.loads(decoded["data"].split("FINAL_RECEIPT ", 1)[1])
    assert receipt["s6fall_used"] is True
    assert receipt["r7_flip_tta"] == (item["tag"] != "G1"), receipt
    assert receipt["r224_used"] == (item["tag"] not in ("G3", "G4")), receipt
    assert receipt["r7_only"] == (item["tag"] == "G4"), receipt
    assert abs(receipt["s6fall_share"] - (0.5 if item["tag"] == "G4" else 1 / 3)) < 1e-9
    import pandas as pd
    import numpy as np
    frame = pd.read_csv(folder / "submission.csv")
    assert len(frame) == 3 and frame.shape[1] == 13
    assert frame["StudyInstanceUID"].is_unique
    values = frame.drop(columns="StudyInstanceUID").to_numpy()
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    return receipt


while time.monotonic() < deadline:
    for item in items:
        tag = item["tag"]
        entry = state.setdefault(tag, {"kernel": item["kernel"], "status": "not_started"})
        if entry["status"] == "not_started":
            rc, output = command(item, ["kernels", "push", "-p", item["folder"]])
            match = re.search(r"Kernel version (\d+) successfully pushed", output)
            if rc == 0 and match:
                entry.update(status="running", version=int(match.group(1)))
                print(tag, "PUSHED", entry["version"], flush=True)
            elif any(word in output.lower() for word in ["concurrent", "running kernels", "maximum"]):
                print(tag, "waiting for available GPU session", flush=True)
            else:
                entry.update(status="push_failed", error=output[-2500:])
                print(tag, "PUSH FAILED", output[-1000:], flush=True)
            save()
        if entry["status"] != "running":
            continue
        rc, output = command(item, ["kernels", "status", item["kernel"]])
        if rc != 0:
            print(tag, "STATUS QUERY FAILED", output[-500:], flush=True)
            continue
        if "COMPLETE" not in output and "ERROR" not in output and "CANCEL" not in output:
            continue
        if "COMPLETE" not in output:
            entry.update(status="execution_failed", error=output)
            print(tag, "EXECUTION FAILED", output, flush=True)
            save()
            continue
        folder = HERE.parent / "kaggle_out" / item["slug"]
        rc, output = command(item, ["kernels", "output", item["kernel"], "-p", str(folder)])
        if rc != 0:
            print(tag, "OUTPUT DOWNLOAD FAILED", output[-500:], flush=True)
            continue
        try:
            entry["receipt"] = validate(item, folder)
        except (AssertionError, ValueError, KeyError, OSError) as error:
            entry.update(status="validation_failed", error=str(error))
            print(tag, "VALIDATION FAILED", repr(error), flush=True)
            save()
            continue
        entry["status"] = "validated"
        save()
        print(tag, "VALIDATED", entry["receipt"], flush=True)
        if tag in ("G1", "G2"):
            entry["status"] = "submitting"
            save()
            rc, output = command(item, [
                "competitions", "submit", "-c", "rsna-knee-abnormality-detection",
                "-k", item["kernel"], "-v", str(entry["version"]), "-f", "submission.csv",
                "-m", tag + ": controlled ablation of rsna-s6-convnext 0.957",
            ])
            entry.update(status="submitted" if rc == 0 else "submission_failed",
                         submission_response=output[-3000:])
            print(tag, entry["status"], output[-2000:], flush=True)
            save()
    if all(entry["status"] not in ("not_started", "running")
           for entry in state.values()) and len(state) == len(items):
        break
    time.sleep(45)

print("FINAL STATE", json.dumps(state, indent=2), flush=True)

"""Watch the current Kaggle sessions; exit with diagnostics when intervention is needed."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
JOBS = [
    ("cnxs", "phatdennibba/rsna-tk-train-cnxs-resume-4", "WATCH_TOKEN_CNXS"),
    ("effv2s", "bloctuat/rsna-tk-train-effv2s-resume-2", "WATCH_TOKEN_EFFV2S"),
]


def invoke(args, token):
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = env[token]
    return subprocess.run(
        ["kaggle", *args], env=env, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    deadline = time.monotonic() + args.hours * 3600
    failures = {name: 0 for name, _, _ in JOBS}
    while time.monotonic() < deadline:
        intervention = False
        state = {}
        for name, kernel, token in JOBS:
            try:
                result = invoke(["kernels", "status", kernel], token)
            except subprocess.TimeoutExpired:
                failures[name] += 1
                print(name, "status request timed out", flush=True)
                intervention |= failures[name] >= 3
                continue
            output = result.stdout + result.stderr
            print(time.strftime("%Y-%m-%d %H:%M:%S"), output.strip(), flush=True)
            state[name] = {"kernel": kernel, "status_response": output.strip()}
            if result.returncode:
                failures[name] += 1
                intervention |= failures[name] >= 3
                continue
            failures[name] = 0
            if "RUNNING" in output or "QUEUED" in output:
                continue
            intervention = True
            folder = HERE / "watch-output" / name
            try:
                downloaded = invoke([
                    "kernels", "output", kernel, "-p", str(folder),
                    "--file-pattern", r"\.(json|csv)$",
                ], token)
            except subprocess.TimeoutExpired:
                print(name, "output download timed out; inspect kernel manually", flush=True)
                continue
            print(downloaded.stdout + downloaded.stderr, flush=True)
            if downloaded.returncode:
                continue
            for path in folder.glob("*_log.json"):
                print(name, "METRICS", path.read_text(encoding="utf-8"), flush=True)
            for path in folder.glob("*.log"):
                rows = json.loads(path.read_text(encoding="utf-8"))
                markers = ("FULL_RESUME", "MEMORY", "CGROUP_BREAKDOWN", "EPOCH",
                           "SESSION_", "Error", "RESUME_SAVED", "status=")
                selected = [r.get("data", "") for r in rows
                            if any(m in r.get("data", "") for m in markers)]
                print(name, "DIAGNOSTICS\n" + "".join(selected[-25:]), flush=True)
        (HERE / "watch-status.json").write_text(
            json.dumps({"checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "jobs": state, "intervention_required": intervention}, indent=2),
            encoding="utf-8",
        )
        if intervention:
            print("INTERVENTION_REQUIRED: inspect captured logs before resuming training.", flush=True)
            return 2
        if args.once:
            return 0
        time.sleep(60)
    print("WATCH_WINDOW_ENDED: monitoring stopped after the configured duration.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

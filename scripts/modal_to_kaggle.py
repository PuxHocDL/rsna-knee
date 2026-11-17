"""Copy fold models from the rsna-b-vol volume straight into a PRIVATE Kaggle dataset, from inside a Modal container
(avoids downloading through this machine).  The Kaggle credentials are read from this machine and passed as an argument.
usage: MODAL_PROFILE=<ws> modal run modal_to_kaggle.py --pairs runs/vitS-f2/models/E_prior_s0_f2.pt=VT_s0_f2.pt,... \
       --dataset nguynxunphc662/rsna-vits-models"""
import hashlib, json, os, shutil, subprocess
import modal

app = modal.App("rsna-to-kaggle")
vol = modal.Volume.from_name("rsna-b-vol")
img = modal.Image.debian_slim(python_version="3.11").pip_install("kaggle==2.2.4")


@app.function(image=img, volumes={"/vol": vol}, timeout=1800, cpu=2, memory=4096)
def push(pairs: list, dataset: str, cred: dict):
    vol.reload()
    os.environ.update(cred)
    d = "/tmp/ds"; os.makedirs(d, exist_ok=True)
    sha = {}
    for src, name in pairs:
        shutil.copy("/vol/" + src, os.path.join(d, name))
        sha[name] = hashlib.sha256(open(os.path.join(d, name), "rb").read()).hexdigest()
    json.dump({"title": dataset.split("/")[1], "id": dataset, "licenses": [{"name": "CC0-1.0"}]}, open(d + "/dataset-metadata.json", "w"))
    r = subprocess.run(["kaggle", "datasets", "create", "-p", d], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    if "already" in out.lower() or "exists" in out.lower():
        r = subprocess.run(["kaggle", "datasets", "version", "-p", d, "-m", "update"], capture_output=True, text=True)
        out += " | " + (r.stdout + r.stderr).strip()
    return sha, out[-500:]


@app.local_entrypoint()
def main(pairs: str, dataset: str):
    owner = dataset.split("/")[0]
    acc = json.load(open(os.path.expanduser("~/.kaggle/zoo.json"), encoding="utf-8"))["accounts"][owner]
    tok = acc.get("token") or open(os.path.expanduser("~/.kaggle/access_token"), encoding="utf-8").read().strip()   # default account
    cred = {"KAGGLE_API_TOKEN": tok}
    sha, out = push.remote([p.split("=") for p in pairs.split(",")], dataset, cred)
    for k, v in sha.items():
        print("sha256", k, v)
    print("kaggle:", out)

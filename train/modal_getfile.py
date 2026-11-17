"""Read small files from the rsna-b-vol volume through a Modal function (returns their text over the API channel).
Workaround for local DNS failing on Modal's blob-storage host, which `modal volume get` needs.
usage: MODAL_PROFILE=<ws> modal run modal_getfile.py --paths runs/a.json,runs/b.json --out <local dir>"""
import os
import modal

app = modal.App("rsna-getfile")
vol = modal.Volume.from_name("rsna-b-vol")


@app.function(volumes={"/vol": vol}, timeout=600, cpu=1, memory=1024)
def read(paths: list):
    vol.reload()
    out = {}
    for p in paths:
        f = "/vol/" + p
        out[p] = open(f, encoding="utf-8").read() if os.path.exists(f) else None
    return out


@app.local_entrypoint()
def main(paths: str, out: str):
    os.makedirs(out, exist_ok=True)
    for p, txt in read.remote(paths.split(",")).items():
        name = p.replace("runs/", "").replace("/partial_preds_E_prior_s0.json", ".json").replace("/", "_")
        if txt is None:
            print("missing", p)
            continue
        open(os.path.join(out, name), "w", encoding="utf-8").write(txt)
        print("saved", name, len(txt))

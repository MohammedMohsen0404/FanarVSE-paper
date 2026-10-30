"""Speed-only scaling: replicate the 1M index m times inside each cell (codes, scales, refinement planes), probes fixed,
so the scanned work grows like N. Recall is not meaningful (duplicates); QPS and stage times are."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys, json, time, numpy as np, torch
sys.path.insert(0, f"{FANAR}/src")
from engine3 import make, rf, Engine8
e3, nb = make(256); s1, code = e3.s1, e3.code; dev = "cuda"; out = []
off = s1["off"]
for m in (1, 2, 4):
    seg = [torch.arange(off[c], off[c + 1], device=dev).repeat(m) for c in range(len(off) - 1)]
    idx = torch.cat(seg); offn = np.concatenate([[0], np.cumsum([len(x) for x in seg])])
    s1m = dict(s1); s1m.update(words=s1["words"][idx].contiguous(), scale1=s1["scale1"][idx].contiguous(), order=s1["order"][idx].contiguous(), off=offn,
                              off_t=torch.as_tensor(offn, device=dev))
    cm = dict(code); cm.update(planes=[p[:, idx].contiguous() for p in code["planes"]], scale=code["scale"][:, idx].contiguous(), sig=code["sig"][:, idx].contiguous())
    cm.pop("VP", None)
    e = Engine8(s1m, cm)
    for nprobe, K in ((512, 400), (768, 400), (1024, 800)):
        I, r, C, mq, sq = e.bench(K, 2.0, nprobe)
        x = {"N_millions": round(len(idx) / 1e6, 2), "gpu_GB": round(len(idx) * nb / 1e9, 2), "nprobe": nprobe, "K": K, "QPS": round(mq), "QPS_std": round(sq), "reads": round(float(r.mean()), 2)}
        print(json.dumps(x), flush=True); out.append(x); json.dump(out, open(f"{FANAR}/results/v14_scale.json", "w"), indent=1)
    del e, s1m, cm, idx; torch.cuda.empty_cache()

"""Do repeated queries inflate throughput? Same batch size (2000): 2000 distinct queries vs 1000 queries twice vs 500 queries four times."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, torch
exec(open(FANAR + "/throughput/e_v3final.py").read().split("for nprobe, K, M in")[0].replace("B = int(sys.argv[1])", "B = 2000"))
Q = rf.Qg
for nprobe, K in ((352, 175), (768, 400)):
    E.MSZ = 3072 if K < 250 else 4096
    for rep in (1, 2, 4, 1, 2, 4):
        Qb = torch.cat([Q[:2000 // rep]] * rep).contiguous(); E.graphs.clear(); torch.cuda.empty_cache()
        ms = min(T(lambda: E.search(Qb, K, 1.25, nprobe, 10), 3) for _ in range(3))
        x = dict(tag="dup test", nprobe=nprobe, K=K, batch=2000, distinct=2000 // rep, QPS=round(2000 / ms * 1000)); print(json.dumps(x), flush=True)
        with open(f"{FANAR}/results/throughput_results.jsonl", "a") as fh: fh.write(json.dumps(x) + "\n")

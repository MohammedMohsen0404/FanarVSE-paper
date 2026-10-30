"""Check: fallback routed through the normalised means (no fp16 means copy needed) keeps every query exact; frontier rows with fallbacks."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import json, torch
exec(open(FANAR + "/throughput/e_v3final.py").read().split("for nprobe, K, M in")[0].replace("B = int(sys.argv[1])", "B = 4000"))
for nprobe, K, M in ((256, 150, 3072), (768, 400, 4096), (1536, 600, 8192)):
    E.MSZ = M; row("v3 FINAL fbN", nprobe, K, 1.25)
print("fallback still failing:", getattr(E, "nbad2", 0))

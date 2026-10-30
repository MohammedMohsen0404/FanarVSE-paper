"""Load the reviewer-round engine (rev_fanardb.py) as a module without running an experiment."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import sys
_src = open(f"{FANAR}/src/rev_fanardb.py", encoding="utf-8").read().split("# ---------------------------------------------------------------- experiments")[0]
_argv = sys.argv; sys.argv = [_argv[0], "lib"]
rf = type(sys)("rf"); rf.__dict__["__file__"] = f"{FANAR}/src/rev_fanardb.py"
exec(compile(_src, "rev_fanardb.py", "exec"), rf.__dict__)
sys.argv = _argv

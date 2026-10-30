"""Python wrapper for the batched asynchronous NVMe reader (native/nvme_reader.dll, compiled with the TCC that ships with Triton)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import ctypes, numpy as np
_DLL = f"{FANAR}/src/native/nvme_reader.dll"
if not _os.path.exists(_DLL):                                   # build once with the TCC that ships with Triton (Windows)
    import subprocess, triton
    subprocess.run([_os.path.join(_os.path.dirname(triton.__file__), "runtime", "tcc", "tcc.exe"), "-shared", "-o", _DLL, _DLL[:-4] + ".c"], check=True)
_dll = ctypes.CDLL(_DLL)
_dll.open_tier.restype = ctypes.c_void_p; _dll.open_tier.argtypes = [ctypes.c_wchar_p]
_dll.make_port.restype = ctypes.c_void_p; _dll.make_port.argtypes = [ctypes.c_void_p]
_dll.alloc_bufs.restype = ctypes.c_void_p; _dll.alloc_bufs.argtypes = [ctypes.c_int]
_dll.read_rows.restype = ctypes.c_int
_dll.read_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
class AsyncTier:
    def __init__(self, path, d, qd=64):
        self.h = _dll.open_tier(path); assert self.h and self.h != ctypes.c_void_p(-1).value, "open failed"
        self.port = _dll.make_port(self.h); self.qd = qd; self.bufs = _dll.alloc_bufs(qd); self.d = d
    def read_into(self, ids, out):
        """ids: int64 numpy; out: contiguous uint8/fp16 buffer (numpy or pinned torch) of len(ids) x d fp16."""
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        ptr = out.data_ptr() if hasattr(out, "data_ptr") else out.ctypes.data
        r = _dll.read_rows(self.h, self.port, ids.ctypes.data, len(ids), ptr, self.d * 2, self.qd, self.bufs)
        if r != len(ids): raise OSError(f"read_rows returned {r}")
        return out

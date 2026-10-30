"""ctypes binding to cuVS 26.10 IVF-RaBitQ (C++ only in this release; no Python/C API).
ABI (Itanium x86-64): index<long> has a non-trivial destructor -> returned through a hidden sret pointer;
mdspan<..., extents<long,dyn,dyn>, layout_right, accessor> = {T* ptr; int64 rows; int64 cols} (24 B, trivially copyable,
passed by value in memory). raft::resources comes from cuvs.common.Resources (cuvsResources_t is a raft::resources*)."""
import os as _os
FANAR = _os.environ.get("FANAR_ROOT", _os.getcwd()).replace("\\", "/")   # repo root (run scripts from it)
import ctypes, site, os, cupy as cp, numpy as np
from cuvs.common import Resources

LIB = ctypes.CDLL(os.path.join(site.getsitepackages()[0], "libcuvs/lib64/libcuvs.so"))
DEV, HOST = "2", "0"
def _mangle_build(mem):
    return ("_ZN4cuvs9neighbors10ivf_rabitq5buildERKN4raft9resourcesERKNS1_12index_paramsEN4cuda3std3__46mdspanIKfNSB_7extents"
            "IlJLm18446744073709551615ELm18446744073709551615EEEENSB_12layout_rightENS2_20host_device_accessorINSB_16default_accessor"
            f"ISD_EELNS2_11memory_typeE{mem}EEEEE")
SEARCH = ("_ZN4cuvs9neighbors10ivf_rabitq6searchERKN4raft9resourcesERKNS1_13search_paramsERNS1_5indexIlEEN4cuda3std3__46mdspanIKfNSE_7"
          "extentsIlJLm18446744073709551615ELm18446744073709551615EEEENSE_12layout_rightENS2_20host_device_accessorINSE_16default_accessor"
          "ISG_EELNS2_11memory_typeE2EEEEENSF_IlSI_SJ_NSK_INSL_IlEELSN_2EEEEENSF_IfSI_SJ_NSK_INSL_IfEELSN_2EEEEE")

class View(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p), ("rows", ctypes.c_int64), ("cols", ctypes.c_int64)]

class IndexParams(ctypes.Structure):
    _fields_ = [("metric", ctypes.c_int), ("metric_arg", ctypes.c_float), ("n_lists", ctypes.c_uint32), ("bits_per_dim", ctypes.c_uint32),
                ("kmeans_n_iters", ctypes.c_uint32), ("max_train_points_per_cluster", ctypes.c_uint32), ("fast_quantize_flag", ctypes.c_bool),
                ("streaming_batch_size", ctypes.c_size_t), ("force_streaming", ctypes.c_bool)]

class SearchParams(ctypes.Structure):
    _fields_ = [("n_probes", ctypes.c_uint32), ("mode", ctypes.c_int)]
MODES = {"LUT16": 0, "LUT32": 1, "QUANT4": 2, "QUANT8": 3}

_build = {m: getattr(LIB, _mangle_build(m)) for m in (DEV, HOST)}
for f in _build.values(): f.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(IndexParams), View]; f.restype = ctypes.c_void_p
_dtor = getattr(LIB, "_ZN4cuvs9neighbors10ivf_rabitq5indexIlED1Ev"); _dtor.argtypes = [ctypes.c_void_p]; _dtor.restype = None
_search = getattr(LIB, SEARCH); _search.argtypes = [ctypes.c_void_p, ctypes.POINTER(SearchParams), ctypes.c_void_p, View, View, View]; _search.restype = None


class IVFRaBitQ:
    def __init__(self, n_lists=1024, bits_per_dim=3, kmeans_n_iters=20, max_train=256):
        self.res = Resources(); self.p = IndexParams(0, 2.0, n_lists, bits_per_dim, kmeans_n_iters, max_train, True, 100000, False)
        self.buf = ctypes.create_string_buffer(64); self.built = False                 # holds index<long> (one unique_ptr)

    def build(self, X):
        """X: cupy float32 (device) or numpy float32 / memmap (host, streamed)."""
        if isinstance(X, cp.ndarray): v, m = View(X.data.ptr, *X.shape), DEV
        else: v, m = View(X.ctypes.data if isinstance(X, np.ndarray) else X, *X.shape), HOST
        _build[m](ctypes.addressof(self.buf), self.res.get_c_obj(), ctypes.byref(self.p), v); self.res.sync(); self._keep = X; self.built = True

    def free(self):
        if self.built: _dtor(ctypes.addressof(self.buf)); self.res.sync(); self.built = False

    def search(self, Q, k, n_probes, mode="QUANT4"):
        Q = cp.ascontiguousarray(Q, dtype=cp.float32); I = cp.empty((len(Q), k), cp.int64); D = cp.empty((len(Q), k), cp.float32)
        sp = SearchParams(n_probes, MODES[mode])
        _search(self.res.get_c_obj(), ctypes.byref(sp), ctypes.addressof(self.buf), View(Q.data.ptr, *Q.shape), View(I.data.ptr, *I.shape), View(D.data.ptr, *D.shape))
        self.res.sync(); return I, D


if __name__ == "__main__":                                                               # smoke test against brute force on 100k vectors
    X = np.load(f"{FANAR}/data/oai1m_base_fp16.npy", mmap_mode="r")[:100000].astype(np.float32); Q = np.load(f"{FANAR}/data/oai1m_q.npy")[:500]
    Xg = cp.asarray(X); Qg = cp.asarray(Q); gt = cp.asnumpy(cp.argsort(-(Qg @ Xg.T), 1)[:, :10])
    for bits in (1, 3, 5):
        idx = IVFRaBitQ(n_lists=256, bits_per_dim=bits); idx.build(Xg)
        for npb in (8, 32):
            I, D = idx.search(Qg, 10, npb); I = cp.asnumpy(I)
            print(bits, npb, np.mean([len(set(I[i]) & set(gt[i])) / 10 for i in range(len(Q))]), I[0, :5], gt[0, :5], flush=True)

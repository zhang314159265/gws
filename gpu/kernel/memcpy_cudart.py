import ctypes
from ctypes import c_void_p, c_size_t, c_int

_memcpy = ctypes.CDLL("libcudart.so.12")["cudaMemcpy"]
_memcpy.argtypes = [
    c_void_p,
    c_void_p,
    c_size_t,
    c_int,
]
_d2d = 3

def memcpy_cudart(src, dst):
    _memcpy(dst.data_ptr(), src.data_ptr(), src.numel() * src.itemsize, _d2d)

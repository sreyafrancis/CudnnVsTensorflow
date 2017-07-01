#Inner_product_Cublas

from __future__ import absolute_import
import sys
import ctypes
import ctypes.util




try:
	libcublas=ctypes.cdll.LoadLibrary('/usr/local/cuda-7.5/targets/x86_64-linux/lib/libcublas.so.8.0')
except OSError:
	print("OSError");
print(libcublas.cublasGetVersion())
_libcublas.cublasSgemm_v2.restype = int
_libcublas.cublasSgemm_v2.argtypes = [_types.handle,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_int,
                                      ctypes.c_void_p,
                                      ctypes.c_void_p,
                                      ctypes.c_int]
[docs]
def cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """
    Matrix-matrix product for real single precision general matrix.

    References
    ----------
    `cublas<t>gemm <http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm>`_
    """

    status = _libcublas.cublasSgemm_v2(handle,
                                       _CUBLAS_OP[transa],
                                       _CUBLAS_OP[transb], m, n, k,
                                       ctypes.byref(ctypes.c_float(alpha)),
                                       int(A), lda, int(B), ldb,
                                       ctypes.byref(ctypes.c_float(beta)),
                                       int(C), ldc)
    cublasCheckStatus(status)

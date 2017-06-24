#iplayer_pycuda

import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.cublas

#skcuda.cublas.cublasCheckStatus(status)

m=1
k=3
n=4
x = np.float32(np.random.rand(m,k))
y = np.float32(np.random.rand(k,n))

A = gpuarray.to_gpu(x)
B = gpuarray.to_gpu(y)
C =gpuarray.empty((m,n), np.float32)

h =skcuda.cublas.cublasCreate()

#transa=_CUBLAS_OP[transa]

lda=m
ldb=k
ldc=m
#alf = 1.0;
#bet = 0.0;
#const float *alpha = &alf;
#const float *beta = &bet;
alpha=1.0
beta=0.0

#d=skcuda.cublas.cublasSgemm(h, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
status=skcuda.cublas.cublasSgemm(h, 0, 0, m, n, k, alpha,A.gpudata, lda,B.gpudata, ldb, beta,C.gpudata, ldc)
#skcuda.cublas.cublasCheckStatus(status)
skcuda.cublas.cublasDestroy(h)
print 'Input vector'
print '----------------------------------'
print x
print 'Weight Vector'
print '----------------------------------'
print y
print 'Inner product'
print '----------------------------------'
print C.get()
#print('Success status: ', np.allclose(np.dot(a, np.dot(b, c)), d_gpu.get()))
#np.allclose(d, np.dot(x, y))

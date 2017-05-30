import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.cublas



#skcuda.cublas.cublasCheckStatus(status)


x = np.float32(np.random.rand(5))
y = np.float32(np.random.rand(5))
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
h =skcuda.cublas.cublasCreate()
d =skcuda.cublas.cublasSdot(h, x_gpu.size, x_gpu.gpudata, 1, y_gpu.gpudata, 1)
skcuda.cublas.cublasDestroy(h)
print d
#print('Success status: ', np.allclose(np.dot(a, np.dot(b, c)), d_gpu.get()))
np.allclose(d, np.dot(x, y))

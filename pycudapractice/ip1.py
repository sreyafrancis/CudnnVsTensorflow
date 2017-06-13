					#FULLYCONNECTED LAYER
					#----------------------- 
    
import numpy as np
from pycuda import driver, compiler, gpuarray, tools
   
# initialize the device
import pycuda.autoinit
import time
   
kernel_code_template = """
	__global__ void MatrixMulKernel(float *a, float *b, float *c, float *y)
        {
        // 2D Thread ID (assuming that only *one* block will be executed)
        int tx = threadIdx.x;
        int ty = threadIdx.y;
   
        // Pvalue is used to store the element of the matrix
        // that is computed by the thread
        float Pvalue = 0;
   
        // Each thread loads one row of M and one column of N, 
        //   to produce one element of P.
        for (int k = 0; k < %(INPUT_VECTOR_WIDTH)s; ++k) {
        	float Aelement = a[ty * %(INPUT_VECTOR_WIDTH)s + k];
                float Belement = b[k * %(WEIGHT_MATRIX_WIDTH)s + tx];
                Pvalue += Aelement * Belement;
       		}
   
        // Write the matrix to device memory;
        // each thread writes one element
        c[ty * %(WEIGHT_MATRIX_WIDTH)s + tx] = Pvalue;
	y[ty * %(WEIGHT_MATRIX_WIDTH)s + tx] =c[ty * %(WEIGHT_MATRIX_WIDTH)s + tx] + 1; 
	}
"""
   

INPUT_VECTOR_HEIGHT =4 
INPUT_VECTOR_WIDTH = 6
WEIGHT_MATRIX_HEIGHT =6
WEIGHT_MATRIX_WIDTH =2

# create random INPUT AND WEIGHT matrices

a_cpu = np.random.randn(INPUT_VECTOR_HEIGHT ,INPUT_VECTOR_WIDTH ).astype(np.float32)
b_cpu = np.random.randn(WEIGHT_MATRIX_HEIGHT,WEIGHT_MATRIX_WIDTH ).astype(np.float32)
#bias_cpu = np.random.randn(INPUT_VECTOR_HEIGHT,WEIGHT_MATRIX_WIDTH ).astype(np.float32)
#a_cpu = np.random.randint(3, size=(INPUT_VECTOR_HEIGHT ,INPUT_VECTOR_WIDTH))
#b_cpu = np.random.randint(5, size=(WEIGHT_MATRIX_HEIGHT,WEIGHT_MATRIX_WIDTH))   



start = time.time()
# compute reference on the CPU to verify GPU computation

c_cpu = np.dot(a_cpu, b_cpu)

end = time.time()
  
# transfer host (CPU) memory to device (GPU) memory 
a_gpu = gpuarray.to_gpu(a_cpu) 
b_gpu = gpuarray.to_gpu(b_cpu)
#bias_gpu = gpuarray.to_gpu(bias_cpu)   
# create empty gpu array for the PRODUCT
c_gpu = gpuarray.empty((INPUT_VECTOR_HEIGHT, WEIGHT_MATRIX_WIDTH), np.float32)
y_gpu = gpuarray.empty((INPUT_VECTOR_HEIGHT, WEIGHT_MATRIX_WIDTH), np.float32)
# get the kernel code from the template 
# by specifying the constant MATRIX_SIZE OF INPUT VECTOR AND WEIGHT
kernel_code = kernel_code_template % {
	'WEIGHT_MATRIX_WIDTH': WEIGHT_MATRIX_WIDTH,
        'INPUT_VECTOR_WIDTH': INPUT_VECTOR_WIDTH,
        }
   



# compile the kernel code 
mod = compiler.SourceModule(kernel_code)
   
# get the kernel function from the compiled module
matrixmul = mod.get_function("MatrixMulKernel")
   
start_time = time.time()
# call the kernel on the card
matrixmul(
	# inputs
	a_gpu, b_gpu, 
	# output
        c_gpu, 
	#after adding bial
	y_gpu,
	# (only one) block of MATRIX_SIZE x MATRIX_SIZE threads
        block = ( 16,16, 1),
	#block = (INPUT_VECTOR_HEIGHT, WEIGHT_MATRIX_WIDTH, 1),
        grid = (int( WEIGHT_MATRIX_WIDTH/16)+1, int(INPUT_VECTOR_HEIGHT/16)+1),
	)


end_time=time.time()
  
# print the results
print "-" * 80
print "Fully Connected Layer"


print "-" * 80
print "Input vector X (GPU):"
print a_gpu.get()
   
print "-" * 80
print "Weight Matrix W (GPU):"
print b_gpu.get()
   
print "-" * 80
print "WX (GPU):"
print c_gpu.get()

print "-" * 80
print "WX (CPU):"
print c_cpu
   
print "-" * 80
print "CPU-GPU difference:"
print c_cpu - c_gpu.get()

print "-" * 80
print "Inner Product Y =Wx+b(GPU):"
print y_gpu.get()

print("\n\n--- %s seconds ---FOR CPU\n\n" % (end - start))
  
print("--- %s seconds ---FOR GPU" % (end_time - start_time)) 

np.allclose(c_cpu, c_gpu.get())

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void hello_kernel() {
    printf("  [GPU] Hello from thread %d\n", threadIdx.x);
}

void run_basic_cuda_test() {
    hello_kernel<<<1, 1>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("  [Error] CUDA Error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    printf("  [CPU] Basic CUDA test success.\n");
}


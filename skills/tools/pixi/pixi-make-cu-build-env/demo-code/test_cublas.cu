#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void run_cublas_test() {
    cublasHandle_t handle;
    cublasStatus_t stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf("  [Error] cuBLAS initialization failed\n");
        exit(1);
    }
    
    int version;
    cublasGetVersion(handle, &version);
    printf("  [Info] cuBLAS Version: %d\n", version);

    cublasDestroy(handle);
    printf("  [CPU] cuBLAS test success.\n");
}


#include <stdio.h>
#include <cudnn.h>

void run_cudnn_test() {
    cudnnHandle_t handle;
    cudnnStatus_t stat = cudnnCreate(&handle);
    if (stat != CUDNN_STATUS_SUCCESS) {
        printf("  [Error] cuDNN initialization failed: %s\n", cudnnGetErrorString(stat));
        exit(1);
    }
    
    printf("  [Info] cuDNN Version: %zu\n", cudnnGetVersion());

    cudnnDestroy(handle);
    printf("  [CPU] cuDNN test success.\n");
}


#include <iostream>
#include <vector>
#include <string>

// Forward declarations
void run_basic_cuda_test();

#ifdef USE_CUBLAS
void run_cublas_test();
#endif

#ifdef USE_CUDNN
void run_cudnn_test();
#endif

#ifdef USE_NCCL
void run_nccl_test();
#endif

int main() {
    std::cout << "[BuildCheck] Starting verification..." << std::endl;

    std::cout << "[BuildCheck] Running Basic CUDA Test..." << std::endl;
    run_basic_cuda_test();

#ifdef USE_CUBLAS
    std::cout << "[BuildCheck] Running cuBLAS Test..." << std::endl;
    run_cublas_test();
#endif

#ifdef USE_CUDNN
    std::cout << "[BuildCheck] Running cuDNN Test..." << std::endl;
    run_cudnn_test();
#endif

#ifdef USE_NCCL
    std::cout << "[BuildCheck] Running NCCL Test..." << std::endl;
    run_nccl_test();
#endif

    std::cout << "[BuildCheck] All enabled tests passed." << std::endl;
    return 0;
}

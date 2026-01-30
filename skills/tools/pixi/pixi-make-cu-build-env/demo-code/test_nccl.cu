#include <stdio.h>
#include <nccl.h>

void run_nccl_test() {
    int version;
    ncclResult_t res = ncclGetVersion(&version);
    if (res != ncclSuccess) {
        printf("  [Error] NCCL get version failed: %s\n", ncclGetErrorString(res));
        exit(1);
    }
    
    printf("  [Info] NCCL Version: %d\n", version);
    printf("  [CPU] NCCL test success.\n");
}


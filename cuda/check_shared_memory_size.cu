#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        printf("No CUDA capable devices found\n");
        return 1;
    }

    // Select the first device
    int deviceId = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    printf("Shared memory per block: %lu bytes\n", deviceProp.sharedMemPerBlock);

    return 0;
}

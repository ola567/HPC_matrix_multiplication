#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
extern "C" {
    #include "utils.h"
}

#define BLOCK_SIZE 16
#define NUMBER_OF_RUNS 10

__host__ void errorexit(const char *s)
{
    printf("\n%s\n", s);
    exit(EXIT_FAILURE);
}

__global__ void matrixMultiplication(int *deviceA, int *deviceB, int *deviceC, int rowsA, int colsA, int colsB)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < colsB && row < rowsA) 
    {
        for(int i = 0; i < colsA; i++) 
        {
            sum += deviceA[row * colsA + i] * deviceB[i * colsB + col];
        }
        deviceC[row * colsB + col] = sum;
    }
} 

int main(int argc, char **argv)
{
    char experiment_filename[] = "../experiment_data/16000.txt";
    int *A, *B, *C;
    int rowsA, colsA, rowsB, colsB;
    float milliseconds = 0;

    // Read matrices from file
    read_matrices_from_file(experiment_filename, &A, &B, &rowsA, &colsA, &rowsB, &colsB);
    
    for(int i = 0; i < NUMBER_OF_RUNS; i++) {
        C = (int *)calloc(rowsA * colsB, sizeof(int));

        // CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);

        // Allocate device memory
        int *deviceA, *deviceB, *deviceC;
        cudaMalloc((void **)&deviceA, rowsA * colsA * sizeof(int));
        cudaMalloc((void **)&deviceB, rowsB * colsB * sizeof(int));
        cudaMalloc((void **)&deviceC, rowsA * colsB * sizeof(int));

        // Copy data to device
        cudaMemcpy(deviceA, A, rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, B, rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(deviceC, 0, rowsA * colsB * sizeof(int));

        // Define grid and block dimensions
        dim3 blocks((colsB + BLOCK_SIZE - 1) / BLOCK_SIZE, (rowsA + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

        // Launch kernel
        matrixMultiplication<<<blocks, threads>>>(deviceA, deviceB, deviceC, rowsA, colsA, colsB);

        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            errorexit(cudaGetErrorString(err));
        }

        cudaMemcpy(C, deviceC, rowsA * colsB * sizeof(int), cudaMemcpyDeviceToHost);

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Kernel execution time: %.3f ms\n", milliseconds);

        // Free allocated memory
        free(C);
        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
    }

    // Free alocated memory
    free(A);
    free(B);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
extern "C" {
    #include "utils.h"
}

#define BLOCK_SIZE 32
#define TILE_SIZE 32
#define NUMBER_OF_RUNS 10

__host__ void errorexit(const char *s)
{
    printf("\n%s\n", s);
    exit(EXIT_FAILURE);
}

__global__ void matrixMultiplication(int *deviceA, int *deviceB, int *deviceC, int rowsA, int colsA, int colsB)
{ 
    __shared__ int sharedA[TILE_SIZE][TILE_SIZE]; 
    __shared__ int sharedB[TILE_SIZE][TILE_SIZE];

    int globalRow = blockDim.y * blockIdx.y + threadIdx.y;
    int globalCol = blockDim.x * blockIdx.x + threadIdx.x;
    int value = 0;
    sharedA[threadIdx.y][threadIdx.x] = 0;
    sharedB[threadIdx.y][threadIdx.x] = 0;

    for (int i = 0; i < (((colsA - 1) / TILE_SIZE) + 1); i++) {
        if ((globalRow < rowsA) && (threadIdx.x + (i * TILE_SIZE)) < colsA) {
            sharedA[threadIdx.y][threadIdx.x] = deviceA[(globalRow * colsA) + threadIdx.x + (i * TILE_SIZE)];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0;
        }
        if (globalCol < colsB && (threadIdx.y + i * TILE_SIZE) < colsA) {
            sharedB[threadIdx.y][threadIdx.x] = deviceB[(threadIdx.y + i * TILE_SIZE) * colsB + globalCol];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();

        for (int j = 0; j < TILE_SIZE; ++j) {
            value += sharedA[threadIdx.y][j] * sharedB[j][threadIdx.x];
        }
        __syncthreads();

    }
    if (globalRow < rowsA && globalCol < colsB) {
        deviceC[globalRow * colsB + globalCol] = value;
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
        dim3 blocks((colsB + TILE_SIZE - 1) / TILE_SIZE, (rowsA + TILE_SIZE - 1) / TILE_SIZE);
        dim3 threads(TILE_SIZE, TILE_SIZE);

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

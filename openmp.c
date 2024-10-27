#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "omp.h"

void omp_multiply_matrixes_not_optimized(int** A, int** B, int** C, int rowsA, int colsA, int rowsB, int colsB) {
    #pragma omp parallel for 
    for (int row = 0; row < rowsA; row++) {
        for (int col = 0; col < colsB; col++) {
            for (int i = 0; i < colsA; i++) {
                C[row][col] += A[row][i] * B[i][col];
            }
        }
    }
}

void omp_multiply_matrixes_optimized(int** A, int** B, int** C, int rowsA, int colsA, int rowsB, int colsB) {
    #pragma omp parallel for
    for (int row = 0; row < rowsA; row++) {
        for (int i = 0; i < colsA; i++) {
            for (int col = 0; col < colsB; col++) {
                C[row][col] += A[row][i] * B[i][col];
            }
        }
    }
}



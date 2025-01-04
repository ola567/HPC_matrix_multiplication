#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include "utils.h"


void read_matrices_from_file(const char *filename, int **A, int **B, int *rowsA, int *colsA, int *rowsB, int *colsB) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Cannot open file");
        return;
    }

    fscanf(file, "%d %d", rowsA, colsA);
    *A = (int *)malloc((*rowsA) * (*colsA) * sizeof(int));
    if (*A == NULL) {
        perror("Error in alocating mememory for first matrix");
        fclose(file);
        return;
    }
    for (int i = 0; i < *rowsA; ++i) {
        for (int j = 0; j < *colsA; ++j) {
            fscanf(file, "%d", &((*A)[i * (*colsA) + j]));
        }
    }

    fscanf(file, "%d %d", rowsB, colsB);
    *B = (int *)malloc((*rowsB) * (*colsB) * sizeof(int));
    if (*B == NULL) {
        perror("Error in alocating mememory for second matrix");
        free(*A);
        fclose(file);
        return;
    }
    for (int i = 0; i < *rowsB; ++i) {
        for (int j = 0; j < *colsB; ++j) {
            fscanf(file, "%d", &((*B)[i * (*colsB) + j]));
        }
    }

    fclose(file);
}

void print_matrix(int *matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int compareArrays(int *array1, int *array2, int size) {
    for (int i = 0; i < size; i++) {
        if (array1[i] != array2[i]) {
            return 0;
        }
    }
    return 1;
}

void sequential_matrix_multiplication(int *A, int *B, int *sequentialC, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {            
        for (int j = 0; j < colsB; j++) {         
            for (int k = 0; k < colsA; k++) {
                sequentialC[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        } 
    }
}

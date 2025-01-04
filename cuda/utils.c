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
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

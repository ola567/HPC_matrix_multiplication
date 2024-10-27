#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include "utils.h"

void generate_matrix(int** matrix, int rows, int cols) {
    srand(time(NULL));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % 100;
        }
    }
}

int** allocate_matrix(int rowsA, int colsB) {
    int **C = (int**)malloc(rowsA * sizeof(int*));
    if (C == NULL) {
        printf("Result matrix memory allocation error\n");
        exit(1);
    }
    
    for (int i = 0; i < rowsA; i++) {
        C[i] = (int*)calloc(colsB, sizeof(int));
        if (C[i] == NULL) {
            printf("Result matrix memory allocation error\n");
            exit(1);
        }
    }
    return C;
}

void free_matrix(int** matrix, int rows) {
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void print_matrix(int** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

void save_matrices_to_file(const char* filename, int** A, int rowsA, int colsA, int** B, int rowsB, int colsB) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Cannot open the file\n");
        exit(1);
    }

    fprintf(file, "%d %d\n", rowsA, colsA);
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            fprintf(file, "%d ", A[i][j]);
        }
        fprintf(file, "\n");
    }

    fprintf(file, "%d %d\n", rowsB, colsB);
    for (int i = 0; i < rowsB; i++) {
        for (int j = 0; j < colsB; j++) {
            fprintf(file, "%d ", B[i][j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void read_matrices_from_file(const char* filename, int*** A, int*** B, int* rowsA, int* colsA, int* rowsB, int* colsB) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Cannot open the file\n");
        exit(1);
    }

    // read matrix A dimensions
    fscanf(file, "%d %d", rowsA, colsA);
    *A = (int**) malloc(*rowsA * sizeof(int*));
    for (int i = 0; i < *rowsA; i++) {
        (*A)[i] = (int*) malloc(*colsA * sizeof(int));
    }
    for (int i = 0; i < *rowsA; i++) {
        for (int j = 0; j < *colsA; j++) {
            fscanf(file, "%d", &((*A)[i][j]));
        }
    }

    // read matrix B dimensions
    fscanf(file, "%d %d", rowsB, colsB);
    *B = (int**) malloc(*rowsB * sizeof(int*));
    for (int i = 0; i < *rowsB; i++) {
        (*B)[i] = (int*) malloc(*colsB * sizeof(int));
    }
    for (int i = 0; i < *rowsB; i++) {
        for (int j = 0; j < *colsB; j++) {
            fscanf(file, "%d", &((*B)[i][j]));
        }
    }

    fclose(file);
}

void zeroMatrix(int** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = 0;
        }
    }
}

long get_time(struct timeval *start, struct timeval *stop) {
  long time=1000000*(stop->tv_sec-start->tv_sec)+stop->tv_usec-start->tv_usec;
  return time;
}

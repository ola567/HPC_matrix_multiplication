#include <stdio.h>
#include <stdlib.h>


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

void print_matrix(int** matrix, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}

int main() {
    int **A, **B;
    int rowsA, colsA, rowsB, colsB;

    read_matrices_from_file("../../matrixes.txt", &A, &B, &rowsA, &colsA, &rowsB, &colsB);
    printf("Matrix A (%d x %d):\n", rowsA, colsA);
    print_matrix(A, rowsA, colsA);
    
    printf("\nMatrix B (%d x %d):\n", rowsB, colsB);
    print_matrix(B, rowsB, colsB);

    for (int i = 0; i < rowsA; i++) {free(A[i]);}
    free(A);
    for (int i = 0; i < rowsB; i++) {free(B[i]);}
    free(B);

    return 0;
}

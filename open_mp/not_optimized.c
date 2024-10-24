#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "utils.h"
#include "omp.h"


void generate_matrix(int** matrix, int rows, int cols) {
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

int main(int argc, char **argv) {
    int **A, **B, **C;
    int rowsA, colsA, rowsB, colsB;
    long long threads_nr = 10;
    srand(time(NULL));

    // get input data
    // read_matrices_from_file("../matrixes.txt", &A, &B, &rowsA, &colsA, &rowsB, &colsB);
    // print_matrix(A, rowsA, colsA);
    // print_matrix(B, rowsB, colsB);

    // generate matrixes
    rowsA = atoi(argv[1]);
    colsA = atoi(argv[2]);
    rowsB = atoi(argv[3]);
    colsB = atoi(argv[4]);
    A = allocate_matrix(rowsA, colsA);
    B = allocate_matrix(rowsB, colsB);
    generate_matrix(A, rowsA, colsA);
    generate_matrix(B, rowsB, colsB);
    print_matrix(A, rowsA, colsA);
    print_matrix(B, rowsB, colsB);

    // allocate memory for result data
    C = allocate_matrix(rowsA, colsB);

    // set number of threads
    omp_set_num_threads(rowsA * colsB);

    struct timeval ins__tstart, ins__tstop;
    gettimeofday(&ins__tstart, NULL);
    
    // perform computations
    #pragma omp parallel
    {
        long int thread_id = omp_get_thread_num();
        long int row_number = thread_id / colsB;
        long int col_number = thread_id % colsB;
        for (long int i = 0; i < colsA; i++) {
            C[row_number][col_number] += A[row_number][i] * B[i][col_number];
        }
    }
        
    // synchronize/finalize computations
    gettimeofday(&ins__tstop, NULL);

    // print rezult matrix
    print_matrix(C, rowsA, colsB);

    // print execution time
    ins__printtime(&ins__tstart, &ins__tstop);

    // free memory
    free_matrix(A, rowsA);
    free_matrix(B, rowsB);
    free_matrix(C, rowsA);

    return 0;
}

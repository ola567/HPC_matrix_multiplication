#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "../utils.h"


int main(int argc, char **argv) {
    int **A, **B, **C;
    int rowsA, colsA, rowsB, colsB;
    long long threads_nr = 10;
    srand(time(NULL));

    // generate matrixes
    rowsA = atoi(argv[1]);
    colsA = atoi(argv[2]);
    rowsB = atoi(argv[3]);
    colsB = atoi(argv[4]);
    A = allocate_matrix(rowsA, colsA);
    B = allocate_matrix(rowsB, colsB);
    generate_matrix(A, rowsA, colsA);
    generate_matrix(B, rowsB, colsB);
    save_matrices_to_file("../matrixes.txt", A, rowsA, colsA, B, rowsB, colsB);

    // allocate memory for result data
    C = allocate_matrix(rowsA, colsB);

    struct timeval ins__tstart, ins__tstop;
    gettimeofday(&ins__tstart, NULL);
    
    // perform computations
    if (colsA != rowsB) {
        printf("Matrix multiplication not possible with the given dimensions.\n");
        exit(1);
    }
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            for (int k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
        
    // synchronize/finalize computations
    gettimeofday(&ins__tstop, NULL);

    // print rezult matrix
    print_matrix(C, rowsA, colsB);

    // print execution time
    print_time(&ins__tstart, &ins__tstop);

    // free memory
    free_matrix(A, rowsA);
    free_matrix(B, rowsB);
    free_matrix(C, rowsA);

    return 0;
}

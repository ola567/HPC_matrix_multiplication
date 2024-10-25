#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "../utils.h"
#include "omp.h"


int main(int argc, char **argv) {
    int **A, **B, **C;
    int rowsA, colsA, rowsB, colsB;
    srand(time(NULL));

    // get input data
    read_matrices_from_file("../matrixes.txt", &A, &B, &rowsA, &colsA, &rowsB, &colsB);

    // allocate memory for result data
    C = allocate_matrix(rowsA, colsB);

    // set number of threads
    omp_set_num_threads(omp_get_max_threads());

    struct timeval ins__tstart, ins__tstop;
    gettimeofday(&ins__tstart, NULL);
    
    // perform computations
    #pragma omp parallel for collapse(2) shared(A, B, C)
    for (int row_number = 0; row_number < rowsA; row_number++) {
        for (int col_number = 0; col_number < colsB; col_number++) {
            for (int i = 0; i < colsA; i++) {
                C[row_number][col_number] += A[row_number][i] * B[i][col_number];
            }
        }
    }
        
    // synchronize/finalize computations
    gettimeofday(&ins__tstop, NULL);

    // print rezult matrix
    // print_matrix(C, rowsA, colsB);

    // print execution time
    print_time(&ins__tstart, &ins__tstop);

    // free memory
    free_matrix(A, rowsA);
    free_matrix(B, rowsB);
    free_matrix(C, rowsA);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "utils.h"
#include "omp.h"


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
    print_time(&ins__tstart, &ins__tstop);

    // free memory
    free_matrix(A, rowsA);
    free_matrix(B, rowsB);
    free_matrix(C, rowsA);

    return 0;
}

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "utils.h"
#include "sequential.h"
#include "openmp.h"


int main(int argc, char **argv) {
    int **A, **B, **C;
    int rowsA, colsA, rowsB, colsB;
    int dimension = 2000;
    rowsA = dimension;
    rowsB = dimension;
    colsA = dimension;
    colsB = dimension;
    // srand(time(NULL));

    // get input data
    // read_matrices_from_file("matrixes.txt", &A, &B, &rowsA, &colsA, &rowsB, &colsB);
    A = allocate_matrix(rowsA, colsA);
    B = allocate_matrix(rowsB, colsB);
    generate_matrix(A, rowsA, colsA);
    generate_matrix(B, rowsB, colsB);

    // allocate memory for result data
    C = allocate_matrix(rowsA, colsB);

    // start time measurement
    struct timeval ins__tstart, ins__tstop;
    // without parallelization
    // gettimeofday(&ins__tstart, NULL);
    // sequential_matrix_multiplication(A, B, C, rowsA, colsA, rowsB, colsB);
    // gettimeofday(&ins__tstop, NULL);
    // print_time(&ins__tstart, &ins__tstop, "Without parallelization");
    // print_matrix(C, rowsA, colsB);
    
    // zeroMatrix(C, rowsA, colsB);

    // with parallelization not optimized
    gettimeofday(&ins__tstart, NULL);
    omp_multiply_matrixes_not_optimized(A, B, C, rowsA, colsA, rowsB, colsB);
    gettimeofday(&ins__tstop, NULL);
    print_time(&ins__tstart, &ins__tstop, "With parallelization not optimized");
    
    zeroMatrix(C, rowsA, colsB);

    // with parallelization optimized
    gettimeofday(&ins__tstart, NULL);
    omp_multiply_matrixes_optimized(A, B, C, rowsA, colsA, rowsB, colsB);
    gettimeofday(&ins__tstop, NULL);
    print_time(&ins__tstart, &ins__tstop, "With parallelization optimized");

    // free memory
    free_matrix(A, rowsA);
    free_matrix(B, rowsB);
    free_matrix(C, rowsA);

    return 0;
}

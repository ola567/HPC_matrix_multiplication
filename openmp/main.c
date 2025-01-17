#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "sequential.h"
#include "openmp.h"


int main(int argc, char **argv) {
    int **A, **B, **C;
    struct timeval ins__tstart, ins__tstop;
    int dimensions[1] = {1400};
    char experiment_file_path[100];
    int rowsA, colsA, rowsB, colsB;
    FILE *result_file = fopen("results.txt", "w");
    if (result_file == NULL) {
        printf("Cannot open the file\n");
        exit(1);
    }

    for (int i = 0; i < 1; i++) {
        fprintf(result_file, "Matrix dimension: %d\n", dimensions[i]);
        for (int j = 0; j < 10; j++) {
            sprintf(experiment_file_path, "experiment_data/%d.txt", dimensions[i]);
            read_matrices_from_file(experiment_file_path, &A, &B, &rowsA, &colsA, &rowsB, &colsB);
            C = allocate_matrix(rowsA, colsB);
            
            // without parallelization
            gettimeofday(&ins__tstart, NULL);
            sequential_matrix_multiplication(A, B, C, rowsA, colsA, rowsB, colsB);
            gettimeofday(&ins__tstop, NULL);
            fprintf(result_file, "Sequential: %ld\n", get_time(&ins__tstart, &ins__tstop));

            zeroMatrix(C, rowsA, colsB);

            // with parallelization not optimized
            gettimeofday(&ins__tstart, NULL);
            omp_multiply_matrixes_not_optimized(A, B, C, dimensions[i], dimensions[i], dimensions[i], dimensions[i]);
            gettimeofday(&ins__tstop, NULL);
            fprintf(result_file, "OpenMP no optimized: %ld\n", get_time(&ins__tstart, &ins__tstop));
            // if (compare_matrices(C_sequential, C_parallel_not_optimized, rowsA, colsB) == 0) {
            //     printf("Wrong calculation result (parallel not optimized)");
            // }
            
            zeroMatrix(C, rowsA, colsB);

            // with parallelization optimized
            gettimeofday(&ins__tstart, NULL);
            omp_multiply_matrixes_optimized(A, B, C, dimensions[i], dimensions[i], dimensions[i], dimensions[i]);
            gettimeofday(&ins__tstop, NULL);
            fprintf(result_file, "OpenMP optimized: %ld\n", get_time(&ins__tstart, &ins__tstop));
            // if (compare_matrices(C_sequential, C_parallel_optimized, rowsA, colsB) == 0) {
            //     printf("Wrong calculation result (parallel optimized)");
            // }

            // free memory
            free_matrix(A, rowsA);
            free_matrix(B, rowsB);
            free_matrix(C, rowsA);
        }
    }
    fclose(result_file);
    return 0;
}

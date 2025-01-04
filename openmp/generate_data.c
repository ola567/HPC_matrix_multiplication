#include <stdio.h>
#include <stdlib.h>
#include "utils.h"


int main(int argc, char **argv) {
    int dimensions[10] = {200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000};
    char filename[4];
    int **A, **B;

    for (int i = 0; i < 10; i++) {
        sprintf(filename, "%d", dimensions[i]);
        A = allocate_matrix(dimensions[i], dimensions[i]);
        B = allocate_matrix(dimensions[i], dimensions[i]);
        generate_matrix(A, dimensions[i], dimensions[i]);
        generate_matrix(B, dimensions[i], dimensions[i]);
        save_matrices_to_file(filename, A, dimensions[i], dimensions[i], B, dimensions[i], dimensions[i]);
        free_matrix(A, dimensions[i]);
        free_matrix(B, dimensions[i]);
    }
}
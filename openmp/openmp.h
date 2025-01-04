#ifndef OPENMP_H
#define OPENMP_H

#include <stdio.h>
#include <stdlib.h>

void omp_multiply_matrixes_not_optimized(int** A, int** B, int** C, int rowsA, int colsA, int rowsB, int colsB);
void omp_multiply_matrixes_optimized(int** A, int** B, int** C, int rowsA, int colsA, int rowsB, int colsB);

#endif

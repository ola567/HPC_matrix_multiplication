#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


void read_matrices_from_file(const char* filename, int** A, int** B, int* rowsA, int* colsA, int* rowsB, int* colsB);
void print_matrix(int *matrix, int rows, int cols);
int compareArrays(int *array1, int *array2, int size);
void sequential_matrix_multiplication(int *A, int *B, int *sequentialC, int rowsA, int colsA, int colsB);

#endif

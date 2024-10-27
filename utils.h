#ifndef UTILITY_H
#define UTILITY_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>


void generate_matrix(int** matrix, int rows, int cols);
int** allocate_matrix(int rowsA, int colsB);
void free_matrix(int** matrix, int rows);
void print_matrix(int** matrix, int rows, int cols);
void save_matrices_to_file(const char* filename, int** A, int rowsA, int colsA, int** B, int rowsB, int colsB);
void read_matrices_from_file(const char* filename, int*** A, int*** B, int* rowsA, int* colsA, int* rowsB, int* colsB);
void zeroMatrix(int** matrix, int rows, int cols);
void print_time(struct timeval *start, struct timeval *stop, char* title);

#endif

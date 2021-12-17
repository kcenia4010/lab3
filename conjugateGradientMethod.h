// Copyright 2021 Zaitseva Ksenia
#ifndef MODULES_TASK_1_ZAITSEVA_K_CONJUGATE_GRADIENT_METHOD_H_
#define MODULES_TASK_1_ZAITSEVA_K_CONJUGATE_GRADIENT_METHOD_H_
#include <mpi.h>
#include <vector>
#include <iostream>
#include <random>

void RandomVec(int n, double* A);
double ScalarMult(std::vector<double> x, std::vector<double> y);
double* MultMatrixParallel(double* aa, double* x, int n, int sendcount, MPI_Comm COMM_NEW);
std::vector<double> conjugateGradientMethodSerial(double* A, double* b, int n);
double* conjugateGradientMethodParallel(double* A, double* b, int n);

#endif  // MODULES_TASK_1_ZAITSEVA_K_CONJUGATE_GRADIENT_METHOD_H_

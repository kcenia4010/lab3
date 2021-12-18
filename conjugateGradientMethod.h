// Copyright 2021 Zaitseva Ksenia
#ifndef MODULES_TASK_3_ZAITSEVA_K_CONJUGATE_GRADIENT_METHOD_CONJUGATEGRADIENTMETHOD_H_
#define MODULES_TASK_3_ZAITSEVA_K_CONJUGATE_GRADIENT_METHOD_CONJUGATEGRADIENTMETHOD_H_
#include <mpi.h>

#include <iostream>
#include <vector>
#include <random>

std::vector<double> GenRandNumbers(int n);
double ScalarMult(std::vector<double> x, std::vector<double> y);
std::vector<double> conjugateGradientMethodSerial(double* A, double* b, int n);
std::vector<double> conjugateGradientMethodParallel(double* A, double* b,
                                                    int n);

#endif  // MODULES_TASK_3_ZAITSEVA_K_CONJUGATE_GRADIENT_METHOD_CONJUGATEGRADIENTMETHOD_H_

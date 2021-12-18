// Copyright 2021 Zaitseva Ksenia
#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <gtest-mpi-listener.hpp>

#include "conjugateGradientMethod.h"

TEST(Parallel_MPI, Test_n3) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double A[9] = {4, -1, 2, -1, 6, -2, 2, -2, 5};
  double b[3] = {-1, 9, -10};
  int n = 3;
  std::vector<double> res_seq;
  std::vector<double> res_par = conjugateGradientMethodParallel(A, b, n);
  if (rank == 0) {
    res_seq = conjugateGradientMethodSerial(A, b, n);
    for (int i = 0; i < n; i++)
      EXPECT_NEAR(res_seq[i], res_par[i],
                  std::numeric_limits<double>::epsilon() * 16 * 1000);
  }
}

TEST(Parallel_MPI, Test_n4) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double A[16] = {0, 2, -0.5, 0, 2, 2, -2, 1.5, -0.5, 1, 2, 0, 2, 5, -1, 0};
  double b[4] = {0, -3, 0.5, 0};
  int n = 4;
  std::vector<double> res_par = conjugateGradientMethodParallel(A, b, n);
  if (rank == 0) {
    std::vector<double> res_seq = conjugateGradientMethodSerial(A, b, n);
    for (int i = 0; i < n; i++) {
      EXPECT_NEAR(res_seq[i], res_par[i],
                  std::numeric_limits<double>::epsilon() * 16 * 1000);
    }
  }
}

TEST(Parallel_MPI, Test_n4_2) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double A[16] = {7, 0, 5, 1, -2, 2, 4, -5, -3, 1, 3, 6, 1, -6, 2, -4};
  double b[4] = {1, 3, -1, 7};
  int n = 4;
  std::vector<double> res_par = conjugateGradientMethodParallel(A, b, n);
  if (rank == 0) {
    std::vector<double> res_seq = conjugateGradientMethodSerial(A, b, n);
    for (int i = 0; i < n; i++) {
      EXPECT_NEAR(res_seq[i], res_par[i],
                  std::numeric_limits<double>::epsilon() * 16 * 1000);
    }
  }
}

TEST(Parallel_MPI, Test_n2_2) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double A[4] = {-11, 6, 6, -6};
  double b[2] = {1, 0};
  int n = 2;
  std::vector<double> res_seq;
  std::vector<double> res_par = conjugateGradientMethodParallel(A, b, n);
  if (rank == 0) {
    res_seq = conjugateGradientMethodSerial(A, b, n);
    for (int i = 0; i < n; i++)
      EXPECT_NEAR(res_seq[i], res_par[i],
                  std::numeric_limits<double>::epsilon() * 16 * 1000);
  }
}

TEST(Parallel_MPI, Test_time) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double start, end, stime, ptime;
  std::vector<double> A;
  std::vector<double> b;
  if (rank == 0) {
    A = GenRandNumbers(10000);
    b = GenRandNumbers(100);
  }
  int n = 100;
  std::vector<double> res_seq;
  start = MPI_Wtime();
  std::vector<double> res_par =
      conjugateGradientMethodParallel(A.data(), b.data(), n);
  end = MPI_Wtime();
  if (rank == 0) {
    ptime = end - start;
    start = MPI_Wtime();
    res_seq = conjugateGradientMethodSerial(A.data(), b.data(), n);
    end = MPI_Wtime();
    stime = end - start;
    std::cout << "Sequential time: " << stime << std::endl;

    std::cout << "Speedup: " << stime / ptime << std::endl;
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners =
      ::testing::UnitTest::GetInstance()->listeners();
  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());
  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}

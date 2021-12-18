// Copyright 2021 Zaitseva Ksenia
#include "conjugateGradientMethod.h"

std::vector<double> GenRandNumbers(int n) { 
  std::vector<double> res;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(10, 100);
  int ms = distribution(generator);
  for (int i = 0; i < n; i ++) {
    res.push_back(distribution(generator));
  }
  return res;
}

double ScalarMult(std::vector<double> x, std::vector<double> y) {
  double res = 0;
  int n = x.size();
  for (int i = 0; i < n; i++) res += x[i] * y[i];
  return res;
}
double ScalarMultParallel(double* x, double* y, int n, MPI_Comm COMM_NEW) {
  int size, rank;
  MPI_Comm_size(COMM_NEW, &size);
  MPI_Comm_rank(COMM_NEW, &rank);

  double sum = 0, sum_all = 0;
  for (int i = rank; i < n; i += size) sum += x[i] * y[i];
  MPI_Reduce(&sum, &sum_all, 1, MPI_DOUBLE, MPI_SUM, 0, COMM_NEW);
  return sum_all;
}

std::vector<double> conjugateGradientMethodSerial(double* A, double* b, int n) {
  std::vector<double> x_k_1(n, 0);
  std::vector<double> x_k(n, 0);
  std::vector<double> g_k(n, 0);
  std::vector<double> delta(n);
  
  for (int i = 0; i < n; i++) {
    x_k_1 = x_k;

    for (int j = 0; j < n; j++) {
      g_k[j] = 0;
      for (int k = 0; k < n; k++) g_k[j] += A[j * n + k] * x_k_1[k];
      g_k[j] -= b[j];
    }

    std::vector<double> r1(n);
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) r1[j] += A[j * n + k] * g_k[k];
    }

    if (i > 0) {
      std::vector<double> r2(n);
      for (int j = 0; j < n; j++) {
        for (int k = 0; k < n; k++) r2[j] += A[j * n + k] * delta[k];
      }
      double r3 = ScalarMult(g_k, g_k);
      double r4 = ScalarMult(g_k, delta);
      double r5 = ScalarMult(r1, delta);
      double r6 = ScalarMult(r1, g_k);
      double r7 = ScalarMult(r2, delta);
      double alpha = (r3 * r5 - r4 * r6) / (r7 * r6 - r5 * r5);
      double beta = (r4 * r5 - r3 * r7) / (r7 * r6 - r5 * r5);
      for (int j = 0; j < n; j++) delta[j] = alpha * delta[j] + beta * g_k[j];
    } else {
      double r2 = ScalarMult(g_k, g_k);
      double r3 = ScalarMult(r1, g_k);
      double B_k = -r2 / r3;
      for (int j = 0; j < n; j++) delta[j] = B_k * g_k[j];
    }
    for (int j = 0; j < n; j++) x_k[j] = x_k_1[j] + delta[j];
  }
  return x_k;
}

std::vector<double> conjugateGradientMethodParallel(double* A, double* b,
                                                    int n) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int s = size;
  if (size > n) s = n;
  std::vector<int> ranks(s);
  for (int i = 0; i < s; i++) ranks[i] = i;
  MPI_Group group;
  MPI_Comm_group(MPI_COMM_WORLD, &group);
  MPI_Group new_group;
  MPI_Group_incl(group, s, ranks.data(), &new_group);
  MPI_Comm COMM_NEW;
  MPI_Comm_create(MPI_COMM_WORLD, new_group, &COMM_NEW);
  std::vector<double> x_k(n);
  if (rank == 0) {
    for (int i = 0; i < n; i++) x_k[i] = 0.0;
  }
  if (COMM_NEW != MPI_COMM_NULL) {
    MPI_Comm_size(COMM_NEW, &size);
    MPI_Comm_rank(COMM_NEW, &rank);
    std::vector<int> displs(size);
    std::vector<int> sendcounts(size);
    for (int i = 0; i < size; i++) {
      displs[i] =
          static_cast<int>((static_cast<int>(n * n / size) * i) / n) * n;
    }
    for (int i = 0; i < size - 1; i++) {
      sendcounts[i] = displs[i + 1] - displs[i];
    }
    sendcounts[size - 1] = n * n - displs[size - 1];
    std::vector<double> aa(sendcounts[rank]);

    MPI_Scatterv(A, sendcounts.data(), displs.data(), MPI_DOUBLE, aa.data(),
                 sendcounts[rank], MPI_DOUBLE, 0, COMM_NEW);
    std::vector<double> delta(n);
    std::vector<int> recvcounts(size);
    std::vector<double> x_k_1(n);
    std::vector<double> res(sendcounts[rank] / n);
    std::vector<double> g_k(n);
    std::vector<double> r1(n);
    std::vector<double> r2(n);
    for (int p = 0; p < n;) {
      if (rank == 0) {
        for (int i = 0; i < n; i++) x_k_1[i] = x_k[i];
      }
      MPI_Bcast(x_k_1.data(), n, MPI_DOUBLE, 0, COMM_NEW);
      for (int i = 0; i < (sendcounts[rank] / n); i++) {
        res[i] = 0.0;
        for (int k = 0; k < n; k++) res[i] += aa[i * n + k] * x_k_1[k];
      }
      for (int i = 0; i < size; i++) {
        recvcounts[i] = sendcounts[i] / n;
      }
      displs[size - 1] = n - recvcounts[size - 1];
      for (int i = size-2; i >= 0; i--) {
        displs[i] = displs[i + 1] - recvcounts[i];
      }
      MPI_Gatherv(res.data(), sendcounts[rank] / n, MPI_DOUBLE, g_k.data(),
                  recvcounts.data(), displs.data(), MPI_DOUBLE, 0, COMM_NEW);

      if (rank == 0) {
        for (int j = 0; j < n; j++) g_k[j] -= b[j];
      }

      MPI_Bcast(g_k.data(), n, MPI_DOUBLE, 0, COMM_NEW);
      for (int i = 0; i < (sendcounts[rank] / n); i++) {
        res[i] = 0.0;
        for (int k = 0; k < n; k++) res[i] += aa[i * n + k] * g_k[k];
      }
      for (int i = 0; i < size; i++) {
        recvcounts[i] = sendcounts[i] / n;
      }
      displs[size - 1] = n - recvcounts[size - 1];
      for (int i = size - 2; i >= 0; i--) {
        displs[i] = displs[i + 1] - recvcounts[i];
      }
      MPI_Gatherv(res.data(), sendcounts[rank] / n, MPI_DOUBLE, r1.data(),
                  recvcounts.data(), displs.data(), MPI_DOUBLE, 0, COMM_NEW);

      if (p > 0) {
        MPI_Bcast(delta.data(), n, MPI_DOUBLE, 0, COMM_NEW);
        for (int i = 0; i < (sendcounts[rank] / n); i++) {
          res[i] = 0.0;
          for (int k = 0; k < n; k++) res[i] += aa[i * n + k] * delta[k];
        }
        for (int i = 0; i < size; i++) {
          recvcounts[i] = sendcounts[i] / n;
        }
        displs[size - 1] = n - recvcounts[size - 1];
        for (int i = size - 2; i >= 0; i--) {
          displs[i] = displs[i + 1] - recvcounts[i];
        }
        MPI_Gatherv(res.data(), sendcounts[rank] / n, MPI_DOUBLE, r2.data(),
                    recvcounts.data(), displs.data(), MPI_DOUBLE, 0, COMM_NEW);

        MPI_Bcast(g_k.data(), n, MPI_DOUBLE, 0, COMM_NEW);
        double r3 = ScalarMultParallel(g_k.data(), g_k.data(), n, COMM_NEW);
        double r4 = ScalarMultParallel(g_k.data(), delta.data(), n, COMM_NEW);
        MPI_Bcast(r1.data(), n, MPI_DOUBLE, 0, COMM_NEW);
        double r5 = ScalarMultParallel(r1.data(), delta.data(), n, COMM_NEW);
        double r6 = ScalarMultParallel(r1.data(), g_k.data(), n, COMM_NEW);
        MPI_Bcast(r2.data(), n, MPI_DOUBLE, 0, COMM_NEW);
        double r7 = ScalarMultParallel(r2.data(), delta.data(), n, COMM_NEW);
        if (rank == 0) {
          double alpha = (r3 * r5 - r4 * r6) / (r7 * r6 - r5 * r5);
          double beta = (r4 * r5 - r3 * r7) / (r7 * r6 - r5 * r5);
          for (int j = 0; j < n; j++)
            delta[j] = alpha * delta[j] + beta * g_k[j];
        }
      } else {
        MPI_Bcast(g_k.data(), n, MPI_DOUBLE, 0, COMM_NEW);
        double r8 = ScalarMultParallel(g_k.data(), g_k.data(), n, COMM_NEW);
        MPI_Bcast(r1.data(), n, MPI_DOUBLE, 0, COMM_NEW);
        double r9 = ScalarMultParallel(r1.data(), g_k.data(), n, COMM_NEW);
        if (rank == 0) {
          double B_k = -r8 / r9;
          for (int j = 0; j < n; j++) delta[j] = B_k * g_k[j];
        }
      }
      if (rank == 0) {
        for (int j = 0; j < n; j++) x_k[j] = x_k_1[j] + delta[j];
        p++;
      }
      MPI_Bcast(&p, 1, MPI_INT, 0, COMM_NEW);
    }
    MPI_Comm_free(&COMM_NEW);
  }
  MPI_Bcast(x_k.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  return x_k;
}

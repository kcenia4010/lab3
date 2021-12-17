// Copyright 2021 Zaitseva Ksenia
#include "conjugateGradientMethod.h"

double ScalarMult(std::vector<double> x, std::vector<double> y)
{
	double res = 0;
	for (int i = 0; i < x.size(); i++)
		res += x[i] * y[i];
	return res;
}
double ScalarMultParallel(double* x, double* y, int n , MPI_Comm COMM_NEW) {
	int size, rank;
	int N;
	MPI_Comm_size(COMM_NEW, &size);
	MPI_Comm_rank(COMM_NEW, &rank);
	
	double sum = 0, sum_all = 0;
	for (int i = rank; i < n; i+=size)
		sum += x[i] * y[i];
	MPI_Reduce(&sum, &sum_all, 1, MPI_DOUBLE, MPI_SUM, 0, COMM_NEW);
	return sum_all;
}

std::vector<double> conjugateGradientMethodSerial(double* A, double* b, int n)
{
	std::vector<double> x_k_1(n, 0);
	std::vector<double> x_k(n, 0);
	std::vector<double> g_k(n, 0);
	std::vector<double> delta(n);
	for (int i = 0; i < n; i++)
	{
		x_k_1 = x_k;

		for (int j = 0; j < n; j++)
		{
			g_k[j] = 0;
			for (int k = 0; k < n; k++)
				g_k[j] += A[j * n + k] * x_k_1[k];
			g_k[j] -= b[j];
		}

		std::vector<double> r1(n);
		for (int j = 0; j < n; j++)
		{
			for (int k = 0; k < n; k++)
				r1[j] += A[j * n + k] * g_k[k];
		}

		if (i > 0)
		{
			std::vector<double> r2(n);
			for (int j = 0; j < n; j++)
			{
				for (int k = 0; k < n; k++)
					r2[j] += A[j * n + k] * delta[k];
			}
			double r3 = ScalarMult(g_k, g_k);
			double r4 = ScalarMult(g_k, delta);
			double r5 = ScalarMult(r1, delta);
			double r6 = ScalarMult(r1, g_k);
			double r7 = ScalarMult(r2, delta);
			double alpha = (r3 * r5 - r4 * r6) / (r7 * r6 - r5 * r5);
			double beta = (r4 * r5 - r3 * r7) / (r7 * r6 - r5 * r5);
			for (int j = 0; j < n; j++)
				delta[j] = alpha * delta[j] + beta * g_k[j];
		}
		else
		{
			double r2 = ScalarMult(g_k, g_k);
			double r3 = ScalarMult(r1, g_k);
			double B_k = -r2 / r3;
			for (int j = 0; j < n; j++)
				delta[j] = B_k * g_k[j];
		}
		for (int j = 0; j < n; j++)
			x_k[j] = x_k_1[j] + delta[j];
	}
	return x_k;
}

void RandomVec(int n, double* A)
{
	for (int i = 0; i < n; i++)
		A[i] = rand() % 20;
}

double* conjugateGradientMethodParallel(double* A, double* b, int n)
{
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int s = size;
	if (size > n)
		s = n;
	std::vector<int> ranks(s);
	for (int i = 0; i < s; i++)
		ranks[i] = i;
	MPI_Group group;
	MPI_Comm_group(MPI_COMM_WORLD, &group);
	MPI_Group new_group;
	MPI_Group_incl(group, s, ranks.data(), &new_group);
	MPI_Comm COMM_NEW;
	MPI_Comm_create(MPI_COMM_WORLD, new_group, &COMM_NEW);
	double* x_k = new double[n];
	if (rank == 0)
		for (int i = 0; i < n; i++)
			x_k[i] = 0.0;
	double* g_k = new double[n];
	double* r1 = new double[n];
	double* r2 = new double[n];
	if (COMM_NEW != MPI_COMM_NULL) 
	{
		MPI_Comm_size(COMM_NEW, &size);
		MPI_Comm_rank(COMM_NEW, &rank);
		std::vector<int> displs(size);
		//displs = new int[size];
		std::vector<int> sendcounts (size);
		int tmp = 0;
		for (int i = 0; i < size; i++)
		{
			displs[i] = static_cast<int>((static_cast<int>(n * n / size) * i) / n) * n; // n = 4 , size = 2
		}
		for (int i = 0; i < size - 1; i++)
		{
			sendcounts[i] = displs[i + 1] - displs[i];
		}
		sendcounts[size - 1] = n * n - displs[size - 1];
		double* aa = new double[sendcounts[rank]];

		MPI_Scatterv(A, sendcounts.data(), displs.data(), MPI_DOUBLE, aa, sendcounts[rank], MPI_DOUBLE, 0, COMM_NEW);
		double* delta = new double[n];
		std::vector<int> recvcounts (size);
		double* x_k_1 = new double[n];
		double* res = new double[sendcounts[rank] / n];
		for (int i = 0; i < n;)
		{
			if (rank == 0) {
				for (int i = 0; i < n; i++)
					x_k_1[i] = x_k[i];
				//if (i == 0) x_k_1[0] = 1;
			}
			MPI_Bcast(x_k_1, n, MPI_DOUBLE, 0, COMM_NEW);
			int end;
			if (rank < size - 1) {
				end = static_cast<int>(n / size) * (rank + 1);
			}
			else {
				end = n;
			}
			for (int i = 0; i < end; i++) {
				res[i] = 0.0;
				for (int k = 0; k < n; k++)
					res[i] += aa[i * n + k] * x_k_1[k];
			}
			for (int i = 0; i < size; i++)
			{
				recvcounts[i] = static_cast<int>(n / size) * (i + 1);
				displs[i] = static_cast<int>(n / size) * i;
			}
			recvcounts[size - 1] = n;
			MPI_Gatherv(res, end, MPI_DOUBLE, g_k, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, COMM_NEW);

			if (rank == 0) {
				for (int j = 0; j < n; j++)
					g_k[j] -= b[j];
			}

			MPI_Bcast(g_k, n, MPI_DOUBLE, 0, COMM_NEW);
			if (rank < size - 1) {
				end = static_cast<int>(n / size) * (rank + 1);
			}
			else {
				end = n;
			}
			for (int i = 0; i < end; i++) {
				res[i] = 0.0;
				for (int k = 0; k < n; k++)
					res[i] += aa[i * n + k] * g_k[k];
			}
			for (int i = 0; i < size; i++)
			{
				recvcounts[i] = static_cast<int>(n / size) * (i + 1);
				displs[i] = static_cast<int>(n / size) * i;
			}
			recvcounts[size - 1] = n;
			MPI_Gatherv(res, end, MPI_DOUBLE, r1, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, COMM_NEW);

			if (i > 0)
			{
				MPI_Bcast(delta, n, MPI_DOUBLE, 0, COMM_NEW);
				if (rank < size - 1) {
					end = static_cast<int>(n / size) * (rank + 1);
				} 
				else {
					end = n;
				}
				for (int i = 0; i < end; i++) {
					res[i] = 0.0;
					for (int k = 0; k < n; k++)
						res[i] += aa[i * n + k] * delta[k];
				}
				for (int i = 0; i < size; i++)
				{
					recvcounts[i] = static_cast<int>(n / size) * (i + 1);
					displs[i] = static_cast<int>(n / size) * i;
				}
				recvcounts[size - 1] = n;
				MPI_Gatherv(res, end, MPI_DOUBLE, r2, recvcounts.data(), displs.data(), MPI_DOUBLE, 0, COMM_NEW);

				MPI_Bcast(g_k, n, MPI_DOUBLE, 0, COMM_NEW);
				double r3 = ScalarMultParallel(g_k, g_k, n, COMM_NEW);
				double r4 = ScalarMultParallel(g_k, delta, n, COMM_NEW);
				MPI_Bcast(r1, n, MPI_DOUBLE, 0, COMM_NEW);
				double r5 = ScalarMultParallel(r1, delta, n, COMM_NEW);
				double r6 = ScalarMultParallel(r1, g_k, n, COMM_NEW);
				MPI_Bcast(r2, n, MPI_DOUBLE, 0, COMM_NEW);
				double r7 = ScalarMultParallel(r2, delta, n, COMM_NEW);
				if (rank == 0) {
					double alpha = (r3 * r5 - r4 * r6) / (r7 * r6 - r5 * r5);
					double beta = (r4 * r5 - r3 * r7) / (r7 * r6 - r5 * r5);
					for (int j = 0; j < n; j++)
						delta[j] = alpha * delta[j] + beta * g_k[j];
				}
			}
			else
			{
				MPI_Bcast(g_k, n, MPI_DOUBLE, 0, COMM_NEW);
				double r2 = ScalarMultParallel(g_k, g_k, n, COMM_NEW);
				MPI_Bcast(r1, n, MPI_DOUBLE, 0, COMM_NEW);
				double r3 = ScalarMultParallel(r1, g_k, n, COMM_NEW);
				if (rank == 0)
				{
					double B_k = -r2 / r3;
					for (int j = 0; j < n; j++)
						delta[j] = B_k * g_k[j];
				}
			}
			if (rank == 0)
			{
				for (int j = 0; j < n; j++)
					x_k[j] = x_k_1[j] + delta[j];
				i++;
			}
			MPI_Bcast(&i, 1, MPI_INT, 0, COMM_NEW);
		}
			//delete[] aa;
			//delete[] delta;
			//delete[] x_k_1;
			//delete[] res;
	}
	//delete[] g_k;
	//delete[] r1;
	//delete[] r2;
	if (COMM_NEW != MPI_COMM_NULL)
		MPI_Comm_free(&COMM_NEW);
	MPI_Bcast(x_k, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	return x_k;
}
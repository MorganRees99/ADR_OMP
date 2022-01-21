//C++ code to compute 2D baby skyrmion using gradient flow.


//axial 2d solutions from 1d profile ansatz
#include <stdio.h>
#include <math.h>
#include <iomanip>
#include <sys/time.h>
#include <stdlib.h>
#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <complex>

#define PI (4.0*atan(1.0))
#define NUMFIELDS 3
#define TOL 5e-7
#define MAX_THREADS 10


//wasnt working as need to work out the variation at each step and then minus it from the field, otherwise we write over values we need from the field in the derivatives.
using namespace std;

typedef complex<double> dcomp;

int LATTICEX;
int LATTICEY;

void printTime(double initTime, double finalTime)
{
	printf("calculation took %.3fsec\n", finalTime - initTime);
}

double seconds()
{
	struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
}

int getIndex(int i, int j, int a)
{
	extern int LATTICEX, LATTICEY;
	return (j + i * LATTICEY + a * LATTICEX * LATTICEY);
}

bool saveOutputFile(char* file, double* f, double* en_den, double* bden, double* phase, double hx, double hy, double xmin, double ymin)
{
	extern int LATTICEX, LATTICEY;
	bool success = false;
	FILE* of;
	of = fopen(file, "w");//w write 
	if (!of) {
		printf("failed to open output file %s\n", file);
	}
	else {
		//save data to output file
		for (int i = 0; i < LATTICEX; i++) {
			for (int j = 0; j < LATTICEY; j++) {
				double x = xmin + i * hx;
				double y = ymin + j * hy;
				double f1 = f[getIndex(i, j, 0)];
				double f2 = f[getIndex(i, j, 1)];
				double f3 = f[getIndex(i, j, 2)];
				double b = bden[getIndex(i, j, 0)];
				double e = en_den[getIndex(i, j, 0)];
				double p = phase[getIndex(i, j, 0)];
				double r = sqrt(pow(x, 2) + pow(y, 2));
				fprintf(of, "%g\t%g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\t%.15g\n", x, y, f1, f2, f3, e, b, p, r);
			}
		}
		success = true;
		fclose(of);
	}
	return success;
}


double deriv1(double* f, int i, int j, int a, double hx, double hy, int n1, int p)
{
	extern int LATTICEX, LATTICEY;
	double df0;
	if (p == 1) {
		if (i == 0) {
			df0 = (-f[getIndex(i + 2, j, a)] + 8 * f[getIndex(i + 1, j, a)] - 8 * n1 + n1) / (12 * hx);
		}
		else if (i == 1) {
			df0 = (-f[getIndex(i + 2, j, a)] + 8 * f[getIndex(i + 1, j, a)] - 8 * f[getIndex(i - 1, j, a)] + n1) / (12 * hx);
		}
		else if (i == LATTICEX - 1) {
			df0 = (-n1 + 8 * n1 - 8 * f[getIndex(i - 1, j, a)] + f[getIndex(i - 2, j, a)]) / (12 * hx);
		}
		else if (i == LATTICEX - 2) {
			df0 = (-n1 + 8 * f[getIndex(i + 1, j, a)] - 8 * f[getIndex(i - 1, j, a)] + f[getIndex(i - 2, j, a)]) / (12 * hx);
		}
		else {
			df0 = (-f[getIndex(i + 2, j, a)] + 8 * f[getIndex(i + 1, j, a)] - 8 * f[getIndex(i - 1, j, a)] + f[getIndex(i - 2, j, a)]) / (12 * hx);
		}
	}
	else if (p == 2) {
		if (j == 0) {
			df0 = (-f[getIndex(i, j + 2, a)] + 8 * f[getIndex(i, j + 1, a)] - 8 * n1 + n1) / (12 * hy);
		}
		else if (j == 1) {
			df0 = (-f[getIndex(i, j + 2, a)] + 8 * f[getIndex(i, j + 1, a)] - 8 * f[getIndex(i, j - 1, a)] + n1) / (12 * hy);
		}
		else if (j == LATTICEY - 1) {
			df0 = (-n1 + 8 * n1 - 8 * f[getIndex(i, j - 1, a)] + f[getIndex(i, j - 2, a)]) / (12 * hy);
		}
		else if (j == LATTICEY - 2) {
			df0 = (-n1 + 8 * f[getIndex(i, j + 1, a)] - 8 * f[getIndex(i, j - 1, a)] + f[getIndex(i, j - 2, a)]) / (12 * hy);
		}
		else {
			df0 = (-f[getIndex(i, j + 2, a)] + 8 * f[getIndex(i, j + 1, a)] - 8 * f[getIndex(i, j - 1, a)] + f[getIndex(i, j - 2, a)]) / (12 * hy);
		}
	}

	return df0;
}

double deriv2(double* f, int i, int j, int a, double hx, double hy, int n1, int p)
{
	extern int LATTICEX, LATTICEY;
	double d2f;
	if (p == 1) {
		if (i == 0) {
			d2f = (-f[getIndex(i + 2, j, a)] + 16.0 * f[getIndex(i + 1, j, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * n1 - n1) / (12 * hx * hx);
		}
		else if (i == 1) {
			d2f = (-f[getIndex(i + 2, j, a)] + 16.0 * f[getIndex(i + 1, j, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i - 1, j, a)] - n1) / (12 * hx * hx);
		}
		else if (i == LATTICEX - 2) {
			d2f = (-n1 + 16.0 * f[getIndex(i + 1, j, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i - 1, j, a)] - f[getIndex(i - 2, j, a)]) / (12 * hx * hx);
		}
		else if (i == LATTICEX - 1) {
			d2f = (-n1 + 16.0 * n1 - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i - 1, j, a)] - f[getIndex(i - 2, j, a)]) / (12 * hx * hx);
		}
		else {
			d2f = (-f[getIndex(i + 2, j, a)] + 16.0 * f[getIndex(i + 1, j, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i - 1, j, a)] - f[getIndex(i - 2, j, a)]) / (12 * hx * hx);
		}
	}
	else if (p == 2) {
		if (j == 0) {
			d2f = (-f[getIndex(i, j + 2, a)] + 16.0 * f[getIndex(i, j + 1, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * n1 - n1) / (12 * hy * hy);
		}
		else if (j == 1) {
			d2f = (-f[getIndex(i, j + 2, a)] + 16.0 * f[getIndex(i, j + 1, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i, j - 1, a)] - n1) / (12 * hy * hy);
		}
		else if (j == LATTICEY - 2) {
			d2f = (-n1 + 16.0 * f[getIndex(i, j + 1, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i, j - 1, a)] - f[getIndex(i, j - 2, a)]) / (12 * hy * hy);
		}
		else if (j == LATTICEY - 1) {
			d2f = (-n1 + 16.0 * n1 - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i, j - 1, a)] - f[getIndex(i, j - 2, a)]) / (12 * hy * hy);
		}
		else {
			d2f = (-f[getIndex(i, j + 2, a)] + 16.0 * f[getIndex(i, j + 1, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i, j - 1, a)] - f[getIndex(i, j - 2, a)]) / (12 * hy * hy);
		}
	}
	return d2f;
}

double derivxy(double* f, int i, int j, int a, double hx, double hy, int n1, double dxxf1, double dyyf1)
{
	extern int LATTICEX, LATTICEY;
	//add in j dependance in if statements
	double dxyf;
	if ((i == 0) || (j == 0)) {
		dxyf = (-f[getIndex(i + 2, j + 2, a)] + 16.0 * f[getIndex(i + 1, j + 1, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * n1 - n1) / (24.0 * hx * hy) - 0.5 * (dxxf1 + dyyf1);
	}
	else if ((i == 1) || (j == 1)) {
		dxyf = (-f[getIndex(i + 2, j + 2, a)] + 16.0 * f[getIndex(i + 1, j + 1, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i - 1, j - 1, a)] - n1) / (24.0 * hx * hy) - 0.5 * (dxxf1 + dyyf1);
	}
	else if ((i == LATTICEX - 2) || (j == LATTICEY - 2)) {
		dxyf = (-n1 + 16.0 * f[getIndex(i + 1, j + 1, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i - 1, j - 1, a)] - f[getIndex(i - 2, j - 2, a)]) / (24.0 * hx * hy) - 0.5 * (dxxf1 + dyyf1);
	}
	else if ((i == LATTICEX - 1) || (j == LATTICEY - 1)) {
		dxyf = (-n1 + 16.0 * n1 - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i - 1, j - 1, a)] - f[getIndex(i - 2, j - 2, a)]) / (24.0 * hx * hy) - 0.5 * (dxxf1 + dyyf1);
	}
	else {
		dxyf = (-f[getIndex(i + 2, j + 2, a)] + 16.0 * f[getIndex(i + 1, j + 1, a)] - 30.0 * f[getIndex(i, j, a)] + 16.0 * f[getIndex(i - 1, j - 1, a)] - f[getIndex(i - 2, j - 2, a)]) / (24.0 * hx * hy) - 0.5 * (dxxf1 + dyyf1);
	}
	return dxyf;
}


double getLatticeSize(double size, double h)
{
	int n = (int)(((size / h) + 20 / 2) / 20) * 20;
	return n + 1;
}

void inConf(double* f, double hx, double hy, double xmin, double ymin, int N1, int N2, int method)
{
	extern int LATTICEX, LATTICEY;
	double f_r, r, theta, r0, r1;
		for (int i = 0; i < LATTICEX; i++) {
			for (int j = 0; j < LATTICEY; j++) {
			double x = xmin + i * hx;
			double y = ymin + j * hy;
			if (i == 0 || j == 0 || i == LATTICEX - 1 || j == LATTICEY - 1) {
				f[getIndex(i, j, 0)] = 0;
				f[getIndex(i, j, 1)] = 0;
				f[getIndex(i, j, 2)] = 1;
			}
			else {
				r0 = sqrt(2 * N1);//sqrt(1.5 * N1);for same size
				r1 = sqrt(4 * N2);//sqrt(5 * N2); for same size
				r = sqrt(pow(x, 2) + pow(y, 2));
				theta = atan2(y, x);
				if (r <= r0) {
					f_r = 2 * PI - (PI * r) / r0;
					r = sqrt(pow(x, 2) + pow(y, 2));
					theta = atan2(y, x);
					f[getIndex(i, j, 0)] = sin(f_r) * cos(-N1 * theta);
					f[getIndex(i, j, 1)] = sin(f_r) * sin(-N1 * theta);
					f[getIndex(i, j, 2)] = cos(f_r);
				}
				else if (r > r0 && r < r1) {
					f_r = PI - PI * (r - r0) / (r1 - r0);
					f[getIndex(i, j, 0)] = sin(f_r) * cos(N2 * theta);
					f[getIndex(i, j, 1)] = sin(f_r) * sin(N2 * theta);
					f[getIndex(i, j, 2)] = cos(f_r);
				}
				else {
					f_r = 0;
					f[getIndex(i, j, 0)] = sin(f_r) * cos(N2 * theta);
					f[getIndex(i, j, 1)] = sin(f_r) * sin(N2 * theta);
					f[getIndex(i, j, 2)] = cos(f_r);
				}

			}
		}
	}
}

double en_var(double m, int B, int a, double dxf1, double dxf2, double dxf3, double dxxf1, double dxxf2, double dxxf3, double dyf1, double dyf2, double dyf3, double dyyf1, double dyyf2, double dyyf3, double dxyf1, double dxyf2, double dxyf3, double f03)
{
	double var0;
	if (a == 0)
	{
		var0 = -dxxf1 - dxxf1 * dyf3 * dyf3 - 2.0 * dxf1 * dyf3 * dxyf3 + dxxf3 * dyf1 * dyf3 + dxf3 * dxyf1 * dyf3 + dxf3 * dyf1 * dxyf3 - dxxf1 * dyf2 * dyf2 - 2.0 * dxf1 * dyf2 * dxyf2 + dxyf2 * dxf2 * dyf1 + dyf2 * dxxf2 * dyf1 + dyf2 * dxf2 * dxyf1 - dyyf1 - dyyf1 * dxf3 * dxf3 - 2.0 * dyf1 * dxf3 * dxyf3 + dxyf3 * dxf1 * dyf3 + dxf3 * dxyf1 * dyf3 + dxf3 * dxf1 * dyyf3 - dyyf1 * dxf2 * dxf2 - 2.0 * dyf1 * dxf2 * dxyf2 + dxyf1 * dyf2 * dxf2 + dxf1 * dyyf2 * dxf2 + dxf1 * dyf2 * dxyf2;
		// var0 = -dxxf1 - dyyf1 - ((dxxf1 + dyyf1) * (dxf1 * dxf1 + dxf2 * dxf2 + dxf3 * dxf3 + dyf1 * dyf1 + dyf2 * dyf2 + dyf3 * dyf3) + dxf1 * (dxxf1 * dxf1 + dxxf2 * dxf2 + dxxf3 * dxf3 + dxyf1 * dyf1 + dxyf2 * dyf2 + dxyf3 * dyf3) + dyf1 * (dxyf1 * dxf1 + dxyf2 * dxf2 + dxyf3 * dxf3 + dyyf1 * dyf1 + dyyf2 * dyf2 + dyyf3 * dyf3) - dxxf1 * (dxf1 * dxf1 + dxf2 * dxf2 + dxf3 * dxf3) - 2.0 * dxyf1 * (dxf1 * dyf1 + dxf2 * dyf2 + dxf3 * dyf3) - dyyf1 * (dyf1 * dyf1 + dyf2 * dyf2 + dyf3 * dyf3) - dxf1 * (dxf1 * (dxxf1 + dyyf1) + dxf2 * (dxxf2 + dyyf2) + dxf3 * (dxxf3 + dyyf3)) - dyf1 * (dyf1 * (dxxf1 + dyyf1) + dyf2 * (dxxf2 + dyyf2) + dyf3 * (dxxf3 + dyyf3)));
	}
	else if (a == 1)
	{
		var0 = -dxxf2 - dxxf2 * dyf3 * dyf3 - 2.0 * dxf2 * dyf3 * dxyf3 + dxyf3 * dxf3 * dyf2 + dyf3 * dxxf3 * dyf2 + dyf3 * dxf3 * dxyf2 - dxxf2 * dyf1 * dyf1 - 2.0 * dxf2 * dyf1 * dxyf1 + dxxf1 * dyf2 * dyf1 + dxf1 * dxyf2 * dyf1 + dxf1 * dyf2 * dxyf1 - dyyf2 - dyyf2 * dxf3 * dxf3 - 2.0 * dyf2 * dxf3 * dxyf3 + dxyf2 * dyf3 * dxf3 + dxf2 * dyyf3 * dxf3 + dxf2 * dyf3 * dxyf3 - dyyf2 * dxf1 * dxf1 - 2.0 * dyf2 * dxf1 * dxyf1 + dxyf1 * dxf2 * dyf1 + dxf1 * dxyf2 * dyf1 + dxf1 * dxf2 * dyyf1;
		//var0 = -dxxf2 - dyyf2 - ((dxxf2 + dyyf2) * (dxf1 * dxf1 + dxf2 * dxf2 + dxf3 * dxf3 + dyf1 * dyf1 + dyf2 * dyf2 + dyf3 * dyf3) + dxf2 * (dxxf1 * dxf1 + dxxf2 * dxf2 + dxxf3 * dxf3 + dxyf1 * dyf1 + dxyf2 * dyf2 + dxyf3 * dyf3) + dyf2 * (dxyf1 * dxf1 + dxyf2 * dxf2 + dxyf3 * dxf3 + dyyf1 * dyf1 + dyyf2 * dyf2 + dyyf3 * dyf3) - dxxf2 * (dxf1 * dxf1 + dxf2 * dxf2 + dxf3 * dxf3) - 2.0 * dxyf2 * (dxf1 * dyf1 + dxf2 * dyf2 + dxf3 * dyf3) - dyyf2 * (dyf1 * dyf1 + dyf2 * dyf2 + dyf3 * dyf3) - dxf2 * (dxf1 * (dxxf1 + dyyf1) + dxf2 * (dxxf2 + dyyf2) + dxf3 * (dxxf3 + dyyf3)) - dyf2 * (dyf1 * (dxxf1 + dyyf1) + dyf2 * (dxxf2 + dyyf2) + dyf3 * (dxxf3 + dyyf3)));
	}
	else if (a == 2)
	{
		var0 = (-0.2) * f03 - dxxf3 - dxxf3 * dyf2 * dyf2 - 2.0 * dxf3 * dyf2 * dxyf2 + dxxf2 * dyf3 * dyf2 + dxf2 * dxyf3 * dyf2 + dxf2 * dyf3 * dxyf2 - dxxf3 * dyf1 * dyf1 - 2.0 * dxf3 * dyf1 * dxyf1 + dxyf1 * dxf1 * dyf3 + dyf1 * dxxf1 * dyf3 + dyf1 * dxf1 * dxyf3 - dyyf3 - dyyf3 * dxf2 * dxf2 - 2.0 * dyf3 * dxf2 * dxyf2 + dxyf2 * dxf3 * dyf2 + dxf2 * dxyf3 * dyf2 + dxf2 * dxf3 * dyyf2 - dyyf3 * dxf1 * dxf1 - 2.0 * dyf3 * dxf1 * dxyf1 + dxyf3 * dyf1 * dxf1 + dxf3 * dyyf1 * dxf1 + dxf3 * dyf1 * dxyf1;
  		//var0 = -2 * m * m * f03 - dxxf3 - dyyf3 - ((dxxf3 + dyyf3) * (dxf1 * dxf1 + dxf2 * dxf2 + dxf3 * dxf3 + dyf1 * dyf1 + dyf2 * dyf2 + dyf3 * dyf3) + dxf3 * (dxxf1 * dxf1 + dxxf2 * dxf2 + dxxf3 * dxf3 + dxyf1 * dyf1 + dxyf2 * dyf2 + dxyf3 * dyf3) + dyf3 * (dxyf1 * dxf1 + dxyf2 * dxf2 + dxyf3 * dxf3 + dyyf1 * dyf1 + dyyf2 * dyf2 + dyyf3 * dyf3) - dxxf3 * (dxf1 * dxf1 + dxf2 * dxf2 + dxf3 * dxf3) - 2.0 * dxyf3 * (dxf1 * dyf1 + dxf2 * dyf2 + dxf3 * dyf3) - dyyf3 * (dyf1 * dyf1 + dyf2 * dyf2 + dyf3 * dyf3) - dxf3 * (dxf1 * (dxxf1 + dyyf1) + dxf2 * (dxxf2 + dyyf2) + dxf3 * (dxxf3 + dyyf3)) - dyf3 * (dyf1 * (dxxf1 + dyyf1) + dyf2 * (dxxf2 + dyyf2) + dyf3 * (dxxf3 + dyyf3)));
	}
	return var0;
}

double en_den(double m, double B, double hx, double hy, int p, double f03, double dxf1, double dxf2, double dxf3, double dyf1, double dyf2, double dyf3)
{
	double den1 = 0.5 * (pow(dxf1, 2) + pow(dxf2, 2) + pow(dxf3, 2) + pow(dyf1, 2) + pow(dyf2, 2) + pow(dyf3, 2)) * (hx * hy);
	double den2 = 0.5 * (pow(dxf2 * dyf3, 2) + pow(dxf3 * dyf2, 2) + pow(dxf3 * dyf1, 2) + pow(dxf1 * dyf3, 2) + pow(dxf1 * dyf2, 2) + pow(dxf2 * dyf1, 2) - 2.0 * dxf2 * dyf3 * dxf3 * dyf2 - 2.0 * dxf3 * dyf1 * dxf1 * dyf3 - 2.0 * dxf1 * dyf2 * dxf2 * dyf1) * (hx * hy);
double den3 = pow(m, 2) * (1.0 - pow(f03, 2)) * (hx * hy);
	double energy = 0.0;
	if (p == 1) {
		energy = (1.0 / (4.0 * PI)) * den1;
	}
	else if (p == 2) {
		energy = (1.0 / (4.0 * PI)) * den2;
	}
	else if (p == 3) {
		energy = (1.0 / (4.0 * PI)) * den3;
	}
	else {
		energy = (1.0 / (4.0 * PI)) * (den1 + den2 + den3);
	}
	return energy;
}

double b_den(double m, double hx, double hy, double f1_0, double f2_0, double f3_0, double dxf1, double dxf2, double dxf3, double dyf1, double dyf2, double dyf3)
{
	return -(1.0 / (4.0 * PI) * (f1_0 * dxf2 * dyf3 - f1_0 * dxf3 * dyf2 + f2_0 * dxf3 * dyf1 - f2_0 * dxf1 * dyf3 + f3_0 * dxf1 * dyf2 - f3_0 * dxf2 * dyf1)) * (hx * hy);
}

double calc(double* fin, double* var, double* oldVar, double m, double hx, double hy, double ht, int B, double* etol, bool store_etol, bool store_var)
{
	double delta = 0;
	double t_delta;
	extern int LATTICEX, LATTICEY;
	int loop;
	int steps = (LATTICEX - 1) * (LATTICEY - 1);
	for (loop = 1; loop <= MAX_THREADS; loop++) {
		omp_set_num_threads(loop);
		int id = omp_get_thread_num();
		#pragma omp parallel for 
			for (int loop1 = (loop - 1) * (steps / MAX_THREADS); loop1 < loop * (steps / MAX_THREADS); loop1++) {
				int i = (loop1 / LATTICEY) % LATTICEX;
				int j = loop1 % LATTICEY;

				double f03 = fin[getIndex(i, j, 2)];
				double dxf0 = deriv1(fin, i, j, 0, hx, hy, 0, 1);
				double dyf0 = deriv1(fin, i, j, 0, hx, hy, 0, 2);
				double dxf1 = deriv1(fin, i, j, 1, hx, hy, 0, 1);
				double dyf1 = deriv1(fin, i, j, 1, hx, hy, 0, 2);
				double dxf2 = deriv1(fin, i, j, 2, hx, hy, 1, 1);
				double dyf2 = deriv1(fin, i, j, 2, hx, hy, 1, 2);
				double d2xf0 = deriv2(fin, i, j, 0, hx, hy, 0, 1);
				double d2yf0 = deriv2(fin, i, j, 0, hx, hy, 0, 2);
				double d2xf1 = deriv2(fin, i, j, 1, hx, hy, 0, 1);
				double d2yf1 = deriv2(fin, i, j, 1, hx, hy, 0, 2);
				double d2xf2 = deriv2(fin, i, j, 2, hx, hy, 1, 1);
				double d2yf2 = deriv2(fin, i, j, 2, hx, hy, 1, 2);
				double dxyf0 = derivxy(fin, i, j, 0, hx, hy, 0, d2xf0, d2yf0);
				double dxyf1 = derivxy(fin, i, j, 1, hx, hy, 0, d2xf1, d2yf1);
				double dxyf2 = derivxy(fin, i, j, 2, hx, hy, 1, d2xf2, d2yf2);
				if (store_var) {
					oldVar[getIndex(i, j, 0)] = var[getIndex(i, j, 0)];
					oldVar[getIndex(i, j, 1)] = var[getIndex(i, j, 1)];
					oldVar[getIndex(i, j, 2)] = var[getIndex(i, j, 2)];
				}
				double eom0 = en_var(m, B, 0, dxf0, dxf1, dxf2, d2xf0, d2xf1, d2xf2, dyf0, dyf1, dyf2, d2yf0, d2yf1, d2yf2, dxyf0, dxyf1, dxyf2, f03);
				double eom1 = en_var(m, B, 1, dxf0, dxf1, dxf2, d2xf0, d2xf1, d2xf2, dyf0, dyf1, dyf2, d2yf0, d2yf1, d2yf2, dxyf0, dxyf1, dxyf2, f03);
				double eom2 = en_var(m, B, 2, dxf0, dxf1, dxf2, d2xf0, d2xf1, d2xf2, dyf0, dyf1, dyf2, d2yf0, d2yf1, d2yf2, dxyf0, dxyf1, dxyf2, f03);
				var[getIndex(i, j, 0)] = eom0;
				var[getIndex(i, j, 1)] = eom1;
				var[getIndex(i, j, 2)] = eom2;
			}
	}
	for (int i = 1; i < LATTICEX - 1; i++) {
		for (int j = 1; j < LATTICEY - 1; j++) {
			if (store_var) {
				t_delta = pow(pow(var[getIndex(i, j, 0)] - oldVar[getIndex(i, j, 0)], 10) + pow(var[getIndex(i, j, 1)] - oldVar[getIndex(i, j, 1)], 10) + pow(var[getIndex(i, j, 2)] - oldVar[getIndex(i, j, 2)], 10), 1.0 / 10.0);
				if (t_delta > delta) { delta = t_delta; }
			}
		}
	}
	for (loop = 1; loop <= MAX_THREADS; loop++) {
		omp_set_num_threads(loop);
		int id = omp_get_thread_num();
		#pragma omp parallel for 
			for (int loop1 = (loop - 1) * (steps / MAX_THREADS); loop1 < loop * (steps / MAX_THREADS); loop1++) {
				int i = (loop1 / LATTICEY) % LATTICEX;
				int j = loop1 % LATTICEY;
				
				double f0 = fin[getIndex(i, j, 0)];
				double f1 = fin[getIndex(i, j, 1)];
				double f2 = fin[getIndex(i, j, 2)];

				double newf0 = f0 - ht * var[getIndex(i, j, 0)];
				double newf1 = f1 - ht * var[getIndex(i, j, 1)];
				double newf2 = f2 - ht * var[getIndex(i, j, 2)];
	
				double modf = sqrt(pow(newf0, 2) + pow(newf1, 2) + pow(newf2, 2));
				modf = 1 / modf;
				double varf0 = newf0 * modf;
				double varf1 = newf1 * modf;
				double varf2 = newf2 * modf;
	
				fin[getIndex(i, j, 0)] = varf0;
				fin[getIndex(i, j, 1)] = varf1;
				fin[getIndex(i, j, 2)] = varf2;
			}
	}
	//delta = max(max(delta1,delta2),delta3);
	if (store_var) {
		return delta;
	}
	else return 1;
}

double calcEn(double* fin, int B, double m, double hx, double hy, int p, double* eden)
{
	extern int LATTICEX, LATTICEY;
	double en0 = 0;
	int k, loop;
	int steps = LATTICEX*LATTICEY;
	for (loop = 1; loop <= MAX_THREADS; loop++) {
		omp_set_num_threads(loop);
		en0 = 0;
		#pragma omp parallel
		{
			int loop1, id;
			double sum;
			id = omp_get_thread_num();
			for (loop1 = id, sum = 0.0; loop1 < steps; loop1+=loop) {
				int i = (loop1 / LATTICEY) % LATTICEX;
				int j = loop1 % LATTICEY;
				double f03 = fin[getIndex(i, j, 2)];
				double dxf1 = deriv1(fin, i, j, 0, hx, hy, 0, 1);
				double dyf1 = deriv1(fin, i, j, 0, hx, hy, 0, 2);
				double dxf2 = deriv1(fin, i, j, 1, hx, hy, 0, 1);
				double dyf2 = deriv1(fin, i, j, 1, hx, hy, 0, 2);
				double dxf3 = deriv1(fin, i, j, 2, hx, hy, 1, 1);
				double dyf3 = deriv1(fin, i, j, 2, hx, hy, 1, 2);
				sum += (en_den(m, B, hx, hy, p, f03, dxf1, dxf2, dxf3, dyf1, dyf2, dyf3));
			}
			#pragma omp atomic
				en0 += sum;
		}
	}
	return en0;
}

double calcB(double* fin, double m, double hx, double hy, double* bden)
{
	extern int LATTICEX, LATTICEY;
	double bnumber = 0;
	int k, loop;
	int steps = LATTICEX * LATTICEY;
	for (loop = 1; loop <= MAX_THREADS; loop++) {
		omp_set_num_threads(loop);
		bnumber = 0;
#pragma omp parallel
		{
			int loop1, id;
			double sum;
			id = omp_get_thread_num();
			for (loop1 = id, sum = 0.0; loop1 < steps; loop1 += loop) {
				int i = (loop1 / LATTICEY) % LATTICEX;
				int j = loop1 % LATTICEY;
				double f01 = fin[getIndex(i, j, 0)];
				double f02 = fin[getIndex(i, j, 1)];
				double f03 = fin[getIndex(i, j, 2)];
				double dxf1 = deriv1(fin, i, j, 0, hx, hy, 0, 1);
				double dyf1 = deriv1(fin, i, j, 0, hx, hy, 0, 2);
				double dxf2 = deriv1(fin, i, j, 1, hx, hy, 0, 1);
				double dyf2 = deriv1(fin, i, j, 1, hx, hy, 0, 2);
				double dxf3 = deriv1(fin, i, j, 2, hx, hy, 1, 1);
				double dyf3 = deriv1(fin, i, j, 2, hx, hy, 1, 2);
				sum += b_den(m, hx, hy, f01, f02, f03, dxf1, dxf2, dxf3, dyf1, dyf2, dyf3);
			}
#pragma omp atomic
			bnumber += sum;
		}
	}
	return bnumber;
}

void gradientFlow(double* fin, double* var, double* oldVar, double m, double hx, double hy, double ht, int B, double* etol, double* eden, double* bden)
{
	extern int LATTICEX, LATTICEY;
	//recalculate variational equations
	double delta = 1;
	double delta2 = 1;
	double deltaVar = 1;
	double bnumber = 0;
	double en0 = 0;
	double en1 = 0;
	double en2 = 0;
	double en_s, en_p;
	bool store_var = false;
	int loop = 0;
	en0 = calcEn(fin, B, m, hx, hy, 0, eden);
	bnumber = calcB(fin, m, hx, hy, bden);
	printf("Loop: %d, energy: %.7f, Charge: %.7f, V_tol: %.8f, D_tol: 0.01 \n", loop, en0, bnumber, TOL);
	int k = 0;
	while (delta > TOL)
	{

		if (loop >= 1) { store_var = true; }
		loop++;
		delta = calc(fin, var, oldVar, m, hx, hy, ht, B, etol, false, store_var);
		if (loop % 100 == 0) {
			en0 = calcEn(fin, B, m, hx, hy, 0, eden);
			en_s = calcEn(fin, B, m, hx, hy, 2, eden);
			en_p = calcEn(fin, B, m, hx, hy, 3, eden);
			delta2 = fabs(en_s/en_p);

			bnumber = calcB(fin, m, hx, hy, bden);
			printf("loop: %d, energy: %.7f, charge: %.5f, V_tol: %.8f, D_tol: %.5f \r", loop, en0, bnumber, delta, delta2);
		}
	}
	en_s = calcEn(fin, B, m, hx, hy, 2, eden);
	en_p = calcEn(fin, B, m, hx, hy, 3, eden);
	en0 = calcEn(fin, B, m, hx, hy, 0, eden);
	printf("\nE/B = %0.8f, Skyrme Energy = %0.5f, Potential Energy = %0.5f \n", en0 / B, en_s, en_p);
}

void calcPhase(double* phase, double* f) {
	extern int LATTICEX, LATTICEY;
	for (int i = 0; i < LATTICEX; i++) {
		for (int j = 0; j < LATTICEY; j++) {
			if (fabs(fabs(f[getIndex(i, j, 2)]) - 1) < 0.001) {
				phase[getIndex(i, j, 0)] = -6.28;
			}
			else {
				phase[getIndex(i, j, 0)] = atan2(f[getIndex(i, j, 1)], f[getIndex(i, j, 0)]);
			}
		}
	}
}


int main(int argc, char* argv[])
{
	extern int LATTICEX, LATTICEY;
	if (argc != 5) {
		printf("incorrect number of input arguments, quitting...\n");
		return 1;
	}
	int method = atoi(argv[4]);
	int N1 = atoi(argv[2]);
	int N2 = atoi(argv[3]);
	double scale = 0.8;
	double m = sqrt(atof(argv[1]));
	int B;
	double size, hx, hy, ht;
	char outfileEn[128];
	
	double* etol, * var, * oldVar, * eden, * bden, * phase;
		B = N1 + N2;
		size = 10 + 5 * (N1 + N2);
		hx = 0.2; hy = hx;
		LATTICEX = getLatticeSize(size, hx);
		LATTICEY = getLatticeSize(size, hy);
		size = (LATTICEX - 1) * hx;

		size_t nBytes = NUMFIELDS * LATTICEX * LATTICEY * sizeof(double);
		etol = (double*)malloc(nBytes);
		var = (double*)malloc(nBytes);
		oldVar = (double*)malloc(nBytes);
		eden = (double*)malloc(nBytes);
		bden = (double*)malloc(nBytes);
		phase = (double*)malloc(nBytes);
		double* f = (double*)malloc(nBytes);

		
		ht = 1e-03 / (N1 + N2);
		//ht = 1e-03;
		printf("Grid size: (%g x %g), Lattice: (%d x %d), ht = %.3e, hx = %.3e, hy = %.3e, m = %g, double ring: (%d, %d)\n", size, size, LATTICEX, LATTICEY, ht, hx, hy, m, N1, N2);
		char outfile[128], outfilesc[128];
		sprintf(outfile, "out00/DR4_size%g_lat%d_(%d, %d).dat", size, LATTICEX, N1, N2);
		inConf(f, hx, hy, -size / 2., -size / 2., N1, N2, method);
		double initTime = seconds();
		gradientFlow(f, var, oldVar, m, hx, hy, ht, N1 + N2, etol, eden, bden);
		calcPhase(phase, f);
		double finalTime = seconds();
		printTime(initTime, finalTime);
		saveOutputFile(outfile, f, eden, bden, phase, hx, hy, -size / 2., -size / 2.);
	return 0;
}


//output file at begging to check

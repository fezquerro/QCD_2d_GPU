
__global__ void  suml(double* measure, double* result, int n) {
	int id = (blockDim.x * blockIdx.x + threadIdx.x);
	for (int i = 0; i < n; i++) {
		result[id] += measure[id * n + i];
		measure[id * n + i] = 0;
	}

}
__global__ void  sumlf(double* measure, double* result, int n, int V) {
	int id = (blockDim.x * blockIdx.x + threadIdx.x);
	for (int i = 0; i < n; i++) {
		*result += measure[id * n + i];
		measure[id * n + i] = 0;
	}
	*result /= V;
	//printf("Result sum 1 %.16lf\n", *result);
}


void mean_lattice_par_96(double* measure, double* result, double* res, int V, int y) {//parallel reduction for lattice 96 x 96 x y
	int nthreads = V / y * 2;
	suml << < nthreads / 1024, 1024 >> > (measure, result, y / 2);
	nthreads /= 18;
	suml << <1, nthreads >> > (result, measure, 18);
	nthreads /= 8;
	suml << < 1, nthreads >> > (measure, result, 8);
	nthreads /= 8;
	suml << <1, nthreads >> > (result, measure, 8);
	nthreads /= 16;
	sumlf << < 1, 1 >> > (measure, res, 16, V);

}

void mean_lattice_par_64(double* measure, double* result, double* res, int V, int y) {//parallel reduction for lattice 64 x 64 x y
	int nthreads = V / y * 2;
	suml << < nthreads / 1024, 1024 >> > (measure, result, y / 2);
	nthreads /= 8;
	suml << <1, nthreads >> > (result, measure, 8);
	nthreads /= 8;
	suml << < 1, nthreads >> > (measure, result, 8);
	nthreads /= 8;
	suml << <1, nthreads >> > (result, measure, 8);
	nthreads /= 16;
	sumlf << < 1, 1 >> > (measure, res, 16, V);
	
}


void mean_lattice_par_32(double* measure, double* result, double* res, int V, int y) {//parallel reduction for lattice 32 x 32 x y
	int nthreads = V / y * 2;
	suml << < nthreads / 1024, 1024 >> > (measure, result, y / 2);
	nthreads /= 8;
	suml << <1, nthreads >> > (result, measure, 8);
	nthreads /= 8;
	suml << < 1, nthreads >> > (measure, result, 8);
	nthreads /= 8;
	suml << <1, nthreads >> > (result, measure, 8);
	nthreads /= 4;
	sumlf << < 1, 1 >> > (measure, res, 4, V);

}

void mean_lattice_small(double* measure, double* result,  int V) {
	
		sumlf << < 1, 1 >> > (measure, result, V, V);
}



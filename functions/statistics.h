
double mean_lattice(double* measured,double *measure,int n, struct lattice l, bool print) {
	cudaMemcpy(measure, measured, l.V * sizeof(double), cudaMemcpyDeviceToHost);
	double mean = 0,variance=0;
	for (int i = 0; i < l.V; i++) {
		mean += measure[i];
		//printf("%lf\n", measure[i]);
		variance += measure[i] * measure[i];
	}
		mean = mean / l.V / n;
		variance = sqrt((variance / l.V/n - mean * mean) /(l.V*n - 1));
	if (print) printf("%.10lf +- %.10lf\n",mean,variance);
	cudaMemset(measured, 0, l.V * sizeof(double));
	return  mean;
}

double mean_lattice(int* measured, int* measure, int n, struct lattice l, bool print) {
	cudaMemcpy(measure, measured, l.V * sizeof(int), cudaMemcpyDeviceToHost);
	double mean = 0, variance = 0;
	for (int i = 0; i < l.V; i++) {
		mean += measure[i];
		variance += (double)measure[i] * measure[i];
	}
	mean = mean / l.V / n;
	variance = sqrt((variance / l.V / n - mean * mean) / l.V / (n-1));
	if (print) printf("%.10lf +- %.10lf\n", mean, variance);
	return  mean;
}

void mean_variance(double* measure, int N, double* results, bool print) {//compute mean and variance of an array of measures
	double s = 0, s2 = 0;
	for (int i = 0; i < N; i++) {
		s += measure[i];
		s2 += measure[i] * measure[i];
	}
	results[0] = s / N;
	results[1] = (s2 / N - results[0] * results[0]);
	if (print) printf("%.10lf +- %.10lf\n", results[0], sqrt(results[1]/(N-1)));
}

void autocorrelation(double* measure, int N, int W, double mean,double variance,double* result) {//compute autocorrelation of an array of measures
	double s = 0;
	for (int i=1; i < W+1; i++) {
		for (int j = 0; j < (N - i); j++) 
			s += (measure[j] - mean) * (measure[j + i] - mean);
		
		result[i-1] = s / (N - i) / variance;
		if (i<11) printf("%d %lf\n", i, result[i-1]);
		s = 0;
	}
}

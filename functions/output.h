void save_autocorrelation(double* result, int W, int N, int N_cor, int N_term, struct lattice l, char bound[12], char nsave[2]) {
	char filename[256]; FILE* f;
	snprintf(filename, sizeof(filename), "Autocorrelation_%d_%d_%d_B_%.2lf_%s_%s.txt", l.t, l.x, l.y, l.beta, bound,nsave);
	//snprintf(filename, sizeof(filename), "Autocorrelation_%d_%d_%d_B_%.2lf.txt", l.t, l.x, l.y, l.beta);
	if ((f = fopen(filename, "w")) == NULL) {
		printf("Error creating the file");
		exit(1);
	}
	fprintf(f, "N_termalizacion %d N %d N_correlation %d  D_t %d D_x %d D_ y %d beta %.2lf deltaphi %.2lf\n", N_term, N, N_cor, l.t, l.x, l.y, l.beta, l.epsilon);
	fprintf(f, "N Autocorrelation\n");
	for (int i = 0; i < W; i++)
		fprintf(f, "%d %.10lf\n", i + 1, result[i]);
	fclose(f);

}
void save_energy(double mean, double variance, int N_term, int N, int N_cor, double accept, struct lattice l, char bound[12], char nsave[2]) {
	char filename[256]; FILE* f;

	snprintf(filename, sizeof(filename), "Energy_%d_%d_%d_B_%.2lf_%s_%s.txt", l.t, l.x, l.y, l.beta, bound,nsave);
	//snprintf(filename, sizeof(filename), "Energy_%d_%d_%d_B_%.2lf.txt", l.t, l.x, l.y, l.beta);
	if ((f = fopen(filename, "w")) == NULL) {
		printf("Error creating the file");
		exit(1);
	}
	fprintf(f, "N_termalizacion %d N %d N_correlation %d  D_t %d D_x %d D_ y %d deltaphi %.2lf accept %.2lf\n", N_term, N, N_cor, l.t, l.x, l.y, l.epsilon, accept);
	fprintf(f, "M Energy Error\n");
	fprintf(f, "%f %.10lf %.10lf\n", l.beta, mean, sqrt(variance / (N - 1)));
	fclose(f);

}

void save_energy_av(double mean, double variance, double mean2, double variance2, int N_term, int N, int N_cor, double accept, struct lattice l, char bound[12], char nsave[2]) {
	char filename[256]; FILE* f;

	snprintf(filename, sizeof(filename), "Energy_av_%d_%d_%d_B_%.2lf_%s_%s.txt", l.t, l.x, l.y, l.beta, bound,nsave);
	//snprintf(filename, sizeof(filename), "Energy_%d_%d_%d_B_%.2lf.txt", l.t, l.x, l.y, l.beta);
	if ((f = fopen(filename, "w")) == NULL) {
		printf("Error creating the file");
		exit(1);
	}
	fprintf(f, "N_termalizacion %d N %d N_correlation %d N_updates %d  D_t %d D_x %d D_ y %d deltaphi %.2lf accept %.2lf\n", N_term, N, N_cor, l.updates, l.t, l.x, l.y, l.epsilon, accept);
	fprintf(f, "M Energy/Avplaquette Error\n");
	fprintf(f, "%f %.10lf %.10lf\n", l.beta, mean, sqrt(variance / (N - 1)));
	fprintf(f, "%f %.10lf %.10lf\n", l.beta, mean2, sqrt(variance2 / (N - 1)));
	fclose(f);

}
void save_autocorrelation_av(double* result, int W, int N, int N_cor, int N_term, struct lattice l, char bound[12], char nsave[2]) {
	char filename[256]; FILE* f;
	snprintf(filename, sizeof(filename), "Autocorrelation_av_%d_%d_%d_B_%.2lf_%s_%s.txt", l.t, l.x, l.y, l.beta, bound,nsave);
	//snprintf(filename, sizeof(filename), "Autocorrelation_%d_%d_%d_B_%.2lf.txt", l.t, l.x, l.y, l.beta);
	if ((f = fopen(filename, "w")) == NULL) {
		printf("Error creating the file");
		exit(1);
	}
	fprintf(f, "N_termalizacion %d N %d N_correlation %d  D_t %d D_x %d D_ y %d beta %.2lf deltaphi %.2lf\n", N_term, N, N_cor, l.t, l.x, l.y, l.beta, l.epsilon);
	fprintf(f, "N Autocorrelation\n");
	for (int i = 0; i < W; i++)
		fprintf(f, "%d %.10lf\n", i + 1, result[i]);
	fclose(f);

}


void save_en_seq(double* mean, double* energy, int N_term, int N, int N_cor, double accept, struct lattice l, char bound[12], char nsave[3]) {
	char filename[256]; FILE* f;

	snprintf(filename, sizeof(filename), "Check_%d_%d_%d_B_%.2lf_%s_%s.txt", l.t, l.x, l.y, l.beta, bound, nsave);
	//snprintf(filename, sizeof(filename), "Energy_%d_%d_%d_B_%.2lf.txt", l.t, l.x, l.y, l.beta);
	if ((f = fopen(filename, "w")) == NULL) {
		printf("Error creating the file");
		exit(1);
	}
	fprintf(f, "N_termalizacion %d N %d N_correlation %d N_updates %d  D_t %d D_x %d D_y %d deltaphi %.2lf accept %.2lf\n", N_term, N, N_cor, l.updates, l.t, l.x, l.y, l.epsilon, accept);
	fprintf(f, "I Energy Avpl\n");
	for (int i = 0; i < N; i++)
		fprintf(f, "%d %.10lf %.10lf\n", i, energy[i], mean[i]);
	fclose(f);

}


struct lattice {//a data type to save the lattice parameters
	int y;
	int x;
	int t;
	int V;
	int d;
	int dg;
	int dt;
	float beta;
	float epsilon;
	int updates;

};
__global__ void setup_kernel(curandStateMRG32k3a_t* state, int seed)//initialize GPU random generator, each sequence has the same seed but an offset so they are not correlated
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	/* Each thread gets same seed, a different sequence
	   number, no offset */
	curand_init(seed, id, 0, &state[id]);
}


__global__ void rand_init2(double* phi, curandStateMRG32k3a_t* state, struct lattice l) {//initialize the lattice with random links using half the RG of GPU
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	su2_rand(phi + i * l.dt, state+i);//first sublattice
	su2_rand(phi + i * l.dt + l.dg, state+i);
	su2_rand(phi + i * l.dt + 2 * l.dg, state+i);
	su2_rand(phi + (i + l.V) * l.dt, state+i); //second sublattice
	su2_rand(phi + (i + l.V) * l.dt + l.dg, state+i);
	su2_rand(phi + (i + l.V) * l.dt + 2 * l.dg, state+i);
	//phi[l] = 1;
	//printf(" %d %lf %d %lf\n",l,phi[l], l + l.x * l.y * l.z * l.t,phi[l+ l.x * l.y * l.z * l.t]);


}
__global__ void id_init2(double* phi, struct lattice l) {//initialize identity all links
	int i = threadIdx.x + blockIdx.x * blockDim.x;

	su2_id(phi + i * l.dt);//first sublattice
	su2_id(phi + i * l.dt + l.dg);
	su2_id(phi + i * l.dt + 2 * l.dg);
	su2_id(phi + (i + l.V) * l.dt);//second sublattice
	su2_id(phi + (i + l.V) * l.dt + l.dg);
	su2_id(phi + (i + l.V) * l.dt + 2 * l.dg);
	//phi[l] = 1;
	//printf(" %d %lf %d %lf\n",l,phi[l], l + l.x * l.y * l.z * l.t,phi[l+ l.x * l.y * l.z * l.t]);


}
__global__ void rand_init2_pcc(double* phi, curandStateMRG32k3a_t* state, struct lattice l) {//initialize with dirichlet, 
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int pos_y = id % l.y;
	int pos_x = (id % (l.x * l.y)) /  l.y;
	int pos_t = id / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	su2_rand(phi + id * l.d * l.dg, state+id);//random links for y direction
		if (pos_y == 0 && par_aux == 0) {// if y=0 x and y links are indentity, else random
			su2_id(phi + id * l.d * l.dg + l.dg);
			su2_id(phi + id * l.d * l.dg + 2 * l.dg);
		}
		else {
			su2_rand(phi + id * l.d * l.dg + l.dg, state+id);
			su2_rand(phi + id * l.d * l.dg + 2 * l.dg, state+id);
		}
		su2_rand(phi + (id+l.V) * l.d * l.dg, state+id);
		if (pos_y == 0 && par_aux == 1) {//same for second sublattice
			su2_id(phi + (id+l.V) * l.d * l.dg + l.dg);
			su2_id(phi + (id+l.V) * l.d * l.dg + 2 * l.dg);
		}
	    else
		{
			su2_rand(phi + (id+l.V) * l.d * l.dg + l.dg, state+id);
			su2_rand(phi + (id+l.V) * l.d * l.dg + 2 * l.dg, state+id);
		}
	//phi[l] = 1;
	//printf(" %d %lf %d %lf\n",l,phi[l], l + l.x * l.y * l.z * l.t,phi[l+ l.x * l.y * l.z * l.t]);


}

void read_conf(double* phi, struct lattice l, int* Nterm, int* initt, char bound[12], char nsave[2])
{
	char filename[256]; FILE* f;
	snprintf(filename, sizeof(filename), "Config_%d_%d_%d_B_%.2lf_%s_%s.txt", l.t, l.x, l.y, l.beta, bound, nsave);
	if ((f = fopen(filename, "rb")) == NULL) {
		printf("Error creating the file");
		exit(1);
	}
	fread(phi, sizeof(phi[0]), l.V * l.dt, f);
	fclose(f);
	snprintf(filename, sizeof(filename), "Config_%d_%d_%d_B_%.2lf_%s_%s_data.txt", l.t, l.x, l.y, l.beta, bound, nsave);
	if ((f = fopen(filename, "r")) == NULL) {
		printf("Error creating the file");
		exit(1);
	}
	fscanf(f, "N_termalization %d Type_termalization %d", Nterm, initt);
	fclose(f);
}

void printlattice_matrix(double* phi, struct lattice l)//prints the matrices of the lattice
{
	for (int t = 0; t < l.t; t++)
	{
		for (int x = 0; x < l.x; x++)
		{
			for (int y = 0; y < l.y; y++) {
				su2_print(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x));
				printf("\n");
				su2_print(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x) + l.dg);
				printf("\n");
				su2_print(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x) + 2 * l.dg);
				printf("\n");
			}
			printf("\n\n");

		}
		printf("\n\n\n\n");
	}

}
void printlattice(double* phi, struct lattice l)//prints the 4 real values of each U of the lattice
{
	for (int t = 0; t < l.t; t++)
	{
		for (int x = 0; x < l.x; x++)
		{
			for (int y = 0; y < l.y; y++) {
				if ((x + t) % 2 == 0) {
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x));
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x) + l.dg);
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x) + 2 * l.dg);
					printf("\n");
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x + l.V));
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x + l.V) + l.dg);
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x + l.V) + 2 * l.dg);
					printf("\n");
				}
				else {
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x + l.V));
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x + l.V) + l.dg);
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x + l.V) + 2 * l.dg);
					printf("\n");
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x));
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x) + l.dg);
					su2_print_vec(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x) + 2 * l.dg);
					printf("\n");
				}
			}
			printf("\n\n");

		}
		printf("\n\n\n\n");
	}

}
void check_unit(double* phi, struct lattice l) {//prints the value of the determinant of each U
	for (int t = 0; t < l.t; t++)
	{
		for (int x = 0; x < l.x; x++)
		{
			for (int y = 0; y < l.y; y++) {
				printf("%.16lf %.16lf %.16lf", su2_det(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x)), su2_det(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x) + l.dg), su2_det(phi + l.d * l.dg * (y + x * l.y + t * l.y * l.x) + 2 * l.dg));
				printf("\n");
			}
			printf("\n\n");

		}
		printf("\n\n\n\n");
	}


}

void print_det(double* phi, struct lattice l) {//prints the average value of the determinant of the links in the lattice
	int i; double det = 0;
	for (i = 0; i < l.V * l.d; i++) {
		//printf("%.16lf\n",su2_det(phi+i*l.dg));
		det += su2_det(phi + i * l.dg);
	}
	printf("Av det %.16lf\n", det / l.V / l.d);
}


__global__ void sumd(double* phi, struct lattice l)//sum over a lattice
{
	double sum = 0;
	for (int t = 0; t < l.t; t++)
	{
		for (int x = 0; x < l.x; x++)
		{
			for (int y = 0; y < l.y; y++)
					sum += phi[ y + l.y * x + l.x * l.y  * t];


		}

	}
	printf("Result sum 2 %.16lf\n", sum / l.V);
}
__global__ void normalize(double* phi, struct lattice l) {//normalize thelinks of the lattice
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	su2_normalize(phi + id * l.dt);
	su2_normalize(phi + id * l.dt + l.dg);
	su2_normalize(phi + id * l.dt+  2 * l.dg);
}

__global__ void sum_q(double* phi, int n,int V, int jump, int start)//sum values of an array
{
	double sum = 0;
	for (int i = 0; i < n; i++)
		sum += phi[i*jump+start];
	printf("Result sum %d %d %.16lf\n",n,jump, sum/V);

}


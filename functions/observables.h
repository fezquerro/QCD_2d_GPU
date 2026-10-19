__device__ double plaquette_yx(double* phi, int tp, int xp, int yp, int dg,int sub) {
	double A[4], B[4];
	su2_product(phi, phi + yp + dg+sub, A);
	su2_product_tc(A, phi + xp+sub, B);
	su2_product_tc(B, phi + dg, A);
	return 1 - A[0];
}
__device__ double plaquette_yt(double* phi,  int tp, int xp, int yp, int dg, int sub) {
	double A[4], B[4];
	su2_product(phi, phi + yp + 2 * dg+sub, A);
	su2_product_tc(A, phi + tp+sub, B);
	su2_product_tc(B, phi + 2 * dg, A);
	return 1 - A[0];
}
__device__ double plaquette_xt(double* phi,  int tp, int xp, int yp, int dg, int sub) {
	double A[4], B[4];
	su2_product(phi + dg, phi + xp + 2 * dg+sub, A);
	su2_product_tc(A, phi + tp + dg+sub, B);
	su2_product_tc(B, phi + 2 * dg, A);
	return 1 - A[0];
}

__device__ double plaquette_yx_anti(double* phi, int tp, int xp, int yp, int dg, int sub) {
	double A[4], B[4];
	su2_product_tc(phi, phi + yp + dg + sub, A);
	su2_product_tc(A, phi + xp + sub, B);
	su2_product_tc(B, phi + dg, A);
	return 1 - A[0];
}
__device__ double plaquette_yt_anti(double* phi, int tp, int xp, int yp, int dg, int sub) {
	double A[4], B[4];
	su2_product_tc(phi, phi + yp + 2 * dg + sub, A);
	su2_product_tc(A, phi + tp + sub, B);
	su2_product_tc(B, phi + 2 * dg, A);
	return 1 - A[0];
}

__global__ void plaquette_av(double* phi, double* result, struct lattice l) { //compute the average of plaquette
	int posf = (blockDim.x * blockIdx.x + threadIdx.x);// define the position in the lattice
	int parity= posf / l.V; //#the first sublattice is "even" and the second "odd" sites
	int pos = posf % l.V;//position in the sublattice
	int pos_y = pos %  l.y;
	int pos_x = (pos % (l.x * l.y )) /  l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x  + parity) % 2;
	int  yp, xp, tp,aux=(1-2*parity)*l.V*l.dt,posfr=posf*l.dt;

	yp = par_aux*l.dt; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = (- l.y + 1)*l.dt;
	xp =  l.y*l.dt;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1)*l.dt;
	tp =  l.y * l.x*l.dt;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = - l.y * l.x * (l.t - 1)*l.dt;
	double s = 0;


	s+= plaquette_yx(phi + posfr, tp, xp , yp , l.dg,aux);
	s+= plaquette_yt(phi + posfr, tp , xp , yp , l.dg,aux);
	s+= plaquette_xt(phi + posfr, tp , xp , yp , l.dg,aux);
	result[posf] += s / l.d;
	//printf("%lf\n", result[posf]);
}
__global__ void energy(double* phi, double* result, struct lattice l) { //compute the average of the energy
	int posf = (blockDim.x * blockIdx.x + threadIdx.x);// define the position in the lattice
	int parity = posf / l.V; //#the first sublattice is "even" and the second "odd" sites
	int pos = posf % l.V;//position in the sublattice
	int pos_y = pos % l.y;
	int pos_x = (pos % (l.x * l.y)) / l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x + parity) % 2;
	int  yp, xp, tp, aux = (1 - 2 * parity) * l.V * l.dt, posfr = posf * l.dt;

	yp = par_aux * l.dt; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = (-l.y + 1) * l.dt;
	xp = l.y * l.dt;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1) * l.dt;
	tp = l.y * l.x * l.dt;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1) * l.dt;
	double s = 0;


	s += plaquette_yx(phi + posfr, tp, xp, yp, l.dg, aux);
	s += -plaquette_yt(phi + posfr, tp, xp, yp, l.dg, aux);
	s += -plaquette_xt(phi + posfr, tp, xp, yp, l.dg, aux);
	result[posf] += s;
}
//compute the energy and the average of plaquettes of one configuration and add them to res and res2
//each thread computes the plaquettes of its site, the block sums them in shared memory (any block size <= 256)
//and thread 0 adds the block sums divided by the volume with atomicAdd (res and res2 must start at 0)
//PMC = 1 for perfect magnetic conductor boundary conditions: PMC imposes F_02 = F_12 = 0 at the boundary, so the y-x and y-t
//plaquettes of y = Ly-1 (outward from the boundary) are fixed to the identity: they are counted with 1/2 tr P = 1
//in the average plaquette and do not enter the energy (only P_xt does)
template <int PMC>
__global__ void en_av(double* phi, double* res, double* res2, struct lattice l) {
	__shared__ double sh[2][256];
	int posf = (blockDim.x * blockIdx.x + threadIdx.x);// define the position in the lattice
	int parity = posf / l.V; //#the first sublattice is "even" and the second "odd" sites
	int pos = posf % l.V;//position in the sublattice
	int pos_y = pos % l.y;
	int pos_x = (pos % (l.x * l.y)) / l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x + parity) % 2;
	int  yp, xp, tp, aux = (1 - 2 * parity) * l.V * l.dt, posfr = posf * l.dt;

	yp = par_aux * l.dt; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = (-l.y + 1) * l.dt;
	xp = l.y * l.dt;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1) * l.dt;
	tp = l.y * l.x * l.dt;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1) * l.dt;

	double a = plaquette_yx(phi + posfr, tp, xp, yp, l.dg, aux);
	double b = plaquette_yt(phi + posfr, tp, xp, yp, l.dg, aux);
	double c = plaquette_xt(phi + posfr, tp, xp, yp, l.dg, aux);
	if (PMC && pos_y == l.y - 1 && par_aux == 1) {//outer plaquettes fixed to the identity (1 - 1/2 tr P = 0)
		a = 0; b = 0;
	}
	sh[0][threadIdx.x] = -c; //energy
	sh[1][threadIdx.x] = (3 - a - b - c) / l.d; //average plaquette
	__syncthreads();
	for (int s = 128; s > 0; s >>= 1) {//tree sum in shared memory, works for any block size <= 256
		if (threadIdx.x < s && threadIdx.x + s < blockDim.x) {
			sh[0][threadIdx.x] += sh[0][threadIdx.x + s];
			sh[1][threadIdx.x] += sh[1][threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		atomicAdd(res, sh[0][0] / (2 * l.V)); //l is the half lattice, 2 * l.V sites
		atomicAdd(res2, sh[1][0] / (2 * l.V));
	}
}

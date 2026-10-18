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
__global__ void en_av(double* phi, double* result,double* result2, struct lattice l) {// compute the average of the energy and plaquette
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
	double a,b,c;


	a= plaquette_yx(phi + posfr, tp, xp, yp, l.dg, aux);
	b= plaquette_yt(phi + posfr, tp, xp, yp, l.dg, aux);
	c= plaquette_xt(phi + posfr, tp, xp, yp, l.dg, aux);
	//printf("%lf %lf %lf\n", a, b, c);
	//result[posf] += a - b - c;
	result[posf] += -c;
	result2[posf] += (3-a-b-c) / l.d;

	//printf("%lf\n", result[posf]);
}

__global__ void en_av_neu(double* phi, double* result, double* result2, struct lattice l) {
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
	double a, b, c;


	a = plaquette_yx(phi + posfr, tp, xp, yp, l.dg, aux);
	b = plaquette_yt(phi + posfr, tp, xp, yp, l.dg, aux);
	c = plaquette_xt(phi + posfr, tp, xp, yp, l.dg, aux);
	if (pos_y == l.y - 1 && par_aux == 1) {
		a = 0; b = 0;
	}
	//printf("%lf %lf %lf\n", a, b, c);
	//result[posf] += a - b - c;
	result[posf] += -c;
	result2[posf] += (3 - a - b - c) / l.d;

	//printf("%lf\n", result[posf]);
}



__device__ int metropolis(double* A, double* phi, double beta, double epsilon, curandStateMRG32k3a_t* state) {//metropolis algorithm for SU2

	double Unew[4], deltaU[4];
	su2_update(phi, Unew, epsilon,state);// generate the candidate SU2 matrix
	//su2_print_vec(phi+pos);su2_print_vec(Unew);    printf("\n");

	su2_sub(Unew, phi, deltaU);// compute the difference between the link and the candidate

	double s = -beta * su2_prod_trace(deltaU, A);// compute the change of action fot the candidate
	// printf("%lf\n",s);
	if (s < 0) {//if the delta of the action is negative it accepts the change
		su2_copy(Unew, phi);
		return 1;
	}
	if (expf(-s) > curand_uniform_double(state)) {//metropolis algorithm for accepting the change
		su2_copy(Unew, phi);
		return 1;
	}
	else
		return 0;
}
#define pi 3.14159265358979324

__device__ int heat_bath(double* A, double* phi, double beta, curandStateMRG32k3a_t* state) {//heatbath algorithm for SU2
	double a = su2_normalize_det(A),//norm of the sum of the neighbours product links and normalize A
		r[3], l=0, aux=2, n, X[4], psi, ctheta;
	int c = 0, i;


	while (aux * aux > 1 - l){//generate X[0] with the correct probability
		for (i = 0; i < 3; i++)
			r[i] = curand_uniform_double(state);
		aux = cos(2 * pi * r[1]);
		l = -(log(r[0]) + aux * aux * log(r[2])) / (2 * a * beta);
		aux = 1 - curand_uniform_double(state);
	}
	X[0] = 1 - 2 * l; n = sqrt(1 - X[0] * X[0]);

	// compute random points in the sphere of radious n for the rest of points X[i]
	/*
	c = 0; // with random points on a cube
	while (c == 0) {
		for (i = 0; i < 3; i++)
			r[i] = 2 * curand_uniform_double(state) - 1;
		aux = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
		if (aux <= 1)
			c = 1;
    }
	for (i = 0; i < 3; i++) {
		X[i + 1] = r[i] * n / sqrt(aux);
	}
	*/
	//generating 2 random angles
	psi = 2 * pi * curand_uniform_double(state); //generate psi from 0 to 2pi
	ctheta = 2 * curand_uniform_double(state) - 1; //generate cos (theta) from -1 to 1
	aux = n * sqrt(1 - ctheta * ctheta);//n*arcsin(theta)
	X[1] = n * ctheta;
	X[2] = aux * cos(psi);
	X[3] = aux * sin(psi);

	su2_product_tc(X, A, phi); // obtain the new link

	return 1;
}
__device__ int over_relaxation(double* A, double* phi) {//heatbath algorithm for SU2
	double a = su2_normalize_det(A),//norm of the sum of the neighbours product links and normalize A
		B[4];
	su2_product_tc2(phi, A,B);
	su2_product_tc1(A, B, phi);
	return 1;
}
__global__ void MC_update_even_y(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l,int update) {//update one site
	int id = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	int pos_y = id % l.y;
	int pos_x = (id % (l.x * l.y )) / (l.y);
	int pos_t = id / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int pos = (id + l.V)*l.dt;
	int idr = id * l.dt;
	int yp, ym, xp, xm, tp, tm;

	yp = par_aux*l.dt; //the positive derivative in y
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = ( - l.y + 1)*l.dt;
	xp = l.y*l.dt;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = - l.y * (l.x - 1)*l.dt;
	xm = -l.y*l.dt;//the negative derivative in x
	if (pos_x == 0)
		xm =  l.y * (l.x - 1)*l.dt;
	tp =  l.y * l.x*l.dt;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = - l.y * l.x * (l.t - 1)*l.dt;
	tm = - l.y * l.x*l.dt;//the negative derivative in t
	if (pos_t == 0)
		tm =  l.y * l.x * (l.t - 1)*l.dt;
	
	double aux[4], aux2[4],A[4];
	su2_zero(A);
	
	su2_product_tc(phi + pos + yp+ l.dg, phi + pos + xp , aux);//x positive plaquette
	su2_product_tc(aux, phi + idr +l. dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc2(phi + idr + yp  + l.dg + xm , phi + pos + xm , aux);//x negative plaquette
	su2_product(aux, phi + pos + xm + l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);
	
	su2_product_tc(phi + pos + yp  + 2 * l.dg, phi + pos + tp , aux);//t positive plaquette
	su2_product_tc(aux, phi + idr + 2 * l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc2(phi + idr + yp  + 2 * l.dg + tm , phi + pos + tm , aux);//t negative plaquette
	su2_product(aux, phi + pos + tm + 2 * l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);
	
	//for (int i = 0; i < l.updates; i++)
	//accept[id] += metropolis(A, phi + idr, l.beta, l.epsilon, state + id);
	if (update==0)
	 accept[id] += heat_bath(A, phi + idr, l.beta, state + id);
	else
	accept[id] += over_relaxation(A, phi + idr);
}

__global__ void MC_update_odd_y(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l,int update) {//update one site

	int pos = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	int pos_y = pos % l.y;
	int pos_x = (pos % (l.x * l.y)) /  l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int id = (pos + l.V)*l.dt;
	int posr = pos * l.dt;
	int  yp, ym, xp, xm, tp, tm;

	yp = (1 - par_aux)*l.dt; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 0)
		yp = ( - l.y + 1)*l.dt;
	xp = l.y*l.dt;////the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = - l.y * (l.x - 1)*l.dt;
	xm = -l.y*l.dt;///the negative derivative in x
	if (pos_x == 0)
		xm =  l.y * (l.x - 1)*l.dt;
	
	tp =  l.y * l.x*l.dt;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = - l.y * l.x * (l.t - 1)*l.dt;
	tm = - l.y * l.x*l.dt;//the negative derivative in t
	if (pos_t == 0)
		tm =  l.y * l.x * (l.t - 1)*l.dt;
	
	double aux[4], aux2[4], A[4];
	su2_zero(A);
	su2_product_tc(phi + posr + yp + l.dg, phi + posr + xp, aux);//x positive plaquette
	su2_product_tc(aux, phi + id + l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc2(phi + id + yp  + l.dg + xm , phi + posr + xm , aux);//x negative plaquette
	su2_product(aux, phi + posr + xm  + l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc(phi + posr + yp  + 2 * l.dg, phi + posr + tp, aux);//t positive plaquette
	su2_product_tc(aux, phi + id + 2 * l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc2(phi + id + yp  + 2 * l.dg + tm , phi + posr + tm , aux);//t negative plaquette
	su2_product(aux, phi + posr + tm  + 2 * l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	//for (int i = 0; i < l.updates; i++)
	//accept[id/l.dt] += metropolis(A, phi + id, l.beta, l.epsilon, state + pos);
	if (update==0)
	accept[id/l.dt] += heat_bath(A, phi + id, l.beta, state + pos);
	else
     accept[id / l.dt] += over_relaxation(A, phi + id);
}
__global__ void MC_update_even_x(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l,int update) {//update one site
	int id = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	int pos_y = id % l.y;
	int pos_x = (id % (l.x * l.y)) / (l.y);
	int pos_t = id / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int pos = (id + l.V) * l.dt;
	int idr = id * l.dt;
	int yp, ym, xp, xm, tp, tm;
	
	yp = par_aux * l.dt; //the positive derivative in y
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = (-l.y + 1) * l.dt;
	ym = (par_aux - 1) * l.dt; //the negative derivative in y
	if (pos_y == 0 && par_aux == 0)
		ym = (l.y - 1) * l.dt;
	xp = l.y * l.dt;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1) * l.dt;
	tp = l.y * l.x*l.dt;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1) * l.dt;
	tm = -l.y * l.x*l.dt;//the negative derivative in t
	if (pos_t == 0)
		tm = l.y * l.x * (l.t - 1) * l.dt;

	double aux[4], aux2[4], A[4];
	su2_zero(A);
	su2_product_tc(phi + pos + xp, phi + pos + yp  + l.dg, aux);//y positive plaquette
	su2_product_tc(aux, phi + idr, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);
	
	su2_product_tc2(phi + idr + xp  + ym , phi + pos + ym  + l.dg, aux);//y negative plaquette
	su2_product(aux, phi + pos + ym , aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc(phi + pos + xp  + 2 * l.dg, phi + pos + tp + l.dg, aux);//t positive plaquette
	su2_product_tc(aux, phi + idr + 2 * l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc2(phi + idr + xp  + 2 * l.dg + tm , phi + pos + tm  + l.dg, aux);//t negative plaquette
	su2_product(aux, phi + pos + tm  + 2 * l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);
	
	//for (int i = 0; i < l.updates; i++)
	//accept[id] += metropolis(A, phi + idr+l.dg, l.beta, l.epsilon, state + id);
	if (update == 0)
	accept[id] += heat_bath(A, phi + idr+l.dg, l.beta, state + id);
	
		
	else
	accept[id] += over_relaxation(A, phi + idr + l.dg);
}

__global__ void MC_update_odd_x(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l,int update) {//update one site

	int pos = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	int pos_y = pos % l.y;
	int pos_x = (pos % (l.x * l.y)) / l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int id = (pos + l.V) * l.dt;
	int posr = pos * l.dt;
	int  yp, ym, xp, xm, tp, tm;
	
	yp = (1 - par_aux) * l.dt; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 0)
		yp = (-l.y + 1) * l.dt;
	ym = -par_aux*l.dt; //the negative derivative in z
	if (pos_y == 0 && par_aux == 1)
		ym = (l.y - 1) * l.dt;

	xp = l.y * l.dt;////the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1) * l.dt;
	tp = l.y * l.x * l.dt;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1) * l.dt;
	tm = -l.y * l.x*l.dt;//the negative derivative in t
	if (pos_t == 0)
		tm = l.y * l.x * (l.t - 1) * l.dt;

	double aux[4], aux2[4], A[4];
	su2_zero(A);
	su2_product_tc(phi + posr + xp , phi + posr + yp  + l.dg, aux);//y positive plaquette
	su2_product_tc(aux, phi + id, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc2(phi + id + xp  + ym , phi + posr + ym  + l.dg, aux);//y negative plaquette
	su2_product(aux, phi + posr + ym , aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc(phi + posr + xp  + 2 * l.dg, phi + posr + tp  + l.dg, aux);//t positive plaquette
	su2_product_tc(aux, phi + id + 2 * l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc2(phi + id + xp  + 2 * l.dg + tm , phi + posr + tm  + l.dg, aux);//t negative plaquette
	su2_product(aux, phi + posr + tm  + 2 * l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);
	
	//for (int i = 0; i < l.updates; i++)
	//accept[id/l.dt] += metropolis(A, phi + id+l.dg, l.beta, l.epsilon, state + pos);
	if (update == 0)
		accept[id / l.dt] += heat_bath(A, phi + id + l.dg, l.beta, state + pos);
	else
		accept[id / l.dt] += over_relaxation(A, phi + id+l.dg);
	

}
__global__ void MC_update_even_t(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l,int update) {//update one site
	int id = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	int pos_y = id % l.y;
	int pos_x = (id % (l.x * l.y)) / (l.y);
	int pos_t = id / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int pos = (id + l.V) * l.dt;
	int idr = id * l.dt;
	int yp, ym, xp, xm, tp, tm;
	
	yp = par_aux * l.dt; //the positive derivative in y
	if (pos_y == (l.y - 1) && par_aux == 1)
		yp = (-l.y + 1) * l.dt;
	ym = (par_aux - 1) * l.dt; //the negative derivative in y
	if (pos_y == 0 && par_aux == 0)
		ym = (l.y - 1) * l.dt;
	xp = l.y * l.dt;//the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1) * l.dt;
	xm = -l.y * l.dt;//the negative derivative in x
	if (pos_x == 0)
		xm = l.y * (l.x - 1) * l.dt;
	tp = l.y * l.x*l.dt;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1) * l.dt;
	double aux[4], aux2[4], A[4];
	su2_zero(A);
	su2_product_tc(phi + pos + tp, phi + pos + yp  + 2 * l.dg, aux);//y positive plaquette
	su2_product_tc(aux, phi + idr, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc2(phi + idr + tp  + ym , phi + pos + ym  + 2 * l.dg, aux);//y negative plaquette
	su2_product(aux, phi + pos + ym , aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc(phi + pos + tp  + l.dg, phi + pos + xp + 2 * l.dg, aux);//t positive plaquette
	su2_product_tc(aux, phi + idr + l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc2(phi + idr + tp  + l.dg + xm , phi + pos + xm  + 2 * l.dg, aux);//t negative plaquette
	su2_product(aux, phi + pos + xm + l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);
	
	//for (int i = 0; i < l.updates; i++)
	//accept[id] += metropolis(A, phi + idr+2*l.dg, l.beta, l.epsilon, state + id);

	
	if (update == 0)
		accept[id] += heat_bath(A, phi + idr + 2 * l.dg, l.beta, state + id);


	else
		accept[id] += over_relaxation(A, phi + idr +2* l.dg);
}

__global__ void MC_update_odd_t(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l,int update) {//update one site

	int pos = blockDim.x * blockIdx.x + threadIdx.x;// define the position in the lattice
	int pos_y = pos % l.y;
	int pos_x = (pos % (l.x * l.y)) / l.y;
	int pos_t = pos / (l.x * l.y);
	int par_aux = (pos_t + pos_x) % 2;
	int id = (pos + l.V) * l.dt;
	int posr = pos * l.dt;
	int  yp, ym, xp, xm, tp, tm;
	
	yp = (1 - par_aux) * l.dt; //the positive derivative in z
	if (pos_y == (l.y - 1) && par_aux == 0)
		yp = (-l.y + 1) * l.dt;
	ym = -par_aux*l.dt; //the negative derivative in z
	if (pos_y == 0 && par_aux == 1)
		ym = (l.y - 1) * l.dt;

	xp = l.y * l.dt;////the positive derivative in x
	if (pos_x == (l.x - 1))
		xp = -l.y * (l.x - 1) * l.dt;
	xm = -l.y*l.dt;///the negative derivative in x
	if (pos_x == 0)
		xm = l.y * (l.x - 1) * l.dt;

	tp = l.y * l.x * l.dt;//the positive derivative in t
	if (pos_t == (l.t - 1))
		tp = -l.y * l.x * (l.t - 1) * l.dt;
	double aux[4], aux2[4], A[4];
	su2_zero(A);
	su2_product_tc(phi + posr + tp , phi + posr + yp  + 2 * l.dg, aux);//y positive plaquette
	su2_product_tc(aux, phi + id, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc2(phi + id + tp  + ym , phi + posr + ym  + 2 * l.dg, aux);//y negative plaquette
	su2_product(aux, phi + posr + ym , aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc(phi + posr + tp  + l.dg, phi + posr + xp  + 2 * l.dg, aux);//t positive plaquette
	su2_product_tc(aux, phi + id + l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);

	su2_product_tc2(phi + id + tp + l.dg + xm , phi + posr + xm  + 2 * l.dg, aux);//t negative plaquette
	su2_product(aux, phi + posr + xm  + l.dg, aux2);
	//su2_print_vec(aux2);
	su2_sum(A, aux2);
	
	//for (int i = 0; i < l.updates; i++)
	//accept[id/l.dt] += metropolis(A, phi + id+2*l.dg, l.beta, l.epsilon, state + pos);

	
	if (update == 0)
		accept[id / l.dt] += heat_bath(A, phi + id + 2 * l.dg, l.beta, state + pos);
	else
		accept[id / l.dt] += over_relaxation(A, phi + id +2* l.dg);
}

void mc_up_per(int nthreads,int nblocks,double* phi, struct lattice l, int* accept, curandStateMRG32k3a_t* devStates,int update) {
	MC_update_even_y << < nblocks / 2, nthreads >> > (phi, devStates, accept, l,update);
	MC_update_odd_y << < nblocks / 2, nthreads >> > (phi, devStates, accept, l,update);
	MC_update_even_x << < nblocks / 2, nthreads >> > (phi, devStates, accept, l,update);
	MC_update_odd_x<< < nblocks / 2, nthreads >> > (phi, devStates, accept, l,update);
	MC_update_even_t << < nblocks / 2, nthreads >> > (phi, devStates, accept, l,update);
	MC_update_odd_t << < nblocks / 2, nthreads >> > (phi, devStates, accept, l,update);

}

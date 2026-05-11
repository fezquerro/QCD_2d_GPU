


__global__ void MC_update_even_x_pcc(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l) {//update one site
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
	tm = -l.y * l.x*l.dt;//the negative derivative in t
	if (pos_t == 0)
		tm = l.y * l.x * (l.t - 1) * l.dt;
	if (!(pos_y == 0 && par_aux == 0)) {
		double aux[4], aux2[4], A[4];
		su2_zero(A);
		su2_product_tc(phi + pos + xp, phi + pos + yp + l.dg, aux);//y positive plaquette
		su2_product_tc(aux, phi + idr, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc2(phi + idr + xp + ym, phi + pos + ym + l.dg, aux);//y negative plaquette
		su2_product(aux, phi + pos + ym, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc(phi + pos + xp + 2 * l.dg, phi + pos + tp + l.dg, aux);//t positive plaquette
		su2_product_tc(aux, phi + idr + 2 * l.dg, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc2(phi + idr + xp + 2 * l.dg + tm, phi + pos + tm + l.dg, aux);//t negative plaquette
		su2_product(aux, phi + pos + tm + 2 * l.dg, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);
	
		//for (int i = 0; i < l.updates; i++)
		//accept[id] += metropolis(A, phi + idr + l.dg, l.beta, l.epsilon, state + id);
		accept[id] += heat_bath(A, phi + idr+l.dg, l.beta, state + id);
	}
}

__global__ void MC_update_odd_x_pcc(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l) {//update one site

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
	tm = -l.y * l.x*l.dt;//the negative derivative in t
	if (pos_t == 0)
		tm = l.y * l.x * (l.t - 1) * l.dt;
	if (!(pos_y == 0 && par_aux == 1)) {
		double aux[4], aux2[4], A[4];
		su2_zero(A);
		su2_product_tc(phi + posr + xp, phi + posr + yp + l.dg, aux);//y positive plaquette
		su2_product_tc(aux, phi + id, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc2(phi + id + xp + ym, phi + posr + ym + l.dg, aux);//y negative plaquette
		su2_product(aux, phi + posr + ym, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc(phi + posr + xp + 2 * l.dg, phi + posr + tp + l.dg, aux);//t positive plaquette
		su2_product_tc(aux, phi + id + 2 * l.dg, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc2(phi + id + xp + 2 * l.dg + tm, phi + posr + tm + l.dg, aux);//t negative plaquette
		su2_product(aux, phi + posr + tm + 2 * l.dg, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		//for (int i = 0; i < l.updates; i++)
		//accept[id / l.dt] += metropolis(A, phi + id + l.dg, l.beta, l.epsilon, state + pos);
		accept[id/l.dt] += heat_bath(A, phi + id+l.dg, l.beta, state + pos);
	}
}
__global__ void MC_update_even_t_pcc(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l) {//update one site
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
	tm = -l.y * l.x*l.dt;//the negative derivative in t
	if (pos_t == 0)
		tm = l.y * l.x * (l.t - 1) * l.dt;
	if (!(pos_y == 0 && par_aux == 0)) {
		double aux[4], aux2[4], A[4];
		su2_zero(A);
		su2_product_tc(phi + pos + tp, phi + pos + yp + 2 * l.dg, aux);//y positive plaquette
		su2_product_tc(aux, phi + idr, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc2(phi + idr + tp + ym, phi + pos + ym + 2 * l.dg, aux);//y negative plaquette
		su2_product(aux, phi + pos + ym, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc(phi + pos + tp + l.dg, phi + pos + xp + 2 * l.dg, aux);//t positive plaquette
		su2_product_tc(aux, phi + idr + l.dg, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc2(phi + idr + tp + l.dg + xm, phi + pos + xm + 2 * l.dg, aux);//t negative plaquette
		su2_product(aux, phi + pos + xm + l.dg, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		//for (int i = 0; i < l.updates; i++)
		//accept[id] += metropolis(A, phi + idr + 2 * l.dg, l.beta, l.epsilon, state + id);
		accept[id] += heat_bath(A, phi + idr+2*l.dg, l.beta, state + id);
	}
}

__global__ void MC_update_odd_t_pcc(double* phi, curandStateMRG32k3a_t* state, int* accept, struct lattice l) {//update one site

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
	tm = -l.y * l.x*l.dt;//the negative derivative in t
	if (pos_t == 0)
		tm = l.y * l.x * (l.t - 1) * l.dt;
	if(!(pos_y == 0 && par_aux == 1)) {
		double aux[4], aux2[4], A[4];
		su2_zero(A);
		su2_product_tc(phi + posr + tp, phi + posr + yp + 2 * l.dg, aux);//y positive plaquette
		su2_product_tc(aux, phi + id, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc2(phi + id + tp + ym, phi + posr + ym + 2 * l.dg, aux);//y negative plaquette
		su2_product(aux, phi + posr + ym, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc(phi + posr + tp + l.dg, phi + posr + xp + 2 * l.dg, aux);//t positive plaquette
		su2_product_tc(aux, phi + id + l.dg, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);

		su2_product_tc2(phi + id + tp + l.dg + xm, phi + posr + xm + 2 * l.dg, aux);//t negative plaquette
		su2_product(aux, phi + posr + xm + l.dg, aux2);
		//su2_print_vec(aux2);
		su2_sum(A, aux2);
		
		//for (int i = 0; i < l.updates; i++)
		//accept[id / l.dt] += metropolis(A, phi + id + 2 * l.dg, l.beta, l.epsilon, state + pos);
		accept[id/l.dt] += heat_bath(A, phi + id+2*l.dg, l.beta, state + pos);
	}
}

void mc_up_dir(int nthreads,int nblocks,double* phi, struct lattice l, int* accept, curandStateMRG32k3a_t* devStates) {
	MC_update_even_y << < nblocks / 2, nthreads >> > (phi, devStates, accept, l,0);
	MC_update_odd_y << < nblocks / 2, nthreads >> > (phi, devStates, accept, l,0);
	MC_update_even_x_pcc << < nblocks / 2, nthreads >> > (phi, devStates, accept, l);
	MC_update_odd_x_pcc<< < nblocks / 2, nthreads >> > (phi, devStates, accept, l);
	MC_update_even_t_pcc << < nblocks / 2, nthreads >> > (phi, devStates, accept, l);
	MC_update_odd_t_pcc << < nblocks / 2, nthreads >> > (phi, devStates, accept, l);

}


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>
#include "functions\su2.h"
#include "functions\lattice_setup.h"
#include "functions\mc_updates.h"
#include "functions\mc_updates_pcc.h"
#include "functions\observables.h"
#include "functions\parallel_reduction.h"
#include "functions\statistics.h"
#include "functions\output.h"


#define N 1000 // number of MC steps
#define N_term 100//number of MC steps to termalize
#define N_block 10 //number of times where the time left is printed
#define N_cor 1 //number of MC steps between measures

#define d_y 8//dimension y of the lattice
#define d_x 32 //dimension x of the lattice
#define d_t 32 //dimension t of the lattice

#define Boundary 0 //define boundary condition, 0 for Periodic and 1 for Perfect colour conductor(PCC)

#define betal 10 // coupling constant 
#define epsilonl 0.1 //epsilon for the metropolis algorithm
#define upd 1 //metropolis updates in every hit of each point
#define W 80 //number of autocorrelation coeficcientes saved 
#define pmeasure 8


int main()
{
	char nsave[2] = "1";//counter for saving different runs with the same parameters
	int co = 0;
	for (int co = 0; co < 1; co++) {//a loop for having different size of dy

		// we define the lattice parameters
		struct lattice l = { 10 * co + d_y,d_x,d_t, (10 * co + d_y) * d_x * d_t,3,4,12,betal,epsilonl,upd };// dimz,dimy,dimx,dimt,V,dimensions,size of the group,d*sg,beta,delta,upd
		struct lattice l2 = { l.y / 2,l.x,l.t,l.V / 2,3,4,12,l.beta,l.epsilon,upd };//same for half of the lattice
		printf("N %d Lt %d L %d Ly %d V %d Dim %d DimRep %d Dimtot %d \n", N, l.t, l.x, l.y,l.V,l.d,l.dg,l.dt);

		int nblocks, nthreads;//define the number of blocks and threads needed for the lattice
		if (l.V > 256) {
			nthreads = 256;
			nblocks = l.V / 256;
		}
		else {
			nthreads = l.V / 2;
			nblocks = 2;
		}
		printf("nblocks %d nthreads %d\n", nblocks, nthreads);


		curandStateMRG32k3a_t* devStates;// use the MRG32k random number generator from cuRAND
		cudaMalloc((void**)&devStates, l2.V * sizeof(curandStateMRG32k3a_t));//initialize the GPU random number generator state
		setup_kernel << <nblocks / 2, nthreads >> > (devStates, time(NULL));//setup the GPU random number generator


		double* phi = (double*)malloc(l.V*l.dt * sizeof(double)), * phid; cudaMalloc(&phid, l.V * l.dt * sizeof(double));//create the lattice array in cpu & GPU


        #if Boundary==0
		printf("PERIODIC BOUNDARY CONDITIONS\n\n");
		rand_init2 << <nblocks / 2, nthreads >> > (phid, devStates, l2);//create the lattice with random entries in (-1,1)
		//id_init2 << <nblocks / 2, nthreads >> > (phid, l2);//create the lattice with unity

        #else
		printf("PERFECT COLOUR CONDUCTOR BOUNDARY CONDITIONS\n\n");
		rand_init2_pcc << <nblocks / 2, nthreads >> > (phid, devStates, l2);//create the lattice with random entries in (-1,1) and unity in y=0 for PCC
		//id_init2 << <nblocks / 2, nthreads >> > (phid, l2);//create the lattice with unity

        #endif
				

		//cudaMemcpy(phi, phid, l.V * l.dt * sizeof(double), cudaMemcpyDeviceToHost); printlattice(phi, l2);
		//cudaMemcpy(phid, phi, l.V * l.dt * sizeof(double), cudaMemcpyHostToDevice); //printlattice(phi, l2);
		//plaquette_printd<<<nblocks, nthreads >>>(phid, l2);
		cudaMemcpy(phi, phid, l.V * l.dt * sizeof(double), cudaMemcpyDeviceToHost); print_det(phi, l); //check average det (should be 1)

		int* accept, * accepth = (int*)malloc(l.V * sizeof(int));// create the cuda array that measures the accept rate
		cudaMalloc(&accept, l.V * sizeof(int)); cudaMemset(accept, 0, l.V * sizeof(int));

		double* sum,*sum2,*sumP;// create the cuda arrays to save the results
		cudaMalloc(&sum, l.V * sizeof(double)); cudaMemset(sum, 0, l.V * sizeof(double)); cudaMalloc(&sum2, l.V * sizeof(double)); cudaMemset(sum2, 0, l.V * sizeof(double));
		//cudaMalloc(&sumP, P_R*l.V/l.x * sizeof(double)); cudaMemset(sumP, 0, P_R*l.V/l.x * sizeof(double));



		double* end, * en = (double*)malloc(N * sizeof(double)); cudaMalloc(&end, N * sizeof(double)); cudaMemset(end, 0, N * sizeof(double));//create the array to save measures each MC step
		double* end2, * en2 = (double*)malloc(N * sizeof(double)); cudaMalloc(&end2, N * sizeof(double)); cudaMemset(end2, 0, N * sizeof(double));
		double* endaux; cudaMalloc(&endaux, l.V / 2 * sizeof(double)); cudaMemset(endaux, 0, l.V / 2 * sizeof(double));



		int i=1,j, i_block = 0,k;
		int t_ini, t_fin, t_n, block_size = N / N_block;
		for (j = 0; j < N_term; j++) {
			if (j % 100 == 0)
				normalize<<<nblocks,nthreads>>>(phid, l); //normalize the links to keep unitarity
           #if Boundary==0
			mc_up_per(nthreads, nblocks, phid, l2, accept, devStates, 0); //update for periodic boundary conditions
           #else
			mc_up_pcc(nthreads, nblocks, phid, l2, accept, devStates); //update for perfect colour conductor boundary conditions
           #endif
			

		}
		normalize << <nblocks, nthreads >> > (phid, l);
		//plaquette_printd << <nblocks, nthreads >> > (phid, l2);
		cudaMemcpy(phi, phid, l.V * l.dt * sizeof(double), cudaMemcpyDeviceToHost); print_det(phi, l); //check average det (should be 1)
		cudaMemset(accept, 0, l.V * sizeof(int));
		t_ini = time(NULL); t_n = time(NULL);
		for (i = 1; i < N + 1; i++) {
			for (j = 0; j < N_cor; j++) {
				
                #if Boundary==0
				 mc_up_per(nthreads, nblocks, phid, l2, accept, devStates, 0); //update for periodic boundary conditions
                #else
				 mc_up_pcc(nthreads, nblocks, phid, l2, accept, devStates); //update for perfect colour conductor boundary conditions
                #endif
			}
			if (i % 100 == 0)
				normalize << <nblocks, nthreads >> > (phid, l); //normalize the links to keep unitarity

			//plaquette_av << <nblocks, nthreads >> > (phid, sum, l2);
			//energy << <nblocks, nthreads >> > (phid, sum, l2);
		    en_av << <nblocks, nthreads >> > (phid, sum,sum2, l2);//compute the energy and average of plaquettes

            #if (d_x==96 && d_t==96) //compute with parallel reduction for lattice 96 x 96 x y 
			mean_lattice_par_96(sum, endaux, end + i - 1, l.V, l.y);
		    mean_lattice_par_96(sum2, endaux, end2 + i - 1, l.V, l.y);

            #elif (d_x==64 && d_t==64) //compute with parallel reduction for lattice 64 x 64 x y 
			mean_lattice_par_64(sum, endaux, end + i - 1, l.V, l.y);
		    mean_lattice_par_64(sum2, endaux, end2 + i - 1,  l.V, l.y);

            #elif (d_x==32 && d_t==32) //compute with parallel reduction for lattice 32 x 32 x y 
			mean_lattice_par_32(sum, endaux, end + i - 1,  l.V, l.y);
			mean_lattice_par_32(sum2, endaux, end2 + i - 1,  l.V, l.y);

            #else //compute the sum with just one core only for other lattices (slow) 
			mean_lattice_small(sum,  end + i - 1, l.V);
			mean_lattice_small(sum2,  end2 + i - 1, l.V);
            #endif


			
			if (i % block_size == 0) {
				//eblock_average[i_block] = mean_lattice(sum,sumh, block_size, l,true); i_block++;
				printf("%lf Time %d Left(min)%lf\n", (double)i / N, time(NULL) - t_n, (double)(time(NULL) - t_ini) * (N - i) / i / 60);
				t_n = time(NULL);
			}

		}
		t_fin = time(NULL);
		printf("Time %d\n", t_fin - t_ini);
		//cudaMemcpy(phi, phid, l.V*l.dt * sizeof(double), cudaMemcpyDeviceToHost); printlattice(phi,l);
		//cudaMemcpy(phi, phid, l.V * l.dt * sizeof(double), cudaMemcpyDeviceToHost); print_det(phi, l);
		//plaquette_printd << <nblocks, nthreads >> > (phid, l2);

		cudaMemcpy(en, end, N * sizeof(double), cudaMemcpyDeviceToHost); //copy the results into cpu
		cudaMemcpy(en2, end2, N * sizeof(double), cudaMemcpyDeviceToHost); //copy the results into cpu

		double saccept = mean_lattice(accept, accepth, N * N_cor*l.d*l.updates, l, true);
        #if Boundary==1
		saccept = saccept / ((double)(l.y-1)/l.y+1/(double)l.y/3);
        #endif
		printf("accept %lf\n\n",saccept);

		double energy[2], energy2[2], energy3[2*P_R]; //mean_variance(eblock_average, N_block, energy,true);
		mean_variance(en, N, energy, true);
		mean_variance(en2, N, energy2, true);


        #if (Boundary==0)
		 char bound[12] = "per";
        #else
		 char bound[12] = "pcc";
        #endif

	    //save_energy(energy[0], energy[1], N_term, N, N_cor, saccept, l,bound,nsave); //save energy observable
		//save_energy_av(energy[0], energy[1], energy2[0], energy2[1], N_term, N, N_cor, saccept, l,bound,nsave); //save average plaquette
		//save_en_seq(en2, en, N_term, N, N_cor, 1, l, bound, nsave);//save each step of the energy and plaquette

	   double autoc[W], autoc2[W], autoc3[W]; autocorrelation(en, N, W, energy[0], energy[1], autoc); autocorrelation(en2, N, W, energy2[0], energy2[1], autoc2);
	 //save_autocorrelation(autoc, W, N, N_cor, N_term, l,bound,nsave);save_autocorrelation_av(autoc2, W, N, N_cor, N_term, l,bound,nsave);//save autocorrelation
	

		free(phi);  free(accepth); cudaFree(phid); cudaFree(accept); cudaFree(sum); cudaFree(devStates); cudaFree(sum2);  
		free(en); free(en2);  cudaFree(endaux); cudaFree(end); cudaFree(end2); 
		
    }
    
}

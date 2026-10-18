
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include <math.h>

//define the flag for each update algorithm
#define METROPOLIS 0 //Metropolis, upd hits per link with proposals of size epsilonl
#define HEATBATH 1 //heat-bath, new link drawn from its exact local distribution (decorrelates faster)
#define ALGO HEATBATH //select the update algorithm

#include "functions/su2.h"
#include "functions/lattice_setup.h"
#include "functions/mc_updates.h"
#include "functions/mc_updates_pcc.h"
#include "functions/mc_updates_pmc.h"
#include "functions/observables.h"
#include "functions/parallel_reduction.h"
#include "functions/statistics.h"
#include "functions/output.h"

#define d_y 4//dimension y of the lattice
#define d_x 96 //dimension x of the lattice
#define d_t 96 //dimension t of the lattice
#define betal 20 // coupling constant 
#define Boundary 2 //define boundary condition, 0 for Periodic and 1 for Perfect colour conductor(PCC)

#define readconf 0 //read configuration if 1 and termalize, read conf and not termalize if 2
#define init 0 // 0 hot start, 1 cold start
#define sconf 0 // save configuration if 0
#define save 0 //save if 0

#define N 50000 // number of MC steps
#define N_term 20000//number of MC steps to termalize
#define N_block 20 //number of time the time left is printed
#define Nblock_term 10 //number of times the time left is printed for thermalization
#define Nsave 5 //number of times the results are saved and printed during simulation
#define N_cor 1 //number of MC steps between measures


#define epsilonl 0.1 //epsilon for the metropolis algorithm
#define upd 1 //metropolis updates in every hit of each point
#define W 80 //number of autocorrelation terms saved 
#define pmeasure 8


int main()
{
	char nsave[2] = "1";//counter for saving different runs with the same parameters
	char nconfr[2] = "1";
	char nconfs[2] = "1";
	int Nterms = 0, inits;
	for (int co = 0; co < 1; co++) {//a loop for having different size of dy

		// we define the lattice parameters
		struct lattice l = { 10 * co + d_y,d_x,d_t, (10 * co + d_y) * d_x * d_t,3,4,12,betal,epsilonl,upd };// dimz,dimy,dimx,dimt,V,dimensions,size of the group,d*sg,beta,delta,upd
		struct lattice l2 = { l.y / 2,l.x,l.t,l.V / 2,3,4,12,l.beta,l.epsilon,upd };//same for half of the lattice
		printf("N %d Lt %d L %d Ly %d V %d Dim %d DimRep %d Dimtot %d \n", N, l.t, l.x, l.y,l.V,l.d,l.dg,l.dt);
		if (l.t % 2 || l.x % 2 || l.y % 2) { // the checkerboard decomposition needs even sizes in every direction
			printf("Lt, Lx and Ly must be even\n");
			return 1;
		}

		int nblocks, nthreads;//define the number of blocks and threads needed for the lattice
		nthreads = 256; // largest block size <= 256 that divides the half lattice, so no site is skipped
		while (l2.V % nthreads != 0)
			nthreads--;
		nblocks = 2 * (l2.V / nthreads); // blocks for the full lattice, the MC kernels use nblocks / 2
		printf("nblocks %d nthreads %d\n", nblocks, nthreads);
		if (ALGO == METROPOLIS) printf("METROPOLIS UPDATES epsilon %.2f hits %d\n", l.epsilon, l.updates);
		if (ALGO == HEATBATH) printf("HEAT-BATH UPDATES\n");
		int hits = (ALGO == METROPOLIS) ? l.updates : 1; //updates of each link per sweep, to normalize the acceptance


		curandStateMRG32k3a_t* devStates;// use the MRG32k random number generator from cuRAND
		cudaMalloc((void**)&devStates, l2.V * sizeof(curandStateMRG32k3a_t));//initialize the GPU random number generator state
		setup_kernel << <nblocks / 2, nthreads >> > (devStates, time(NULL));//setup the GPU random number generator


		double* phi = (double*)malloc(l.V*l.dt * sizeof(double)), * phid; cudaMalloc(&phid, l.V * l.dt * sizeof(double));//create the lattice array in cpu & GPU


#if Boundary==0
		printf("PERIODIC BOUNDARY CONDITIONS\n\n");
		char bound[12] = "per";
 #if readconf==0
		inits = init;
  #if init==0
		rand_init2 << <nblocks / 2, nthreads >> > (phid, devStates, l2);//create the lattice with random entries in (-1,1)

  #else
		id_init2 << <nblocks / 2, nthreads >> > (phid, l2);//create the lattice with unity
  #endif

 #else
		read_conf(phi, l, &Nterms, &inits, bound, nconfr);
		cudaMemcpy(phid, phi, l.V * l.dt * sizeof(double), cudaMemcpyHostToDevice);
 #endif	

#elif Boundary==1
		printf("PERFECT COLOUR CONDUCTOR BOUNDARY CONDITIONS\n\n");
		char bound[12] = "pcc";
 #if readconf==0
		inits = init;
  #if init==0
		rand_init2_pcc << <nblocks / 2, nthreads >> > (phid, devStates, l2);//create the lattice with random entries in (-1,1) and unity in y=0 for PCC

  #else
		id_init2 << <nblocks / 2, nthreads >> > (phid, l2);//create the lattice with unity
  #endif

 #else
		read_conf(phi, l, &Nterms, &inits, bound, nconfr);
		cudaMemcpy(phid, phi, l.V * l.dt * sizeof(double), cudaMemcpyHostToDevice);
 #endif	

#else
		printf("PERFECT MAGNETIC CONDUCTOR BOUNDARY CONDITIONS\n\n");
		char bound[12] = "pmc";
#if readconf==0
		inits = init;
#if init==0
		rand_init2 << <nblocks / 2, nthreads >> > (phid, devStates, l2);//create the lattice with random entries in (-1,1)

#else
		id_init2 << <nblocks / 2, nthreads >> > (phid, l2);//create the lattice with unity
#endif

#else
		read_conf(phi, l, &Nterms, &inits, bound, nconfr);
		cudaMemcpy(phid, phi, l.V * l.dt * sizeof(double), cudaMemcpyHostToDevice);
#endif	

#endif
				

		//cudaMemcpy(phi, phid, l.V * l.dt * sizeof(double), cudaMemcpyDeviceToHost); printlattice(phi, l2);
		//cudaMemcpy(phid, phi, l.V * l.dt * sizeof(double), cudaMemcpyHostToDevice); //printlattice(phi, l2);
		//plaquette_printd<<<nblocks, nthreads >>>(phid, l2);
		cudaMemcpy(phi, phid, l.V * l.dt * sizeof(double), cudaMemcpyDeviceToHost); print_det(phi, l); //check average det (should be 1)

		int* accept, * accepth = (int*)malloc(l.V * sizeof(int));// create the cuda array that measures the accept rate
		cudaMalloc(&accept, l.V * sizeof(int)); cudaMemset(accept, 0, l.V * sizeof(int));

		double* sum,*sum2;// create the cuda arrays to save the results
		cudaMalloc(&sum, l.V * sizeof(double)); cudaMemset(sum, 0, l.V * sizeof(double)); cudaMalloc(&sum2, l.V * sizeof(double)); cudaMemset(sum2, 0, l.V * sizeof(double));



		double* end, * en = (double*)malloc(N * sizeof(double)); cudaMalloc(&end, N * sizeof(double)); cudaMemset(end, 0, N * sizeof(double));//create the array to save measures each MC step
		double* end2, * en2 = (double*)malloc(N * sizeof(double)); cudaMalloc(&end2, N * sizeof(double)); cudaMemset(end2, 0, N * sizeof(double));
		double* endaux; cudaMalloc(&endaux, l.V / 2 * sizeof(double)); cudaMemset(endaux, 0, l.V / 2 * sizeof(double));



		int i,j;
		int t_ini,t_ini_mc, t_fin, t_n, block_size_t = N_term / Nblock_term, block_size = N / N_block, block_s_size = N / Nsave;//blocks for print time and saving
		t_ini = time(NULL); t_n = time(NULL);
		double energy[2], energy2[2]; //variables to save the energy and av plaquette
		double autoc[W], autoc2[W]; //variables to save autocorrelation
		double saccept;
#if readconf<2
		for (i = 1; i < N_term+1; i++) {
			if (i % 100 == 0)
				normalize<<<nblocks,nthreads>>>(phid, l); //normalize the links to keep unitarity
           #if Boundary==0
			mc_up_per(nthreads, nblocks, phid, l2, accept, devStates, 0); //update for periodic boundary conditions
           #elif Boundary==1
			mc_up_pcc(nthreads, nblocks, phid, l2, accept, devStates); //update for perfect colour conductor boundary conditions
           #else
			mc_up_pmc(nthreads, nblocks, phid, l2, accept, devStates); //update for perfect colour conductor boundary conditions
           #endif
			if (i % block_size_t == 0) {
				printf("%lf Time %d Left(min)%lf\n", (double)i / (N_term), time(NULL) - t_n, (double)(time(NULL) - t_ini) * (N_term - i) / i / 60);
				fflush(stdout);
				t_n = time(NULL);
			}

		}
		normalize << <nblocks, nthreads >> > (phid, l);
		Nterms = N_term + Nterms; //total number of thermalization steps of this configuration
  #if sconf == 0

		cudaMemcpy(phi, phid, l.V * l.dt * sizeof(double), cudaMemcpyDeviceToHost);

		save_conf(phi, l, Nterms, inits, bound, nconfs);
  #endif
#endif
		//plaquette_printd << <nblocks, nthreads >> > (phid, l2);
		cudaMemcpy(phi, phid, l.V * l.dt * sizeof(double), cudaMemcpyDeviceToHost); print_det(phi, l); //check average det (should be 1)
		cudaMemset(accept, 0, l.V * sizeof(int));
		t_ini_mc = time(NULL); t_n = time(NULL);
		for (i = 1; i < N + 1; i++) {
			for (j = 0; j < N_cor; j++) {
				
               #if Boundary==0
				   mc_up_per(nthreads, nblocks, phid, l2, accept, devStates, 0); //update for periodic boundary conditions
               #elif Boundary==1
				mc_up_pcc(nthreads, nblocks, phid, l2, accept, devStates); //update for perfect colour conductor boundary conditions
               #else
				mc_up_pmc(nthreads, nblocks, phid, l2, accept, devStates); //update for perfect colour conductor boundary conditions
               #endif
			}
			if (i % 100 == 0)
				normalize << <nblocks, nthreads >> > (phid, l); //normalize the links to keep unitarity

			//plaquette_av << <nblocks, nthreads >> > (phid, sum, l2);
			//energy << <nblocks, nthreads >> > (phid, sum, l2);
           #if Boundary==0 || Boundary==1
		    en_av << <nblocks, nthreads >> > (phid, sum,sum2, l2);//compute the energy and average of plaquettes
           #else
			en_av_neu << <nblocks, nthreads >> > (phid, sum, sum2, l2);//compute the energy and average of plaquettes
           #endif
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
				printf("%lf Time %d Left(min)%lf\n", (double)i / N, time(NULL) - t_n, (double)(time(NULL) - t_ini_mc) * (N - i) / i / 60);
				t_n = time(NULL);
				fflush(stdout);
			}

			if (i % block_s_size == 0) {//compute average and save them
				cudaMemcpy(en, end, i * sizeof(double), cudaMemcpyDeviceToHost); //copy the results into cpu
				cudaMemcpy(en2, end2, i * sizeof(double), cudaMemcpyDeviceToHost); //copy the results into cpu
				printf("Energy "); mean_variance(en, i, energy, true);autocorrelation(en, i, W, energy[0], energy[1], autoc); 
				printf("Plaquette "); mean_variance(en2, i, energy2, true); autocorrelation(en2, i, W, energy2[0], energy2[1], autoc2);
				fflush(stdout);
#if save==0
				saccept = mean_lattice(accept, accepth, i * N_cor * l.d * hits, l, false);
        #if Boundary==1
				saccept = saccept / ((double)(l.y-1)/l.y+1/(double)l.y/3);
        #endif
				save_energy(energy[0], energy[1], Nterms, N, N_cor, saccept, l, bound, nsave); //save energy observable
				save_energy_av(energy[0], energy[1], energy2[0], energy2[1], Nterms, N, N_cor, saccept, l, bound, nsave); //save average plaquette
				save_autocorrelation(autoc, W, i, N_cor, Nterms, l, bound, nsave); save_autocorrelation_av(autoc2, W, i, N_cor, Nterms, l, bound, nsave);
				save_en_seq(en2, en, Nterms, i, N_cor, 1, l, bound, nsave);//save each step of the energy and plaquette
#endif
			}

		}
		t_fin = time(NULL);
		printf("Time %d\n", t_fin - t_ini);
		//cudaMemcpy(phi, phid, l.V*l.dt * sizeof(double), cudaMemcpyDeviceToHost); printlattice(phi,l);
		//cudaMemcpy(phi, phid, l.V * l.dt * sizeof(double), cudaMemcpyDeviceToHost); print_det(phi, l);
		//plaquette_printd << <nblocks, nthreads >> > (phid, l2);

		cudaMemcpy(en, end, N * sizeof(double), cudaMemcpyDeviceToHost); //copy the results into cpu
		cudaMemcpy(en2, end2, N * sizeof(double), cudaMemcpyDeviceToHost); //copy the results into cpu

		 saccept = mean_lattice(accept, accepth, N * N_cor * l.d * hits, l, true);
        #if Boundary==1
		saccept = saccept / ((double)(l.y-1)/l.y+1/(double)l.y/3);
        #endif
		printf("accept %lf\n\n",saccept);

		printf("Energy "); mean_variance(en, N, energy, true); autocorrelation(en, N, W, energy[0], energy[1], autoc);
		printf("Plaquette "); mean_variance(en2, N, energy2, true); autocorrelation(en2, N, W, energy2[0], energy2[1], autoc2);		   
#if save==0

	    save_energy(energy[0], energy[1], Nterms, N, N_cor, saccept, l,bound,nsave); //save energy observable
		save_energy_av(energy[0], energy[1], energy2[0], energy2[1], Nterms, N, N_cor, saccept, l,bound,nsave); //save average plaquette
		save_en_seq(en2, en, Nterms, N, N_cor, 1, l, bound, nsave);//save each step of the energy and plaquette
	    save_autocorrelation(autoc, W, N, N_cor, Nterms, l,bound,nsave);save_autocorrelation_av(autoc2, W, N, N_cor, Nterms, l,bound,nsave);//save autocorrelation
#endif
	

		free(phi);  free(accepth); cudaFree(phid); cudaFree(accept); cudaFree(sum); cudaFree(devStates); cudaFree(sum2);  
		free(en); free(en2);  cudaFree(endaux); cudaFree(end); cudaFree(end2); 
		
    }
    
}

# Monte Carlo SU(2) Lattice Gauge Theory GPU Simulator (2+1)

**A high-performance lattice simulation code for GPU-accelerated Monte Carlo simulation of SU(2) Yang-Mills theory in 2+1 dimensions using CUDA.**

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Configuration](#configuration)
- [Compilation](#compilation)
- [Output files](#output-files)
- [Validation](#validation)

## Overview

This project implements a Monte Carlo simulation of pure SU(2) lattice gauge theory in three-dimensional (2+1) Euclidean spacetime using the Wilson plaquette action. The average plaquette and the energy are computed. The code uses **CUDA** for NVIDIA GPUs and is parallelized at the maximum extent allowed by the Wilson action (checkerboard decomposition, one kernel per link direction and parity). Three different boundary conditions are implemented for the y spatial direction: periodic, perfect colour conductor (PCC) and perfect magnetic conductor (PMC) boundary conditions.

This code was primarily developed and used to generate the results presented in the PhD thesis **[Properties of the quantum vacuum in non-abelian gauge theories](https://arxiv.org/abs/2605.18220)**. See Chapter 4 for an explanation of the boundary conditions and the parallelization. The results of this code are presented in Chapter 5 and Appendix B.

## Features

- **Efficient SU(2) updates**: heat-bath (Kennedy-Pendleton, with `sincospi` and `rsqrt` normalization to reduce the double precision work) or Metropolis, selected with the `ALGO` flag, fully implemented in CUDA kernels (an over-relaxation kernel is also available for the periodic updates)
- **Fast exact Metropolis test**: the acceptance exp(-dS) is first estimated in single precision and only computed in double precision when the single precision value is too close to the random number to decide, so the decision is always the same as in double precision
- **GPU Acceleration**: High-performance CUDA implementation with the MRG32k3a generator of cuRAND
- **Flexible lattice sizes**: any even $N_t \times N_x \times N_y$ lattice (the size of every direction is checked to be even, as required by the checkerboard decomposition); the block size is chosen as the largest one ≤ 256 that divides half of the volume, so no site is skipped
- **Boundary conditions** in the y direction (`Boundary` flag):
  - `0` periodic
  - `1` perfect colour conductor: the x and t links of the plane y = 0 are fixed to the identity
  - `2` perfect magnetic conductor: F_02 = F_12 = 0 at the boundary, so the y-x and y-t plaquettes between y = Ly-1 and y = 0 are fixed to the identity (they do not enter the updates of the boundary links and count as 1 in the average plaquette)
- **Observables**:
  - Plaquette (average action)
  - Energy (first component of the energy-momentum tensor), measured as the mean over sites of -(1 - P_tx), which uses the t ↔ x symmetry of the lattice (Lt = Lx)
- **Thermalization and autocorrelation analysis**
- **Restarts**: configurations can be saved and read again; the total number of thermalization sweeps is accumulated and written in every output file

## Configuration

All parameters are `#define`s at the top of `kernel.cu`:

| Flag | Meaning |
|---|---|
| `ALGO` | `HEATBATH` or `METROPOLIS` |
| `d_t`, `d_x`, `d_y` | Lattice size (all even) |
| `betal` | Coupling β |
| `Boundary` | 0 periodic, 1 PCC, 2 PMC |
| `readconf` | 0 new start, 1 read configuration and thermalize, 2 read configuration without thermalization |
| `init` | 0 hot start, 1 cold start |
| `sconf`, `save` | 0 to save the configuration / the results |
| `N`, `N_term`, `N_cor` | Measurements, thermalization sweeps, sweeps between measurements |
| `epsilonl`, `upd` | Metropolis proposal size and number of hits per link |
| `W` | Number of autocorrelation terms saved |

The energy and the average plaquette are computed and summed over the lattice in a single kernel (block sums in shared memory added with `atomicAdd`), for any lattice size.

## Compilation

```bash
nvcc -O3 -arch=sm_86 kernel.cu -o QCD_2d
```

(set `-arch` for your GPU; on Windows `cl.exe` from Visual Studio must be in the `PATH` or given with `-ccbin`). A Visual Studio project can also be used.

## Output files

The names contain the lattice size, β, the boundary condition (`per`, `pcc`, `pmc`) and a run counter:

| File | Content |
|---|---|
| `Energy_*.txt` | Mean energy and naive error |
| `Energy_av_*.txt` | Mean energy and average plaquette with naive errors |
| `Autocorrelation_*.txt`, `Autocorrelation_av_*.txt` | Normalized autocorrelation of the energy and the plaquette |
| `Check_*.txt` | Energy and plaquette of every measurement |
| `Config_*.txt`, `Config_*_data.txt` | Binary configuration and its thermalization data |

The errors in the files do not include the autocorrelation; use the `Check_` files (binning) or the autocorrelation files for the final errors.



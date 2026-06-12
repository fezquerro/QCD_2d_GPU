# Monte Carlo SU(2) Lattice Gauge Theory GPU Simulator (2+1)

**A high-performance lattice simulation code for GPU-accelerated Monte Carlo simulation of SU(2) Yang-Mills theory in 2+1 dimensions using CUDA.**

## Table of Contents
- [Overview](#overview)
- [Features](#features)

## Overview

This project implements a Monte Carlo simulation of pure SU(2) lattice gauge theory in four-dimensional Euclidean spacetime using the Wilson plaquette action. The average plaquette and internal energy are computed. The code uses **CUDA** for NVIDIA GPUs and is parallelized at the maximum extend allowed by the Wilson action. Two different boundary conditions are implemented for the second spatial dimensions, periodic and perfect colour conductor boundaty conditions. 

This code was primarily developed and used to generate the results presented in the PhD thesis **[Properties of the quantum vacuum in non-abelian gauge theories](https://arxiv.org/abs/2605.18220)**. See Chapter 4 for an explanation of the two different boundary conditions and parallelization. The results of this code are presented in Chapter 5 and Appendix B.

## Features

- **Efficient SU(2) updates**: Heat-bath algorithm, over-relaxation and metropolis, fully implemented in CUDA kernels
- **GPU Acceleration**: High-performance CUDA implementation
- **Flexible lattice sizes**: Support for different $N_t \times N_x \times N_y$ lattices and coupling constants $\beta$
- **Boundary conditions**: Periodic and perfect colour conductor boundary conditions
- **Observables**:
  - Plaquette (average action)
  - Energy (first component of the energy-momentum tensor)
- **Thermalization and autocorrelation analysis**
- **Configurable random number generator** (with reproducible seeds, CUDA-compatible)



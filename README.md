# Monte Carlo SU(2) Lattice Gauge Theory GPU Simulator (2+1)

**A high-performance lattice simulation code for GPU-accelerated Monte Carlo simulation of SU(2) Yang-Mills theory in 2+1 dimensions using CUDA.**

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Running](#running)

## Overview

This project implements a Monte Carlo simulation of pure SU(2) lattice gauge theory in three-dimensional Euclidean spacetime (2+1 dimensions) using the Wilson plaquette action. The average plaquette and internal energy are computed. The code uses **CUDA** for NVIDIA GPUs, . 

Two different boundary conditions are implemented in the spatial directions: periodic and perfect colour conductor boundary conditions.

This code is a GPU port and adaptation optimized for 2+1 dimensions.

## Features

- **Efficient SU(2) updates**: Heat-bath algorithm + over-relaxation, fully implemented in CUDA kernels
- **GPU Acceleration**: High-performance CUDA implementation with optimized memory access patterns
- **Flexible lattice sizes**: Support for different $N_t \times N_x \times N_y$ lattices and coupling constants $\beta$
- **Boundary conditions**: Periodic and perfect colour conductor boundary conditions
- **Observables**:
  - Plaquette (average action)
  - Energy (first component of the energy-momentum tensor)
- **Thermalization and autocorrelation analysis**
- **Checkpointing** for long runs
- **Configurable random number generator** (with reproducible seeds, CUDA-compatible)
- **High performance**: Leverages massive GPU parallelism for large-scale simulations

## Running

### Prerequisites
- NVIDIA GPU with Compute Capability ≥ 3.5 (recommended: Ampere or newer)
- CUDA Toolkit installed

### Compilation

```bash
nvcc -o su2_lgt_2p1 main.cu -lm -O3 -arch=sm_80

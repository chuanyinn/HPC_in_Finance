# CUDA Monte Carlo European Call Option Pricing
This project implements the Monte Carlo simulation method for pricing European call options using CUDA for parallel computation.

## Requirements
- CUDA Toolkit
- NVIDIA GPU
- C++ Compiler

## Setup on Midway
```sh
module load cuda/11.5
nvcc -lcurand monte_carlo.cu -o monte_carlo
sinteractive --partition=gpu --gres=gpu:1 --time=0:30:00 --account=finm32950
./monte_carlo
```

## Introduction
Price the following European call options using Monte Carlo technique, and measure time to price them. Write the results to console.
1. S = 90;  r = 0.01; v = 0.3, T = 1 (year),    N = 5 million
1. S = 95;  r = 0.02, v = 0.3; T = 1.2 (years), N = 5 million
1. S = 100; r = 0.03; v = 0.3; T = 1.5 (years), N = 5 million
1. S = 105; r = 0.04, v = 0.3; T = 2 (years),   N = 5 million
1. S = 110; r = 0.05, v = 0.3; T = 2.5 (years), N = 5 million
K = 100 for all cases.

Input Parameters:
- `S`: Initial stock price.
- `K`: Strike price.
- `T`: Time to maturity in years.
- `r`: Risk-free interest rate.
- `v`: Volatility.
- `N`: Number of time steps.

CUDA Steps
1. allocate and initialize CPU memory
1. allocate GPU memory - include error handling
1. copy data (inputs) to GPU memory (from CPU memory)
1. launch CUDA kernel; compute using `MonteCarloKernel`; generate results on GPU memory
1. copy data (results) to CPU memory (from GPU memory)
1. deallocate GPU memory
1. deallocate CPU memory

Device Algorithm `MonteCarloKernel`
1. Get the thread id. Since 5 million is larger than what can fit on a grid, we use a grid-stride loop.
1. Compute final stock price. For an arbitrary initial value S0 the geometric Brownian motion SDE has the analytic solution (under Ito's interpretation):
$$S_T = S_0 \exp{(r - \sigma^2 / 2)T + \sigma W_T}$$
1. In our Monte Carlo simulation, we randomize the Brownian motion variable using 
$$W_t \sim N(0, \sqrt{T})$$
1. We generate the random numbers directly within the device function (CUDA kernel) as it might be more efficient for Monte Carlo simulations
1. Each call price is the discounted payoff:
$$C_0 = (S_T - K)^+ \exp{-rT}$$
1. The final call price of is the average of 5 million option prices. Perform this on the host because threads need to be independent.

Resources
- https://docs.nvidia.com/cuda/curand/device-api-overview.html#distributions
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix
- https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__CUDA__MATH__INTRINSIC__SINGLE
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities

## Optimize
For the complete implementation, see the monte_carlo.cu file in this repository.
- benchmark against plain C++ code, C++ with openmp parallelization
- use local variables stored in registers rather than accessing global memory
- reduce computational overhead by intrinsics
- test on different execution configurations - no change, unlikely the bottleneck
- curand call from GPU vs from CPU

## Performance
```sh
[chuanyin@midway3-0286 final_CY]$ ./monte_carlo 
Elapsed time: 179.248 ms
Option a: 7.32054
Option b: 11.3065
Option c: 16.5482
Option d: 23.6542
Option e: 31.329
```

```sh
[chuanyin@midway3-0285 final_CY]$ nvprof ./monte_carlo_curand
==1096009== NVPROF is profiling process 1096009, command: ./monte_carlo_curand
Elapsed time: 187.198 ms
Option a: 7.31012
Option b: 11.2775
Option c: 16.5284
Option d: 23.6197
Option e: 31.3183
```

Below is a comparison of execution times for different implementations:
| Implementation         | Time (ms) |
|------------------------|-----------|
| CPU (single-threaded)  | 722.625      |
| CPU (OpenMP multi-threaded) | 51233  |
| GPU (CUDA) <<<N/256, 256>>>             | 179.248        |
| GPU (CUDA) <<<N/512, 512>>>            | 178.191        |
| GPU (CUDA) <<<N/256, 1024>>>            | 178.345        |
| GPU (CUDA) host call curand            | 184.6 ms        |

Profiling
```sh
nvprof ./add_vector
==1693823== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   56.80%  62.930ms         1  62.930ms  62.930ms  62.930ms  initRandomStates(curandStateXORWOW*, unsigned int)
                   21.35%  23.652ms         5  4.7304ms  4.0820ms  7.3032ms  [CUDA memcpy DtoH]
                   17.51%  19.399ms         5  3.8798ms  3.8439ms  3.9080ms  [CUDA memcpy HtoD]
                    4.34%  4.8045ms         5  960.90us  954.01us  971.16us  MonteCarloKernel(float*, float, float, float, float, float*, curandStateXORWOW*)
      API calls:   50.12%  116.18ms         3  38.726ms  105.91us  115.95ms  cudaMalloc
                   29.34%  68.008ms         6  11.335ms  1.0016ms  62.933ms  cudaDeviceSynchronize
                   19.66%  45.584ms        10  4.5584ms  4.0357ms  8.0256ms  cudaMemcpy
```



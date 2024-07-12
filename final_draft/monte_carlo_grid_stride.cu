#include <iostream>
#include <stdio.h>
#include <cmath>
#include <chrono> // For measuring execution time
#include <cuda.h> 
#include <curand.h>
#include <curand_kernel.h>

using namespace std;

// Price the following European call options using Monte Carlo technique, 
// and measure time to price them. Write the results to console.

// Useful constants
// Maximum grid size: x-dimension: 2^31 - 1 (for all Compute Capabilities)
// Maximum block size: x-dimension: 1024 (for all Compute Capabilities)

const int N = 5000000;
const int blockSize = 1024;
const int gridSize = 128;

// CUDA kernel to initialize the random states
__global__ void initRandomStates(curandState *states, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// CUDA kernel function to calculate the price of European call options using Monte Carlo
__global__ void MonteCarloKernel(
    float* S, float K, float T, float r, float v, float *C, curandState *states)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int idx = threadId; idx < N; idx += stride) {
        if (idx < N) {
            curandState localState = states[idx];

            float S0 = S[idx];
            float drift = (r - 0.5f * v * v) * T;
            float vol = v * sqrtf(T);
            float gauss_bm = curand_normal(&localState);

            float S_T = S0 * expf(drift + vol * gauss_bm);
            float payoff = fmaxf(S_T - K, 0.0f);

            C[idx] = payoff * expf(-r * T);

            states[idx] = localState;  // Save the state back
        }
    }
}

int main()
{
    // Option parameters
    float S_values[5] = {90.0f, 95.0f, 100.0f, 105.0f, 110.0f};
    float K = 100.0f;
    float T_values[5] = {1.0f, 1.2f, 1.5f, 2.0f, 2.5f};
    float r_values[5] = {0.01f, 0.02f, 0.03f, 0.04f, 0.05f};
    float v = 0.3f;

   //   1. allocate and initialize CPU memory
    float *h_S = new float[N];
    float *h_C = new float[N];
    float h_Cs[5] = {0};

    //   2. allocate GPU memory
    float *d_S, *d_C;
    curandState *d_states;
    cudaError_t errCode_S = cudaMalloc(&d_S, N * sizeof(float));
    cudaError_t errCode_C = cudaMalloc(&d_C, N * sizeof(float));
    cudaError_t errCode_states = cudaMalloc(&d_states, N * sizeof(curandState));

    if (errCode_S != cudaSuccess) 
    {
        printf("cudaMalloc returned %d\n-> %s\n",
                static_cast<int>(errCode_S), cudaGetErrorString(errCode_S));
        exit(EXIT_FAILURE);
    }

    // curand - initialize random states using a seed
    initRandomStates<<<gridSize, blockSize>>>(d_states, 1234);
    cudaDeviceSynchronize();
    
    // Measure start time
    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 5; ++i) {
        std::fill(h_S, h_S + N, S_values[i]);

        //   3. copy data (inputs) to GPU memory (from CPU memory)
        cudaMemcpy(d_S, h_S, N * sizeof(float), cudaMemcpyHostToDevice);

        //   4. launch CUDA kernel; compute;
        //         generate results on GPU memory
        MonteCarloKernel<<<gridSize, blockSize>>>(
                    d_S, K, T_values[i], r_values[i], v, d_C, d_states);
        cudaDeviceSynchronize();

        //   5. copy data (results) to CPU memory (from GPU memory)
        cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Calculate the average option price
        // todo - vectorize this with accu,ulator
        float option_price = 0.0f;
        for (int j = 0; j < N; ++j) {
            option_price += h_C[j];
        }
        option_price /= N;

        h_Cs[i] = option_price;
    }

    // Measure end time
    auto t2 = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    std::chrono::duration<double, std::milli> elapsed = t2 - t1;
    cout << "Elapsed time: " << elapsed.count() << " ms" << endl;

    // Print out results
    for (int i = 0; i < 5; ++i) {
        cout << "Option " << (char)('a' + i) << ": " << h_Cs[i] << endl;
    }

    //   6. deallocate GPU memory
    cudaFree(d_S);
    cudaFree(d_C);
    cudaFree(d_states);

    //   7. deallocate CPU memory
    delete[] h_S;
    delete[] h_C;

    return 0;
}


#include <iostream>
#include <cmath>
#include <chrono> // For measuring execution time
#include <random>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

// Useful constants
const int N = 5000000;

// Function to calculate the price of European call options using Monte Carlo
void MonteCarloKernel(
    const vector<float>& S, float K, float T, float r, float v, vector<float>& C, mt19937& generator)
{
    normal_distribution<float> distribution(0.0f, 1.0f);

    for (int idx = 0; idx < N; ++idx) {
        float S0 = S[idx];
        float drift = (r - 0.5f * v * v) * T;
        float vol = v * sqrtf(T);
        float gauss_bm = distribution(generator);

        float S_T = S0 * expf(drift + vol * gauss_bm);
        float payoff = max(S_T - K, 0.0f);

        C[idx] = payoff * expf(-r * T);
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

    // Allocate and initialize CPU memory
    vector<float> h_S(N);
    vector<float> h_C(N);
    float h_Cs[5] = {0};

    // Random number generator
    random_device rd;
    mt19937 generator(rd());

    // Measure start time
    auto t1 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 5; ++i) {
        fill(h_S.begin(), h_S.end(), S_values[i]);

        // Compute option prices
        MonteCarloKernel(h_S, K, T_values[i], r_values[i], v, h_C, generator);

        // Calculate the average option price
        float option_price = accumulate(h_C.begin(), h_C.end(), 0.0f) / N;
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

    return 0;
}

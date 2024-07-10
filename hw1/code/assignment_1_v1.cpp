#include <iostream>
#include <cmath>
#include <chrono> // For measuring execution time

// Write a function to price European Call, Put options, and Greeks, using Black Scholes formula.
// Measure time taken to price 1 million (distinct) options. 
// Use random data to initialize parameters for each option.

// Useful constants
const float invsqrt2 = 0.7071068f;

// Function to generate random float between low and hi
float random_data(float low, float hi) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return low + r * (hi - low);
}

// Function to calculate the normal cumulative distribution function
inline float cdf_normal(const float x)
{
    return 0.5 + 0.5 * erff(x * invsqrt2);
}

// Function to calculate the normal probability density function
float pdf_normal(const float x) {
    const float c = 0.3989422804014337; // 1 / sqrt(2 * pi)
    return c * expf(-0.5 * x * x);
}

// Function to calculate option prices and Greeks
void calculate_option(float* K, float* T, float* S0, float* sigma, float* r, 
                      float* C_price, float* P_price, 
                      float* C_delta, float* P_delta,
                      float* C_gamma, float* P_gamma, 
                      float* C_vega, float* P_vega,
                      float* C_rho, float* P_rho, 
                      float* C_theta, float* P_theta, 
                      int NumOptions)
{
    for (int i = 0; i < NumOptions; ++i) {
        // Preliminary calculations
        float erT = expf(-r[i] * T[i]);
        float sqrtT = sqrtf(T[i]);
        
        float d1 = (logf(S0[i] / K[i]) + (r[i] + 0.5 * sigma[i] * sigma[i]) * T[i]) / (sigma[i] * sqrtf(T[i]));
        float d2 = d1 - sigma[i] * sqrtf(T[i]);

        float nd1 = cdf_normal(d1);
        float nd2 = cdf_normal(d2);
        float pdf_d1 = pdf_normal(d1);

        C_price[i] = S0[i] * nd1 - K[i] * erT * nd2;
        P_price[i] = K[i] * erT * (1 - nd2) - S0[i] * (1 - nd1);

        C_delta[i] = nd1;
        P_delta[i] = nd1 - 1;

        C_gamma[i] = pdf_d1 / (S0[i] * sigma[i] * sqrtT);
        P_gamma[i] = C_gamma[i];

        C_vega[i] = S0[i] * pdf_d1 * sqrtT;
        P_vega[i] = C_vega[i];

        C_rho[i] = K[i] * T[i] * erT * nd2;
        P_rho[i] = -K[i] * T[i] * erT * (1 - nd2);

        C_theta[i] = -S0[i] * pdf_d1 * sigma[i] / (2 * sqrtT) - r[i] * K[i] * erT * nd2;
        P_theta[i] = -S0[i] * pdf_d1 * sigma[i] / (2 * sqrtT) + r[i] * K[i] * erT * (1 - nd2);
    }
}

int main() 
{
    static const int NumOptions = 1000000;

    // Allocate memory for parameters and results
    float* K = new float[NumOptions];
    float* T = new float[NumOptions];
    float* S0 = new float[NumOptions];
    float* sigma = new float[NumOptions];
    float* r = new float[NumOptions];
    float* C_price = new float[NumOptions];
    float* P_price = new float[NumOptions];
    float* C_delta = new float[NumOptions];
    float* P_delta = new float[NumOptions];
    float* C_gamma = new float[NumOptions];
    float* P_gamma = new float[NumOptions];
    float* C_vega = new float[NumOptions];
    float* P_vega = new float[NumOptions];
    float* C_rho = new float[NumOptions];
    float* P_rho = new float[NumOptions];
    float* C_theta = new float[NumOptions];
    float* P_theta = new float[NumOptions];

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0))); 

    // Generate random parameters
    for (int i = 0; i < NumOptions; ++i) {
        S0[i] = random_data(0.0f, 200.0f);       // Stock price
        K[i] = random_data(50.0f, 150.0f);       // Strike price
        T[i] = random_data(0.01f, 2.0f);         // Time to maturity
        sigma[i] = random_data(0.01f, 0.99f);    // Volatility 
        r[i] = random_data(0.01f, 0.9f);         // Risk-free rate

        C_price[i] = 0.0f;
        P_price[i] = 0.0f;
        C_delta[i] = 0.0f;
        P_delta[i] = 0.0f;
        C_gamma[i] = 0.0f;
        P_gamma[i] = 0.0f;
        C_vega[i] = 0.0f;
        P_vega[i] = 0.0f;
        C_rho[i] = 0.0f;
        P_rho[i] = 0.0f;
        C_theta[i] = 0.0f;
        P_theta[i] = 0.0f;
    }

    using namespace std::chrono;
    auto t1 = high_resolution_clock::now();

    // Calculate option prices and Greeks
    calculate_option(K, T, S0, sigma, r, 
                     C_price, P_price, 
                     C_delta, P_delta, 
                     C_gamma, P_gamma, 
                     C_vega, P_vega, 
                     C_rho, P_rho, 
                     C_theta, P_theta, 
                     NumOptions);

    auto t2 = high_resolution_clock::now();

    std::cout << "Elapsed time: " <<
        duration_cast<milliseconds>(t2 - t1).count() << " ms" << "\n";

    // Output results for the first few options to verify
    for (int i = 0; i < 10; ++i) {
        std::cout << "Option " << i+1 << ":\n";
        std::cout << "  Call Price: " << C_price[i] << ", Put Price: " << P_price[i] << "\n";
        std::cout << "  Call Delta: " << C_delta[i] << ", Put Delta: " << P_delta[i] << "\n";
        std::cout << "  Call Gamma: " << C_gamma[i] << ", Put Gamma: " << P_gamma[i] << "\n";
        std::cout << "  Call Vega: " << C_vega[i] << ", Put Vega: " << P_vega[i] << "\n";
        std::cout << "  Call Rho: " << C_rho[i] << ", Put Rho: " << P_rho[i] << "\n";
        std::cout << "  Call Theta: " << C_theta[i] << ", Put Theta: " << P_theta[i] << "\n";
    }

    // Clean up
    delete[] K;
    delete[] T;
    delete[] S0;
    delete[] sigma;
    delete[] r;
    delete[] C_price;
    delete[] P_price;
    delete[] C_delta;
    delete[] P_delta;
    delete[] C_gamma;
    delete[] P_gamma;
    delete[] C_vega;
    delete[] P_vega;
    delete[] C_rho;
    delete[] P_rho;
    delete[] C_theta;
    delete[] P_theta;

    return 0;
}

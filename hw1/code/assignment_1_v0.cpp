#include <iostream>
#include <cmath>
#include <chrono> // For measuring execution time

using namespace std::chrono;

// Write a function to price European Call, Put options, and Greeks, using Black Scholes formula.
// Measure time taken to price 1 million (distinct) options. 
// Use random data to initialize parameters for each option.

// Function to generate random float between low and hi
float random_data(float low, float hi) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return low + r * (hi - low);
}

// Function to calculate the normal cumulative distribution function
double norm_cdf(double x) {
    return 0.5 * std::erfc(-x * M_SQRT1_2);
}

// Function to calculate the normal probability density function
double norm_pdf(double x) {
    return (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x * x);
}

// double call_price(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
//     double d2 = d1 - sigma * std::sqrt(T);

//     return S0 * norm_cdf(d1) - K * std::exp(-r * T) * norm_cdf(d2);
// }

// double put_price(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
//     double d2 = d1 - sigma * std::sqrt(T);

//     return K * std::exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1);
// }

// double call_delta(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));

//     return norm_cdf(d1);
// }

// double put_delta(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));

//     return norm_cdf(d1) - 1;
// }

// double call_gamma(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));

//     return norm_pdf(d1) / (S0 * sigma * std::sqrt(T));
// }

// double put_gamma(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));

//     return norm_pdf(d1) / (S0 * sigma * std::sqrt(T));
// }

// double call_vega(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));

//     return S0 * norm_pdf(d1) * std::sqrt(T);
// }

// double put_vega(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));

//     return S0 * norm_pdf(d1) * std::sqrt(T);
// }

// double call_rho(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
//     double d2 = d1 - sigma * std::sqrt(T);

//     return K * T * std::exp(-r * T) * norm_cdf(d2);
// }

// double put_rho(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
//     double d2 = d1 - sigma * std::sqrt(T);

//     return -K * T * std::exp(-r * T) * norm_cdf(-d2);
// }

// double call_theta(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
//     double d2 = d1 - sigma * std::sqrt(T);

//     return -S0 * norm_pdf(d1) * sigma / (2 * std::sqrt(T)) - r * K * std::exp(-r * T) * norm_cdf(d2);
// }

// double put_theta(double S0, double K, double r, double sigma, double T) {
//     double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
//     double d2 = d1 - sigma * std::sqrt(T);

//     return -S0 * norm_pdf(d1) * sigma / (2 * std::sqrt(T)) + r * K * std::exp(-r * T) * norm_cdf(-d2);
// }

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
        float erT = std::exp(-r[i] * T[i]);
        float sqrtT = std::sqrt(T[i]);
        
        float d1 = (std::log(S0[i] / K[i]) + (r[i] + 0.5 * sigma[i] * sigma[i]) * T[i]) / (sigma[i] * std::sqrt(T[i]));
        float d2 = d1 - sigma[i] * std::sqrt(T[i]);

        float nd1 = norm_cdf(d1);
        float nd2 = norm_cdf(d2);
        float pdf_d1 = norm_pdf(d1);

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
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    const int NumOptions = 1000000;

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
        S0[i] = random_data(50.0, 150.0);    // Stock price between 50 and 150
        K[i] = random_data(50.0, 150.0);     // Strike price between 50 and 150
        T[i] = random_data(0.1, 2.0);        // Time to maturity between 0.1 and 2 years
        sigma[i] = random_data(0.1, 0.5);    // Volatility between 0.1 and 0.5
        r[i] = random_data(0.01, 0.1);       // Risk-free rate between 0.01 and 0.1
    }
    
    // Calculate option prices and Greeks
    calculate_option(K, T, S0, sigma, r, 
                     C_price, P_price, 
                     C_delta, P_delta, 
                     C_gamma, P_gamma, 
                     C_vega, P_vega, 
                     C_rho, P_rho, 
                     C_theta, P_theta, 
                     NumOptions);

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

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    std::cout << "Elapsed time: " <<
        duration_cast<milliseconds>(t2 - t1).count() << " ms" << "\n";
    
    return 0;
}

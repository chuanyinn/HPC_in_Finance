#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono> // For measuring execution time
#include <random>
#include <boost/math/distributions/normal.hpp>

using namespace std::chrono;
using matrix = std::vector<std::vector<float>>;

// Write a function to price European Call options using Black Scholes formula.
// Measure time taken to price 1 million options.
// Measure the execution time.

// Function to calculate the option price
double black_scholes_call_price(double S0, double K, double r, double sigma, double T) {
    double d1 = ((std::log(S0 / K) + r * T) / (sigma * std::sqrt(T)) + 0.5 * sigma * std::sqrt(T));
    double d2 = ((std::log(S0 / K) + r * T) / (sigma * std::sqrt(T)) - 0.5 * sigma * std::sqrt(T));
    
    boost::math::normal_distribution<> normal_dist(0.0, 1.0);
    double call_price = S0 * boost::math::cdf(normal_dist, d1) - K * std::exp(-r * T) * boost::math::cdf(normal_dist, d2);

    return call_price;
}

int main() 
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0))); 

    // Initial stock price
    double S0 = 100.0;
    // Strike price
    double K = 100.0;
    // Risk-free interest rate
    double r = 0.05;
    // Volatility of the underlying stock
    double sigma = 0.2;
    // Time to expiration in years
    double T = 1.0;
    // Trials 
    int M = 1000000;

    for (int i = 0; i < M; ++i) {
        // Price option using Black Scholes formula
        double option_price = black_scholes_call_price(S0, K, r, sigma, T);
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    std::cout << "Elapsed time: " <<
        duration_cast<milliseconds>(t2 - t1).count() << " ms";
    
    return 0;
}



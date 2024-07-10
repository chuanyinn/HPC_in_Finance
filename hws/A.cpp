#include <iostream>
#include <cmath>
#include <algorithm>
#include <boost/math/distributions/normal.hpp>
#include <chrono> // For measuring execution time

using namespace std::chrono;

// Write a function to price European Call options using Black Scholes formula.
// Measure time taken to price 1 million (distinct) options. 
// Use random data to initialize parameters for each option.

// Function to generate random float between low and hi
float random_data(float low, float hi) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return low + r * (hi - low);
}

class OptionPricing {
    public:
        double S, K, r, sigma, T;
        OptionPricing(double S, double K, double r, double sigma, double T);

        double get_price();
};

// Constructor definition
OptionPricing::OptionPricing(double S, double K, double r, double sigma, double T) 
    : S(S), K(K), r(r), sigma(sigma), T(T) {}

// Function to calculate the option price
double OptionPricing::get_price() {
    double d1 = ((std::log(S / K) + r * T) / (sigma * std::sqrt(T)) + 0.5 * sigma * std::sqrt(T));
    double d2 = ((std::log(S / K) + r * T) / (sigma * std::sqrt(T)) - 0.5 * sigma * std::sqrt(T));
    
    boost::math::normal_distribution<> normal_dist(0.0, 1.0);
    double call_price = S * boost::math::cdf(normal_dist, d1) - K * std::exp(-r * T) * boost::math::cdf(normal_dist, d2);

    return call_price;
}

int main() 
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0))); 

    double price = 0.0;
    for (int i = 0; i < 1000000; ++i) {
        // Generate random parameters for the option
        double S = random_data(50.0, 150.0);    // Stock price between 50 and 150
        double K = random_data(50.0, 150.0);    // Strike price between 50 and 150
        double r = random_data(0.01, 0.1);      // Risk-free rate between 0.01 and 0.1
        double sigma = random_data(0.1, 0.5);   // Volatility between 0.1 and 0.5
        double T = random_data(0.5, 2.0);       // Time to maturity between 0.5 and 2 years

        OptionPricing option(S, K, r, sigma, T);
        price = option.get_price(); // Perform the calculation
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    std::cout << "Elapsed time: " <<
        duration_cast<milliseconds>(t2 - t1).count() << " ms";
    
    return 0;
}



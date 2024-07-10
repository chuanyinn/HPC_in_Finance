#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono> // For measuring execution time
#include <random>
#include <boost/math/distributions/normal.hpp>

using namespace std::chrono;
using matrix = std::vector<std::vector<float>>;

// Write a function to price European Call options using Monte Carlo.
// Measure time taken to price 1 million options.
// Measure the execution time.

float generate_standard_normal() {
    // Seed with a real random value, if available
    static std::random_device rd;
    
    // Initialize a random number generator
    static std::mt19937 gen(rd());
    
    // Define a standard normal distribution with mean 0 and standard deviation 1
    std::normal_distribution<float> dis(0.0, 1.0);
    
    // Generate and return a random number from the standard normal distribution
    return dis(gen);
}

double monte_carlo_option_pricing(double S0, double K, double r, double sigma, double T, int M) {
    std::vector<double> payoffs;
    payoffs.reserve(M);

    for (int i = 0; i < M; ++i) {
        // Generate z ~ N(0, 1)
        double z = generate_standard_normal();

        // Calculate the stock price
        double S = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * z);

        // Calculate the payoff
        double payoff = std::exp(-r * T) * std::max(S - K, 0.0);

        // Store the payoff
        payoffs.push_back(payoff);
    }

    // Calculate the mean option payoffs
    double sum_payoffs = std::accumulate(payoffs.begin(), payoffs.end(), 0.0);
    double option_price = sum_payoffs / static_cast<double>(M);

    return option_price;
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
    // Number of Monte Carlo trials
    int M = 1000000;

    // Calculate the option price using Monte Carlo simulation
    double option_price = monte_carlo_option_pricing(S0, K, r, sigma, T, M);

    std::cout << "Option price: " << option_price << std::endl;

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    std::cout << "Elapsed time: " <<
        duration_cast<milliseconds>(t2 - t1).count() << " ms";
    
    return 0;
}



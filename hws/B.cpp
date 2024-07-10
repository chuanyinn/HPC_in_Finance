#include <iostream>
#include <cmath>
#include <algorithm>
#include <boost/math/distributions/normal.hpp>
#include <chrono> // For measuring execution time

using namespace std::chrono;
using matrix = std::vector<std::vector<float>>;

// Use the function you wrote to multiply two 1000x1000 matrices.
// Measure the execution time.

// Use std::vector:

float random_data(float low, float hi) {
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    return low + r * (hi - low);
}


int main() 
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0))); 

    const int rows = 1000, columns = 1000;

    // Create a 1000x1000 matrix
    matrix m1, m2;
    m1.resize(rows);
    m2.resize(rows);

    for (int i = 0; i < rows; i++) {
        m1[i].resize(columns);
        m2[i].resize(columns);
    }

    // Fill the matrix with random data
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            m1[i][j] = random_data(0, 10);
            m2[i][j] = random_data(0, 10);
        }
    }

    // Multiply the matrices
    matrix result;
    result.resize(rows);

    for (int i = 0; i < rows; i++) {
        result[i].resize(columns);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            float res = 0;
            for (int k = 0; k < columns; k++) {
                res += m1[i][k] * m2[k][j];
            }
            result[i][j] = res;
        }
    }
    
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    std::cout << "Elapsed time: " <<
        duration_cast<milliseconds>(t2 - t1).count() << " ms";
    
    return 0;
}



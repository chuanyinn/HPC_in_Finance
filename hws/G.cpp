#include <iostream>
#include <cmath>
#include <algorithm>
#include <chrono> // For measuring execution time
#include <random>
#include <boost/math/distributions/normal.hpp>
#include <vector>
#include <complex>

using namespace std::chrono;
using matrix = std::vector<std::vector<float>>;

// Write a function to generate a normal Julia set
// Measure the execution time.

std::vector<std::vector<int>> generate_julia_set(int width, int height, 
                                                 double R, std::complex<double> c,
                                                 int max_iter) {
    std::vector<std::vector<int>> image(height, std::vector<int>(width));
    double scale_real = 2.0 * R / width;
    double scale_imag = 2.0 * R / height;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double zx = x * scale_real - R;
            double zy = y * scale_imag - R;

            std::complex<double> z(zx, zy);
            int iter = 0;
            while (std::norm(z) < R * R && iter < max_iter) {
                z = z * z + c;
                ++iter;
            }

            if (iter == max_iter) {
                image[y][x] = 0;
            } 
            else {
                image[y][x] = iter;
            }

        }
    }
    return image;
}

int main() 
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    // Seed the random number generator
    srand(static_cast<unsigned int>(time(0))); 

    double width = 1000;
    double height = 1000;
    double R = 2.0; // Escape radius
    std::complex<double> c(-0.7, 0.27015); // Complex parameter for Julia set
    int max_iter = 1000;

    std::vector<std::vector<int>> image = generate_julia_set(width, height, R, c, max_iter);

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    std::cout << "Elapsed time: " <<
        duration_cast<milliseconds>(t2 - t1).count() << " ms";
    
    return 0;
}



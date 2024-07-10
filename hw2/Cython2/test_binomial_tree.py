import time
import assignment2
import numpy as np


if __name__ == "__main__":
    K = 100
    T = 1
    r = 0.03
    sigma = 0.3
    N = 1000

    start_time = time.time()

    print('S = 90, r = 0.03, v = 0.3, T = 1, N = 1000 call price:', 
        assignment2.jarrow_rudd_binomial_tree(90, K, T, r, sigma, N, option_type='call'))
    print('S = 95, r = 0.03, v = 0.3, T = 1, N = 1000 call price:', 
        assignment2.jarrow_rudd_binomial_tree(95, K, T, r, sigma, N, option_type='call'))
    print('S = 100, r = 0.03, v = 0.3, T = 1, N = 1000 call price:',
        assignment2.jarrow_rudd_binomial_tree(100, K, T, r, sigma, N, option_type='call'))
    print('S = 105, r = 0.03, v = 0.3, T = 1, N = 1000 call price:', 
        assignment2.jarrow_rudd_binomial_tree(105, K, T, r, sigma, N, option_type='call'))
    print('S = 110, r = 0.03, v = 0.3, T = 1, N = 1000 call price:', 
        assignment2.jarrow_rudd_binomial_tree(110, K, T, r, sigma, N, option_type='call'))

    print('S = 90, r = 0.03, v = 0.3, T = 1, N = 1000 put price:',
        assignment2.jarrow_rudd_binomial_tree(90, K, T, r, sigma, N, option_type='put'))
    print('S = 95, r = 0.03, v = 0.3, T = 1, N = 1000 put price:',
        assignment2.jarrow_rudd_binomial_tree(95, K, T, r, sigma, N, option_type='put'))
    print('S = 100, r = 0.03, v = 0.3, T = 1, N = 1000 put price:',
        assignment2.jarrow_rudd_binomial_tree(100, K, T, r, sigma, N, option_type='put'))
    print('S = 105, r = 0.03, v = 0.3, T = 1, N = 1000 put price:',
        assignment2.jarrow_rudd_binomial_tree(105, K, T, r, sigma, N, option_type='put'))
    print('S = 110, r = 0.03, v = 0.3, T = 1, N = 1000 put price:',
        assignment2.jarrow_rudd_binomial_tree(110, K, T, r, sigma, N, option_type='put'))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.5f} seconds")
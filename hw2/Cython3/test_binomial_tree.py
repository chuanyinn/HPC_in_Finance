import time
import assignment2
import numpy as np


if __name__ == "__main__":
    S = [90, 95, 100, 105, 110]
    K = [100] * len(S)
    T = [1] * len(S)
    r = [0.03] * len(S)
    sigma = [0.3] * len(S)
    N = 1000

    start_time = time.time()

    for i in range(len(S)):
        print(f'S = {S[i]}, r = {r[i]}, v = {sigma[i]}, T = {T[i]}, N = {N} call price: ', 
            assignment2.jarrow_rudd_binomial_tree(S[i], K[i], T[i], r[i], sigma[i], N, option_type='call'))
        print(f'S = {S[i]}, r = {r[i]}, v = {sigma[i]}, T = {T[i]}, N = {N} put price: ', 
            assignment2.jarrow_rudd_binomial_tree(S[i], K[i], T[i], r[i], sigma[i], N, option_type='put'))

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.5f} seconds")
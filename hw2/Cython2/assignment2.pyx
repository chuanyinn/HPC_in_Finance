from cython.parallel import prange
import numpy as np
cimport numpy as np
from libc.math cimport exp

def populate_asset_prices(float S0, float u, float d, unsigned int N):
    """
    S0 : float : Initial stock price
    u : float : Up factor
    d : float : Down factor
    N : int : Number of time steps
    
    Populate the asset prices at each node in the binomial tree.
    """
    cdef unsigned int i, j
    cdef np.ndarray[np.float64_t, ndim=2] asset_prices = np.zeros((N + 1, N + 1), dtype=np.float64)

    asset_prices[0, 0] = S0
    for i in range(1, N + 1):
        for j in range(i + 1):
            asset_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)

    return asset_prices

def initialize_option_values(np.ndarray[np.float64_t, ndim=2] asset_prices, 
                             float K, unsigned int N, option_type='call'):
    """
    asset_prices : np.array : Asset prices at each node in the binomial tree
    K : float : Strike price
    option_type : str : 'call' for call option, 'put' for put option
    
    Initialize the option values at maturity.
    """
    cdef unsigned int j
    cdef np.ndarray[np.float64_t, ndim=2] option_values = np.zeros((N + 1, N + 1), dtype=np.float64)

    for j in range(N + 1):
        if option_type == 'call':
            option_values[j, N] = (asset_prices[j, N] - K) if (asset_prices[j, N] - K) > 0 else 0
        else:
            option_values[j, N] = (K - asset_prices[j, N]) if (K - asset_prices[j, N]) > 0 else 0
            
    return option_values

def backtrack_option_values(np.ndarray[np.float64_t, ndim=2] option_values, 
                            float r, float dt, float p, unsigned int N):
    """
    option_values : np.array : Option values at each node in the binomial tree
    r : float : Risk-free interest rate (annual)
    dt : float : Time step
    p : float : Probability of up move
    N : int : Number of time steps
    
    Backtrack through the tree to calculate the option price.
    """
    cdef unsigned int i, j
    cdef float exp_factor = np.exp(-r * dt)

    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j, i] = exp_factor * (p * option_values[j, i + 1] 
                                             + (1 - p) * option_values[j + 1, i + 1])
            
    return option_values[0, 0]

def jarrow_rudd_binomial_tree(float S0, float K, float T, float r, float sigma, int N, option_type='call'):
    """
    S0 : float : Initial stock price
    K : float : Strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate (annual)
    sigma : float : Volatility (annual)
    N : int : Number of time steps
    option_type : str : 'call' for call option, 'put' for put option
    
    Returns the price of the option
    """
    cdef double dt = T / N
    cdef double u = exp((r - (sigma ** 2) / 2) * dt + sigma * (dt ** 0.5))
    cdef double d = exp((r - (sigma ** 2) / 2) * dt - sigma * (dt ** 0.5))
    cdef double p = 0.5

    cdef np.ndarray[np.float64_t, ndim=2] asset_prices = populate_asset_prices(S0, u, d, N)
    cdef np.ndarray[np.float64_t, ndim=2] option_values = initialize_option_values(asset_prices, K, N, option_type)
    cdef double option_price = backtrack_option_values(option_values, r, dt, p, N)

    return option_price
import numpy as np
import time

# @profile
def populate_asset_prices(S0, u, d, N):
    """
    S0 : float : Initial stock price
    u : float : Up factor
    d : float : Down factor
    N : int : Number of time steps
    
    Populate the asset prices at each node in the binomial tree.
    """
    asset_prices = np.zeros((N + 1, N + 1))
    asset_prices[0, 0] = S0
    for i in range(1, N + 1):
        for j in range(i + 1):
            asset_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)
            
    return asset_prices

# @profile
def initialize_option_values(asset_prices, K, N, option_type='call'):
    """
    asset_prices : np.array : Asset prices at each node in the binomial tree
    K : float : Strike price
    option_type : str : 'call' for call option, 'put' for put option
    
    Initialize the option values at maturity.
    """
    option_values = np.zeros((N + 1, N + 1))
    for j in range(N + 1):
        if option_type == 'call':
            option_values[j, N] = max(0, asset_prices[j, N] - K)
        else:
            option_values[j, N] = max(0, K - asset_prices[j, N])

    return option_values

# @profile
def backtrack_option_values(option_values, r, dt, p, N):
    """
    option_values : np.array : Option values at each node in the binomial tree
    r : float : Risk-free interest rate (annual)
    dt : float : Time step
    p : float : Probability of up move
    N : int : Number of time steps
    
    Backtrack through the tree to calculate the option price.
    """
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j, i] = np.exp(-r * dt) * (p * option_values[j, i + 1] 
                                                     + (1 - p) * option_values[j + 1, i + 1])
            
    return option_values[0, 0]

# @profile
def jarrow_rudd_binomial_tree(S0, K, T, r, sigma, N, option_type='call'):
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
    dt = T / N
    u = np.exp((r - (sigma ** 2) / 2) * dt + sigma * np.sqrt(dt))
    d = np.exp((r - (sigma ** 2) / 2) * dt - sigma * np.sqrt(dt))
    p = 0.5
    
    asset_prices = populate_asset_prices(S0, u, d, N)
    option_values = initialize_option_values(asset_prices, K, N, option_type)
    option_price = backtrack_option_values(option_values, r, dt, p, N)

    return option_price

def jarrow_rudd_binomial_tree_orig(S0, K, T, r, sigma, N, option_type='call'):
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
    dt = T / N
    u = np.exp((r - (sigma ** 2) / 2) * dt + sigma * np.sqrt(dt))
    d = np.exp((r - (sigma ** 2) / 2) * dt - sigma * np.sqrt(dt))
    p = 0.5
    
    # Initialize asset prices at maturity
    asset_prices = np.zeros((N + 1, N + 1))
    option_values = np.zeros((N + 1, N + 1))
    
    asset_prices[0, 0] = S0
    for i in range(1, N + 1):
        for j in range(i + 1):
            asset_prices[j, i] = S0 * (u ** (i - j)) * (d ** j)
    
    # Initialize option values at maturity
    for j in range(N + 1):
        if option_type == 'call':
            option_values[j, N] = max(0, asset_prices[j, N] - K)
        else:
            option_values[j, N] = max(0, K - asset_prices[j, N])
    
    # Step back through the tree
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option_values[j, i] = np.exp(-r * dt) * (p * option_values[j, i + 1] 
                                                     + (1 - p) * option_values[j + 1, i + 1])
    
    return option_values[0, 0]

K = 100
T = 1
r = 0.03
sigma = 0.3
N = 1000

start_time = time.time()

print('S = 90, r = 0.03, v = 0.3, T = 1, N = 1000 call price:', 
      jarrow_rudd_binomial_tree(90, K, T, r, sigma, N, option_type='call'))
print('S = 95, r = 0.03, v = 0.3, T = 1, N = 1000 call price:', 
      jarrow_rudd_binomial_tree(95, K, T, r, sigma, N, option_type='call'))
print('S = 100, r = 0.03, v = 0.3, T = 1, N = 1000 call price:',
      jarrow_rudd_binomial_tree(100, K, T, r, sigma, N, option_type='call'))
print('S = 105, r = 0.03, v = 0.3, T = 1, N = 1000 call price:', 
      jarrow_rudd_binomial_tree(105, K, T, r, sigma, N, option_type='call'))
print('S = 110, r = 0.03, v = 0.3, T = 1, N = 1000 call price:', 
      jarrow_rudd_binomial_tree(110, K, T, r, sigma, N, option_type='call'))

print('S = 90, r = 0.03, v = 0.3, T = 1, N = 1000 put price:',
      jarrow_rudd_binomial_tree(90, K, T, r, sigma, N, option_type='put'))
print('S = 95, r = 0.03, v = 0.3, T = 1, N = 1000 put price:',
      jarrow_rudd_binomial_tree(95, K, T, r, sigma, N, option_type='put'))
print('S = 100, r = 0.03, v = 0.3, T = 1, N = 1000 put price:',
      jarrow_rudd_binomial_tree(100, K, T, r, sigma, N, option_type='put'))
print('S = 105, r = 0.03, v = 0.3, T = 1, N = 1000 put price:',
      jarrow_rudd_binomial_tree(105, K, T, r, sigma, N, option_type='put'))
print('S = 110, r = 0.03, v = 0.3, T = 1, N = 1000 put price:',
      jarrow_rudd_binomial_tree(110, K, T, r, sigma, N, option_type='put'))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.5f} seconds")
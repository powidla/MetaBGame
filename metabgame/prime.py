'''
Simple function to generate seeds. 
'''
from jax import random
from typing import Any, List

def prime_seeds(n_keys: int) -> List[int]:
    return [i**2 + i + 41 for i in range(n_keys)]

def gauss_prng_keys(n_keys: int) -> List[random.PRNGKey]:
    seeds = prime_seeds(n_keys)
    return [random.PRNGKey(seed) for seed in seeds]
  

"""
Flax CNN Script
"""

import jax
import jax.numpy as jnp                # JAX NumPy

from flax import linen as nn           # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state

import numpy as np                     # Ordinary NumPy
import optax                           # Optimizers
import tensorflow_datasets as tfds     # TFDS for MNIST

# Suppress warning and info messages from jax
import os  
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main() -> None:
    print("Jax CNN Script...")

if __name__ == "__main__":
    main()



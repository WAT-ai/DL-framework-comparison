import os
import time
import logging
import jax
import mlp

# Suppress warning and info messages from jax 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def main():

    print(f"\nEnvironment Config: ")
    print(f"TF_CPP_MIN_LOG_LEVEL = {os.environ['TF_CPP_MIN_LOG_LEVEL']}")

    layers = [784, 100, 10]
    model = mlp.MLPModel(layers, learning_rate=0.01, log_level=logging.INFO)

if __name__ == "__main__":
    main()
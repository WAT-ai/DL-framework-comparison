import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp
import logging

class MLPModel:

    def __init__(self, layer_sizes, learning_rate=0.01, log_level=logging.DEBUG):

        # Array of each layer size
        self.layer_sizes = layer_sizes

        # Also refered to as the step size
        # Scaling apply to estimated weight error during backpropagation
        self.learning_rate = learning_rate

        # Configure logging 
        self.log_level =  log_level
        self.logger = logging.getLogger("mlp-logger")
        # logging.basicConfig(level=self.log_level)

        # Initialize all layers with random values
        self.__init_layers

        self.logger.debug("Layers Initialized:")
        print(f"Layers Initialized: {self.layer_sizes}")

    def __init_layer_params(self, m, n, key, scale=1e-2):
        """
        A helper function to randomly initialize weights and biases
        for a dense neural network layer.
        """
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

    def __init_layers(self, key=random.PRNGKey(0) ):
        """
        Initialize all layers for a fully-connected neural network.
        """
        keys = random.split(key, len(sizes))
        sizes = self.layer_sizes
        self.params =  [self.__init_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

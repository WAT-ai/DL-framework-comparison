import time
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

# For benchmarking and logging
import jax.profiler

# Start server to connect to tensorboard
jax.profiler.start_trace("/tmp/tensorboard")

time.sleep(2)

print("Jax Frameworks...")
key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)

# Run the operations to be profiled
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (5000, 5000))
y = x @ x
y.block_until_ready()
print(y)

jax.profiler.stop_trace()

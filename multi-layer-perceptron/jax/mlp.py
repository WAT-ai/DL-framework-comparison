import time
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp

class MLPModel:

    def __init__(self, layer_sizes, num_epochs=1, batch_size=128, n_targets=10, learning_rate=0.01):

        # Array of each layer size
        self.layer_sizes = layer_sizes

        # Also refered to as the step size
        # Scaling apply to estimated weight error during backpropagation
        self.learning_rate = learning_rate

        # Configure parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.n_targets = n_targets

        # Initialize all layers with random values
        self.params = self.__init_layers()

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
        sizes = self.layer_sizes
        keys = random.split(key, len(sizes))

        return [self.__init_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

def relu(x):
    """
    Rectified Linear Operation.
    """
    return jnp.maximum(0, x)

def one_hot_encoder(x, k, dtype=jnp.float32):
    """
    Create a one-hot encoding of x of size k.
    """
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def predict(params, input):
    """
    Run a single prediction on the input. 

    Parameters: 
        input (array): A n-dimensional array, typically a 1D or 2D array. For images for instance this is a 2D array. 
    """

    activations = input
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)
    
    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    
    return logits - logsumexp(logits)

# Temp, move this
batched_predict = vmap(predict, in_axes=(None, 0))

def loss(params, inputs, targets):
    """
    Compute loss
    """
    preds = batched_predict(params, inputs)
    loss = -jnp.mean(preds * targets)

    return loss

@jit
def update(params, x, y, learning_rate):
    """
    Perform backpropagation
    """
    grads = grad(loss)(params, x, y)

    return [(w - learning_rate * dw, b - learning_rate * db) for (w, b), (dw, db) in zip(params, grads)]

def accuracy(params, inputs, targets):
    """
    Test accuracy 
    """

    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, inputs), axis=1)
    
    return jnp.mean(predicted_class == target_class)

def train(params, ds, train_images, train_labels, test_images, test_labels, num_labels, num_pixels, learning_rate, epochs=10, debug=False):
    print("\nTraining Model...")

    training_metrics = {}
    ts = time.time()

    epoch_times = []

    for epoch in tqdm(range(epochs)):

        start_time = time.time()

        # Main training loop
        for x, y in ds:
            x = jnp.reshape(x, (len(x), num_pixels))
            y = one_hot_encoder(y, num_labels)
            params = update(params, x, y, learning_rate)
        
        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        train_acc = accuracy(params, train_images, train_labels)
        test_acc = accuracy(params, test_images, test_labels)
        
        if debug: 
            print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
            print("Training set accuracy {}".format(train_acc))
            print("Test set accuracy {}".format(test_acc))
    
    te = time.time()
    training_metrics["total_training_time"] = te - ts

    epoch_times = np.array(epoch_times)
    training_metrics["average_epoch_training_time"] = epoch_times.mean()

    # Compute final accuracy and loss
    final_acc = accuracy(params, test_images, test_labels)
    final_loss = loss(params, test_images, test_labels)
    training_metrics["final_training_loss"] = float(final_loss)
    training_metrics["final_evaluation_accuracy"] = float(final_acc)

    return training_metrics

def compute_inference(params, ds):

    inf_times = []

    for batch in tqdm(ds):
        ts = time.time()
        res = predict(params, batch[0])
        dt = time.time() - ts

        inf_times.append(dt)
        # print(f"prediction: {jnp.argmax(res)}, actual: {jnp.argmax(batch[1])}")

    inf_times = np.array(inf_times)
    mean_inf = inf_times.mean()
    print(f"Mean Inference Time: {mean_inf}s")

    return mean_inf

def compute_inference_batch(params, ds):
    print("\nComputing Batch Inference...")

    inf_times = []

    num_pixels = 784
    num_labels = 10
    for x, y in tqdm(ds):
        ts = time.time()
        x = jnp.reshape(x, (len(x), num_pixels))
        y = one_hot_encoder(y, num_labels)
        batched_predict(params, x)
        dt = time.time() - ts

        inf_times.append(dt)

    inf_times = np.array(inf_times)
    mean_inf = inf_times.mean()
    print(f"Mean Batch Inference Time: {mean_inf}s")

    return mean_inf
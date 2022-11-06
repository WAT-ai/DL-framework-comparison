import os
import json
import time
import optax
import numpy as np
import jax.numpy as jnp
import tensorflow_datasets as tfds
from tqdm import tqdm
from jax import grad, jit, vmap
from jax import random
from jax.scipy.special import logsumexp

# Suppress warning, info and error messages from jax, tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Specify directory for data 
DATA_DIR = './datasets'

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
    
    # Compute log softmax loss
    loss = logits - logsumexp(logits)

    return loss

# Temp, move this
batched_predict = vmap(predict, in_axes=(None, 0))

def loss(params, inputs, targets):
    """
    Computes negative log likehood loss. 

    Each prediction of batched_predict retuns the log softmax loss.
    Computing the mean and returning the mean is this form of loss. 
    """
    preds = batched_predict(params, inputs)
    loss = -jnp.mean(preds * targets)

    return loss

def update(params, x, y, learning_rate):
    """
    Perform backpropagation using stochastic gradient descent.
    """
    grads = grad(loss)(params, x, y)

    return [(w - learning_rate * dw, b - learning_rate * db) for (w, b), (dw, db) in zip(params, grads)]

def update_adam(params, x, y, opt_state, optimizer):
    """
    Perform backpropagation using adam optimizer.
    """
    
    grads = grad(loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state

def accuracy(params, inputs, targets):
    """
    Test accuracy 
    """

    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, inputs), axis=1)
    
    return jnp.mean(predicted_class == target_class)

def train(params, ds, train_images, train_labels, test_images, test_labels, num_labels, num_pixels, learning_rate, epochs=10, debug=False, use_optimizer='sgd'):
    """
    Either set use_optimizer to be adam ('adam') or standard stocastic gradient desent ('sgd'). 
    """
    print(f"\nTraining Model using ({use_optimizer})...")

    if use_optimizer == 'adam':

        # define the adam optimizer
        optimizer = optax.adam(learning_rate, b1=0.9, b2=0.999)

        # Initialize parameters of the model + optimizer.
        opt_state = optimizer.init(params)

    # metrics and tracking
    training_metrics = {}
    ts = time.time()
    epoch_times = []

    for epoch in tqdm(range(epochs)):

        start_time = time.time()

        # Main training loop
        for x, y in ds:
            x = jnp.reshape(x, (len(x), num_pixels))
            y = one_hot_encoder(y, num_labels)

            # default to stochastic gradient descent 
            if use_optimizer == 'adam':
                params, opt_state = update_adam(params, x, y, opt_state, optimizer)
            else:
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

def load_datasets():

    # Fetch full datasets for evaluation
    # tfds.load returns tf.Tensors (or tf.data.Datasets if batch_size != -1)
    # You can convert them to NumPy arrays (or iterables of NumPy arrays) with tfds.dataset_as_numpy
    mnist_data, info = tfds.load(name="mnist", batch_size=-1, data_dir=DATA_DIR, with_info=True)

    mnist_data = tfds.as_numpy(mnist_data)
    train_data, test_data = mnist_data['train'], mnist_data['test']
    num_labels = info.features['label'].num_classes
    h, w, c = info.features['image'].shape
    num_pixels = h * w * c

    # # Full train set
    train_images, train_labels = train_data['image'], train_data['label']
    train_images = jnp.reshape(train_images, (len(train_images), num_pixels))
    train_labels = one_hot_encoder(train_labels, num_labels)

    # # Full test set
    test_images, test_labels = test_data['image'], test_data['label']
    test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
    test_labels = one_hot_encoder(test_labels, num_labels)

    print('Train:', train_images.shape, train_labels.shape)
    print('Test:', test_images.shape, test_labels.shape)
    print('num_pixels:', num_pixels)
    print('num_labels:', num_labels)

    return train_images, train_labels, test_images, test_labels, num_labels, num_pixels

def get_train_batches(batch_size):
    
    # as_supervised=True gives us the (image, label) as a tuple instead of a dict
    ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=DATA_DIR)
    
    # You can build up an arbitrary tf.data input pipeline
    ds = ds.batch(batch_size).prefetch(1)
    
    # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
    return tfds.as_numpy(ds)

def create_batch_ds(ds, num_labels, num_pixels):

    dataset = []
    for x, y in ds:
        x = jnp.reshape(x, (len(x), num_pixels))
        y = one_hot_encoder(y, num_labels)

        for batch in zip(x,y):
            dataset.append(batch)

    dataset = np.array(dataset)
    print(f"dataset: {dataset.shape}")
    print(f"dataset[0]: {dataset[0].shape}")
    print(f"dataset[0][0]: {dataset[0][0].shape}")
    print(f"dataset[0][1]: {dataset[0][1].shape}")

    return dataset


def main():

    print(f"\nEnvironment Config: ")
    print(f"TF_CPP_MIN_LOG_LEVEL = {os.environ['TF_CPP_MIN_LOG_LEVEL']}")

    # Initialize model
    layers = [784, 100, 10]
    batch_size = 128
    epochs = 10
    n_targets = 10
    learning_rate = 0.001
    model = MLPModel(layers, learning_rate=learning_rate, batch_size=batch_size, num_epochs=epochs, n_targets=n_targets)

    # Load Data 
    train_images, train_labels, test_images, test_labels, num_labels, num_pixels = load_datasets()
    ds = get_train_batches(batch_size)

    # Train model
    metrics = train(model.params, ds, train_images, train_labels, test_images, test_labels, num_labels, num_pixels, learning_rate, epochs=epochs, use_optimizer='adam')

    # Compute batch inference time
    batch_inf = compute_inference_batch(model.params, ds)
    metrics["average_batch_inference_time"] = batch_inf * 1000 # Convert to ms

    # Add other info 
    metrics["model_name"] = "MLP"
    metrics["framework_name"] = "Jax"
    metrics["dataset"] = "MNIST Digits"
    metrics["task"] = "classification"

    # Export to JSON
    print(f"\nTraining Metrics: \n{metrics}")
    with open("milestone1-jax-mlp.json", "w") as outfile:
        json.dump(metrics, outfile)

    # Additional metrics - not included
    # Create batched dataset
    # bds = create_batch_ds(ds, num_labels, num_pixels)

    # Compute other metrics
    # mlp.compute_inference(model.params, bds)

if __name__ == "__main__":
    main()
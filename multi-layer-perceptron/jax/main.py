import os
import time
import logging
import jax
import mlp
import jax.numpy as jnp
import tensorflow_datasets as tfds

# Suppress warning, info and error messages from jax, tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Specify directory for data 
DATA_DIR = './datasets'

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
    train_labels = mlp.one_hot_encoder(train_labels, num_labels)

    # # Full test set
    test_images, test_labels = test_data['image'], test_data['label']
    test_images = jnp.reshape(test_images, (len(test_images), num_pixels))
    test_labels = mlp.one_hot_encoder(test_labels, num_labels)

    print('Train:', train_images.shape, train_labels.shape)
    print('Test:', test_images.shape, test_labels.shape)

    return train_images, train_labels, test_images, test_labels, num_labels, num_pixels

def get_train_batches(batch_size):
    
    # as_supervised=True gives us the (image, label) as a tuple instead of a dict
    ds = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=DATA_DIR)
    
    # You can build up an arbitrary tf.data input pipeline
    ds = ds.batch(batch_size).prefetch(1)
    
    # tfds.dataset_as_numpy converts the tf.data.Dataset into an iterable of NumPy arrays
    return tfds.as_numpy(ds)

def main():

    print(f"\nEnvironment Config: ")
    print(f"TF_CPP_MIN_LOG_LEVEL = {os.environ['TF_CPP_MIN_LOG_LEVEL']}")

    # Initialize model
    layers = [784, 100, 10]
    batch_size = 128
    epochs = 10
    n_targets = 10
    learning_rate = 0.01
    model = mlp.MLPModel(layers, learning_rate=learning_rate, batch_size=batch_size, num_epochs=epochs, n_targets=n_targets)

    # Load Data 
    train_images, train_labels, test_images, test_labels, num_labels, num_pixels = load_datasets()
    ds = get_train_batches(batch_size)

    # Train model
    mlp.train(model.params, ds, train_images, train_labels, test_images, test_labels, num_labels, num_pixels, learning_rate, epochs=epochs)

if __name__ == "__main__":
    main()
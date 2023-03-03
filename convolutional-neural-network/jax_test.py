import os
import json
import time
import optax
import numpy as np
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
from flax import linen as nn
from flax.training import train_state
from jax import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
args = parser.parse_args()
tf.random.set_seed(args.seed)

# Suppress warning, info and error messages from jax, tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Specify directory for data 
DATA_DIR = './data'

def get_datasets():
    ds_train, ds_test = tfds.load(
        "mnist", 
        split=["train", "test"],
        as_supervised=True,
        data_dir=DATA_DIR
    )

    pipeline = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale=1./255),
        tf.keras.layers.Flatten()
    ])

    ds_train = ds_train.shuffle(1000)
    ds_train = ds_train.batch(128, drop_remainder=True)
    ds_train = ds_train.map(lambda x, y: (pipeline(x), tf.one_hot(y, depth=10)))
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
    ds_train = ds_train.cache()

    ds_test = ds_test.batch(128, drop_remainder=True)
    ds_test = ds_test.map(lambda x, y: (pipeline(x), tf.one_hot(y, depth=10)))
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)
    ds_test = ds_test.cache()

    return ds_train, ds_test


class MultiLayerPerceptron(nn.Module):
    hidden_size: int = 100
    num_classes: int = 10

    def setup(self):
        self.hidden = nn.Dense(self.hidden_size)
        self.output = nn.Dense(self.num_classes)
        self.relu = nn.relu

    def __call__(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


def loss_fn(logits, labels):
    return optax.softmax_cross_entropy(logits, labels).mean()

def compute_metrics(*, logits, labels):
    loss = loss_fn(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == np.argmax(labels, -1))
    metrics = {
      'loss': loss,
      'accuracy': accuracy,
    }
    return metrics

def create_train_state(rng, learning_rate, batch_size):
    """Creates initial `TrainState`."""
    model = MultiLayerPerceptron()
    variables = model.init(rng, jnp.ones([batch_size, 784]))
    params = variables["params"]
    tx = optax.adam(learning_rate)
    
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, inputs, labels):
    """Train for a single step."""
    
    def objective(params):
        logits = state.apply_fn(
            {'params': params},
            inputs
        )
        loss = loss_fn(logits=logits, labels=labels)
        return loss, logits
    
    grad_fn = jax.value_and_grad(objective, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=labels)
    
    return state, metrics

@jax.jit
def eval_step(state, inputs, labels):
    logits = state.apply_fn(
        {'params': state.params},
        inputs
    )
    metrics = compute_metrics(logits=logits, labels=labels)
    return state, metrics

def train_epoch(state, train_ds, epoch):
    """Train for a single epoch."""
    batch_metrics = []

    for inputs, labels in train_ds:
        inputs = jnp.float32(inputs)
        labels = jnp.int32(labels)
        
        state, metrics = train_step(state, inputs, labels)
        batch_metrics.append(metrics)

    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }

    print('train epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))

    return state, epoch_metrics_np

def eval_model(state, test_ds, epoch):
    batch_metrics = []
    start_eval = time.time()
    for inputs, labels in test_ds:
        inputs = jnp.float32(inputs)
        labels = jnp.int32(labels)

        state, metrics = eval_step(state, inputs, labels)
        batch_metrics.append(metrics)
    end_eval = time.time()
    # compute mean of metrics across each batch in epoch.
    batch_metrics_np = jax.device_get(batch_metrics)
    epoch_metrics_np = {
        k: np.mean([metrics[k] for metrics in batch_metrics_np])
        for k in batch_metrics_np[0]
    }
    epoch_metrics_np["average_batch_inference_time"] = (end_eval - start_eval) / len(test_ds) * 1000  # Batch time in ms
    print('eval epoch: %d, loss: %.4f, accuracy: %.2f' % (epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100))
    return epoch_metrics_np

def train_model(model_state, train_ds, test_ds, epochs=10):
    model_metrics = {}
    epoch_times = []
    start_train = time.time()
    for epoch in range(1, epochs + 1): 
        start_epoch = time.time()     
        model_state, train_metrics = train_epoch(model_state, train_ds, epoch) 
        end_epoch = time.time()
        epoch_times.append(end_epoch - start_epoch)
     
        eval_metrics = eval_model(model_state, test_ds, epoch)
        
    end_train = time.time()

    model_metrics = {
        "model_name": "MLP",
        "framework_name": "Jax",
        "dataset": "MNIST Digits",
        "task": "classification",
        "final_training_loss": float(train_metrics["loss"]),
        "final_evaluation_accuracy": float(eval_metrics["accuracy"]),
        "total_training_time": end_train - start_train,
        "average_epoch_training_time": np.mean(epoch_times[1:]),  # Ignore first epoch for JIT time
        "average_batch_inference_time": eval_metrics["average_batch_inference_time"]
    }
    return model_metrics


def main():
    RNG = random.PRNGKey(args.seed)

    print(f"\nEnvironment Config: ")
    print(f"TF_CPP_MIN_LOG_LEVEL = {os.environ['TF_CPP_MIN_LOG_LEVEL']}")

    # Load data
    train_ds, test_ds = get_datasets()

    RNG, init_rng = jax.random.split(RNG)
    LR = 1e-3
    EPOCHS = 10
    BATCH_SIZE = 128
    state = create_train_state(init_rng, LR, BATCH_SIZE)
    
    metrics = train_model(state, train_ds, test_ds, EPOCHS)
    
    # Export to JSON
    print(f"\nTraining Metrics: \n{metrics}")
    
    date_str = time.strftime("%Y-%m-%d-%H%M%S")
    with open(f"./output/m1-jax-mlp-{date_str}.json", "w") as outfile:
        json.dump(metrics, outfile)


if __name__ == "__main__":
    main()
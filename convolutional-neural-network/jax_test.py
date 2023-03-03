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
from functools import partial
from typing import Any
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

def get_datasets(batch_size=32):
    """
    Creates train, validation, and test datasets.
    Applies data normalization to all datasets and augmentation to training only.
    """
    train_ds, val_ds, test_ds = tfds.load(
        "cifar10", 
        split=["train[:90%]", "train[90%:]", "test"],
        as_supervised=True,
        shuffle_files=True
    )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    var = [x ** 2 for x in std]

    augment_pipeline = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale=1./255),
        tf.keras.layers.Normalization(mean=mean, variance=var),
        tf.keras.layers.ZeroPadding2D(padding=(4, 4)),
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomCrop(height=32, width=32)
    ])

    evaluate_pipeline = tf.keras.Sequential([
        tf.keras.layers.Rescaling(scale=1./255),
        tf.keras.layers.Normalization(mean=mean, variance=var),
    ])

    augment_pipeline.compile()
    evaluate_pipeline.compile()

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.batch(batch_size, drop_remainder=True).map(lambda x, y: (augment_pipeline(x, training=True), y))
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)

    val_ds = val_ds.batch(batch_size, drop_remainder=True).map(lambda x, y: (evaluate_pipeline(x, training=False), y))
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    
    test_ds = test_ds.batch(batch_size, drop_remainder=True).map(lambda x, y: (evaluate_pipeline(x, training=False), y))
    test_ds = test_ds.cache().prefetch(AUTOTUNE)
    
    return train_ds, val_ds, test_ds


class IdentityResidual(nn.Module):
    out_channels: int
    stride: int = 1

    def __call__(self, x):
        _, _, _, c = x.shape  # BHWC
        x = x[:, ::self.stride, ::self.stride, :]  # Downsample spatial dims
        if c != self.out_channels:  # Pad extra channels
            b, h, w, c = x.shape
            pad = jnp.zeros((b, h, w, self.out_channels - c))
            x = jnp.concatenate([x, pad], axis=-1)
        return x


class ResNetV2Layer(nn.Module):
    out_channels: int
    stride: int = 1

    def setup(self):
        conv_kwargs = {"padding": "SAME", "use_bias": False, "kernel_size": (3, 3)}
        self.conv1 = nn.Conv(self.out_channels, strides=self.stride, **conv_kwargs)
        self.conv2 = nn.Conv(self.out_channels, **conv_kwargs)
        self.bn1 = nn.BatchNorm(use_running_average=True, momentum=0.9)  # Momentum set to match PyTorch
        self.bn2 = nn.BatchNorm(use_running_average=True, momentum=0.9)
        self.residual = IdentityResidual(self.out_channels, self.stride)
        self.relu = nn.relu

    def __call__(self, x):
        residual = self.residual(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual


class ResNetV2Model(nn.Module):
    output_classes: int = 10

    @nn.compact
    def __call__(self, x):
        return nn.Sequential([
            nn.Conv(16, kernel_size=(3, 3), padding="SAME", use_bias=False),
            ResNetV2Layer(16),
            ResNetV2Layer(16),
            ResNetV2Layer(16),
            ResNetV2Layer(32, stride=2),
            ResNetV2Layer(32),
            ResNetV2Layer(32),
            ResNetV2Layer(64, stride=2),
            ResNetV2Layer(64),
            ResNetV2Layer(64),
            partial(jnp.mean, axis=(1, 2)),  # Global average pooling over spatial dims
            nn.Dense(self.output_classes)
        ])(x)


def loss_fn(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

def compute_metrics(*, logits, labels):
    loss = loss_fn(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
      'loss': loss,
      'accuracy': accuracy,
    }
    return metrics


class TrainState(train_state.TrainState):
    """Custom train state for BatchNorm stats"""
    batch_stats: Any


def create_train_state(rng, learning_rate, batch_size, weight_decay=1e-4):
    """Creates initial `TrainState`."""
    model = ResNetV2Model()
    variables = model.init(rng, jnp.ones([batch_size, 32, 32, 3]))
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    tx = optax.adamw(learning_rate, weight_decay=weight_decay)
    return TrainState.create(apply_fn=model.apply, params=params, batch_stats=batch_stats, tx=tx)

@jax.jit
def train_step(state, inputs, labels):
    """Train for a single step."""
    
    def objective(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            inputs, mutable=['batch_stats']  # Mutate batch stats during train step
        )
        loss = loss_fn(logits=logits, labels=labels)
        return loss, (logits, updates)
    
    grad_fn = jax.value_and_grad(objective, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])  # Update with new batch stats
    metrics = compute_metrics(logits=logits, labels=labels)
    return state, metrics

@jax.jit
def eval_step(state, inputs, labels):
    logits = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},  # Use current batch stats in state
        inputs  # Don't mutate batch stats in eval
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

def train_model(model_state, train_ds, val_ds, test_ds, epochs=10):
    model_metrics = {}
    epoch_times = []
    start_train = time.time()
    for epoch in range(1, epochs + 1): 
        start_epoch = time.time()     
        model_state, train_metrics = train_epoch(model_state, train_ds, epoch) 
        end_epoch = time.time()
        epoch_times.append(end_epoch - start_epoch)
     
        eval_metrics = eval_model(model_state, val_ds, epoch)
        
    end_train = time.time()

    test_metrics = eval_model(model_state, test_ds, 0)

    model_metrics = {
        "model_name": "ResNetV2-20",
        "framework_name": "Jax",
        "dataset": "CIFAR-10",
        "task": "classification",
        "final_training_loss": float(train_metrics["loss"]),
        "final_evaluation_accuracy": float(eval_metrics["accuracy"]),
        "final_test_accuracy": float(test_metrics["accuracy"]),
        "total_training_time": end_train - start_train,
        "average_epoch_training_time": np.mean(epoch_times[1:]),  # Ignore first epoch for JIT time
        "average_batch_inference_time": eval_metrics["average_batch_inference_time"]
    }
    return model_metrics


def main():
    RNG = random.PRNGKey(args.seed)

    print(f"\nEnvironment Config: ")
    print(f"TF_CPP_MIN_LOG_LEVEL = {os.environ['TF_CPP_MIN_LOG_LEVEL']}")

    RNG, init_rng = jax.random.split(RNG)
    LR = 1e-3
    EPOCHS = 10
    BATCH_SIZE = 128

    # Force Jax to attempt to run the model at least once or else it fails to compile later
    batch = jax.random.normal(RNG, (4, 32, 32, 3))
    model = ResNetV2Model()
    variables = model.init(RNG, batch)
    output = model.apply(variables, batch)
    print("Model sucessfully compiled")

    # Load data
    train_ds, val_ds, test_ds = get_datasets(BATCH_SIZE)

    state = create_train_state(init_rng, LR, BATCH_SIZE)
    
    metrics = train_model(state, train_ds, val_ds, test_ds, EPOCHS)
    
    # Export to JSON
    print(f"\nTraining Metrics: \n{metrics}")
    
    date_str = time.strftime("%Y-%m-%d-%H%M%S")
    with open(f"./output/m2-jax-cnn-{date_str}.json", "w") as outfile:
        json.dump(metrics, outfile)


if __name__ == "__main__":
    main()
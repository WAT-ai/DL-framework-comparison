import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, GlobalAveragePooling2D, Dense
import tensorflow_datasets as tfds
from timeit import default_timer
import time
import json
import numpy as np

# Hyperparameters
BATCH_SIZE = 32

# ResNet Layers
class IdentityResidual(tf.keras.layers.Layer):
    def __init__(self, out_channels, stride):
        super().__init__()
        self.out_channels = out_channels
        self.stride = stride

    def build(self, input_shape):
        b, h, w, in_channels = input_shape
        self.in_channels = in_channels
        self.h = h // self.stride
        self.w = w // self.stride
        self.b = b
        self.c = self.out_channels - self.in_channels

    def call(self, input_tensor):
        # Downsample spatially
        x = input_tensor[:, ::self.stride, ::self.stride, :]
        # Create padding tensor for extra channels 
        if self.out_channels != self.in_channels:
            pad = tf.zeros((self.b, self.h, self.w, self.c))
            # Append padding to the downsampled identity
            x = tf.concat((x, pad), axis=-1)
        return x

class ResNetV2Layer(tf.keras.Model):
    def __init__(self, channels, stride=1):
        super().__init__()
        conv_kwargs = {
            "padding": "same",
            "use_bias": False
        }
        self.stride = stride
        self.channels = channels
        self.relu = tf.nn.relu
        self.residual = IdentityResidual(channels, stride)
        self.conv1 = Conv2D(filters=channels, kernel_size=3, strides=self.stride, **conv_kwargs)
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=channels, kernel_size=3, **conv_kwargs)
        self.bn2 = BatchNormalization()
    
    def call(self, input_tensor, training=False):
        residual = self.residual(input_tensor)
        x = self.bn1(input_tensor, training=training)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual


class TimingCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs={}):
        self.logs = []

        self.start_list = []
        self.end_list = []
        self.difference_list = []

        self.start_train_time = 0.0
        self.end_train_time = 0.0

        self.start_batch_interference_list = []
        self.end_batch_interference_list = []
        self.difference_batch_interference_list = []

        self.final_training_loss = 0.0
        self.final_evaluation_accuracy = 0.0

    def on_train_begin(self, logs={}):
        self.start_train_time = time.time()

    def on_train_end(self, logs={}):
        self.end_train_time = time.time()

    def on_test_batch_begin(self, batch, logs={}):
        self.starttime = time.time()
        self.start_batch_interference_list.append(self.starttime)

    def on_test_batch_end(self, batch, logs={}):
        self.endtime = time.time()
        self.end_batch_interference_list.append(self.endtime)
        self.difference_batch_interference_list.append(self.endtime - self.starttime)

    def on_epoch_begin(self, epoch, logs={}):
        self.starttime = time.time()
        self.start_list.append(self.starttime)

    def on_epoch_end(self, epoch, logs={}):
        self.final_training_loss += logs["loss"]
        self.final_evaluation_accuracy += logs["categorical_accuracy"]
        self.endtime = time.time()
        self.end_list.append(self.endtime)
        self.difference_list.append(self.endtime - self.starttime)


def get_datasets():
    """
    Creates train, validation, and test datasets.
    Applies data normalization to all datasets and augmentation to training only.
    """
    train_ds, val_ds, test_ds = tfds.load(
        "cifar10", 
        split=["train[:90%]", "train[90%:]", "test"],
        as_supervised=True
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

    train_ds = train_ds.map(lambda x, y: (augment_pipeline(x, training=True), y))
    train_ds = train_ds.cache().shuffle(1000).batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)

    val_ds = val_ds.map(lambda x, y: (evaluate_pipeline(x, training=False), y))
    val_ds = val_ds.cache().batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
    
    test_ds = test_ds.map(lambda x, y: (evaluate_pipeline(x, training=False), y))
    test_ds = test_ds.cache().batch(BATCH_SIZE, drop_remainder=True).prefetch(AUTOTUNE)
    
    return train_ds, val_ds, test_ds

def get_compiled_model():
    ResNetV2Model = tf.keras.Sequential([
        Conv2D(filters=16, kernel_size=3, padding="same", use_bias=False, data_format="channels_last"),
        ResNetV2Layer(16),
        ResNetV2Layer(16),
        ResNetV2Layer(16),
        ResNetV2Layer(32, stride=2),
        ResNetV2Layer(32),
        ResNetV2Layer(32),
        ResNetV2Layer(64, stride=2),
        ResNetV2Layer(64),
        ResNetV2Layer(64),
        GlobalAveragePooling2D(),
        Dense(10)
    ])

    # Model needs dummy input to build
    inputs = tf.random.normal((32, 32, 32, 3))
    z = ResNetV2Model(inputs)
    
    ResNetV2Model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-4),
        # optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(),
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        ]
    )
    return ResNetV2Model


def main():
    train_ds, val_ds, test_ds = get_datsets()

    model = get_compiled_model()

    cb = TimingCallback()

    # Fitting the model
    history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[cb])

    # Evaluate the model
    test_metrics = model.evaluate(test_ds)

    total_training_time = cb.end_train_time - cb.start_train_time
    average_epoch_training_time = np.mean(cb.difference_list)
    average_batch_interference_time = np.mean(cb.difference_batch_interference_list) * 1000
    final_eval_accuracy = history["val_sparse_categorical_accuracy"][-1]
    final_train_loss = history["loss"][-1]

    metrics = {
        "model_name": "ResNetV2-20",
        "framework_name": "TensorFlow",
        "dataset": "MNIST Digits",
        "task": "classification",
        "final_training_loss": final_train_loss,
        "final_evaluation_accuracy": final_accuracy,
        "final_test_accuracy": test_metrics["sparse_categorical_accuracy"],
        "total_training_time": total_training_time,  # in seconds
        "average_epoch_training_time": average_epoch_training_time,  # in seconds
        "average_batch_interference_time": average_batch_interference_time  # in milliseconds
    }

    for key, value in metrics.items():
        print(f'{key} : {value}')

    with open("output.json", "w") as outfile:
        json.dump(metrics, outfile)


if __name__ == "__main__":
    main()
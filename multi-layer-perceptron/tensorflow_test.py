import tensorflow as tf
import tensorflow_datasets as tfds
import time
import json
import numpy as np


def main():
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., tf.one_hot(label, depth=10)

    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    # Creating the Model...
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

    # Compiling the Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.CategoricalAccuracy()],
    )

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

    cb = TimingCallback()

    # Fitting the Model...
    history = model.fit(ds_train, epochs=10, validation_data=ds_test, callbacks=[cb])

    total_training_time = cb.end_train_time - cb.start_train_time
    average_epoch_training_time = np.mean(cb.difference_list)
    average_batch_interference_time = np.mean(cb.difference_batch_interference_list) * 1000
    final_eval_accuracy = history["val_categorical_accuracy"][-1]
    final_training_loss = history["loss"][-1]

    metrics = {
        "model_name": "MLP",
        "framework_name": "TensorFlow",
        "dataset": "MNIST Digits",
        "task": "classification",
        "Total Training Time": total_training_time,  # in seconds
        "Final Training Loss": final_training_loss,
        "Final Evaluation Accuracy": final_eval_accuracy,
        "Average Epoch Training Time": average_epoch_training_time,  # in seconds
        "Average Batch Inference Time": average_batch_interference_time  # in milliseconds
    }

    for key, value in metrics.items():
        print(f'{key} : {value}')

    with open("m1-tensorflow-mlp.json", "w") as outfile:
        json.dump(metrics, outfile)


if __name__ == "__main__":
  main()

'''
Wat.ai Python/Julia framework benchmark comparison Project
Milestone 1

Model Specifications:
Model: Multi-layer Perceptron (MLP)
- Input size: 784 (28 x 28 flattened)
- Hidden layer size: 100
- Hidden activation function: ReLU
- Number of outputs: 10
- Loss function: cross entropy
- Metric: accuracy

Data: MNIST handwritten digits 
- Train/Test split: Use the MNIST split (60000,10000)
- Pre-processing: normalize by dividing by 255, flatten from (28 x 28 x 60000) to (784 x 60000)
- Pre-processing targets: one hot vectors

Hyperparameters:
- Optimizer: Adam
- learning rate: 1e-4
- beta_1: 0.9
- beta_2: 0.999
- Number of epochs for training: 10
- Batch size: 128

Metrics to record:
- Total training time (from start of training script to end of training run)
- Training time per 1 epoch (measure from start to end of each epoch and average over all epochs)
- Inference time per batch (measure per batch and average over all batches)
- Final training loss
- Final evaluation accuracy
'''

# importing any necessary libraries

# Mxnet
import mxnet as mx
from mxnet import gluon, autograd as ag, nd
import numpy as np

import time # for benchmark measurements 

# json library neded to export metrics 
import json

# importing the mnist dataset
from keras.datasets import mnist
 

## Data Loading and Pre-processing

#import 60000 (training) and 10000 (testing images from mnist data set
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Changing the np.array to mx.nd.array

X_train = mx.nd.array(X_train)
X_test = mx.nd.array(X_test)

y_train = mx.nd.array(y_train)
y_test = mx.nd.array(y_test)

# Normalizing the training values + reshaping

X_train = X_train/255 
X_test = X_test/255

X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

# Converting y-labels to one-hot vectors

y_train = mx.nd.one_hot(y_train, 10)
y_test = mx.nd.one_hot(y_test, 10)

# Creating a batch data iterator, with batch_size = 128
batch_size = 128

train_data = mx.io.NDArrayIter(X_train, y_train , batch_size, shuffle=True) # shuffle = True since order doesn't particularly matter
val_data = mx.io.NDArrayIter(X_test, y_test, batch_size) 

## Devloping MLP model

 # setting up a sequential neural network initializers, layers
net = gluon.nn.Sequential()
    # creating a chain of neural network layers (one hidden layer, and an output layer with 10 output vars)
with net.name_scope():
    net.add(gluon.nn.Dense(100, activation = 'relu'))
    net.add(gluon.nn.Dense(10))
# Initializing the parameters 

net.initialize()
# Applying the Adam optimizer with its parameters according to our constraints

trainer= gluon.Trainer(net.collect_params(), 'adam', optimizer_params = {'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999})


## Model training


# Initializing time related variables and lists (to make it easier for metric outputs)
tic = time.thread_time()
# time list vars for epoch times
tick = []
timer = []
tick.append(time.thread_time()-tic)

# time list vars for batch inference time
b_tick = []
b_timer = []
b_tick.append(time.thread_time()-tic)

epoch = 10
num_examples = X_train.shape[0]

# Use Accuracy as the evaluation metric.
metric = mx.metric.Accuracy()

# Using Softmax Cross Entropy for the loss function (make sure to set sparse_label = False)
softmax_ce = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)


for i in range(epoch):
    # creating a cumulative loss variable
    cum_loss = 0
    # Reset the train Data Iterator.
    train_data.reset()
    # Loop over the training Data Tterator.
    for batch in train_data:
        # Splits train data and its labels into multiple slices
        # one slice will be used since we are just using 1 context
        data = gluon.utils.split_data(batch.data[0], batch_axis=0, num_slice = 1)
        label = gluon.utils.split_data(batch.label[0], batch_axis=0, num_slice = 1)

        # initializing var to store the output values from the model
        outputs = []

        # Inside the training scope
        with ag.record():
            for x, y in zip(data, label):
                # inputting the data into the network 
                z = net(x)

                # Computing softmax cross entropy loss.
                loss = softmax_ce(z, y)

                # Backpropagate the error for one iteration.
                loss.backward()
                outputs.append(z)

                # summation of the loss (will be divided by the sample_size at the end of the epoch)
                cum_loss += nd.sum(loss).asscalar()
        
        # Decoding the 1H encoded data 
        # (this is IMPORTANT since it affects the input shape and will give an error)
        # metric.update takes inputs of a list of ND array so it is to be as type list 
        label = [np.argmax(mx.nd.array(label[0]), axis = 1)]

        # Evaluating the accuracy based on the training batch datasets
        metric.update(label, outputs)
        # Make one step of parameter update. Trainer needs to know the
        # batch size of data to normalize the gradient by 1/batch_size.
        trainer.step(batch.data[0].shape[0])
        b_tick.append(time.thread_time()-tic)
        b_timer.append(b_tick[-1]-b_tick[-2])
    
    # Gets the evaluation result.
    name, acc = metric.get()  

    # evaluating the time elapsed between one epoch
    tick.append(time.thread_time()-tic)
    timer.append(tick[-1]-tick[-2])

    # resetting the accuracy metric for next epoch
    metric.reset()
    print("Epoch %s | Loss: %.6f, Train_acc: %.6f, in %.2fs" %
    (i+1, cum_loss/num_examples, acc, timer[i]))
print("-"*70)
 

 ## Model validation

 # Reseting the validation Data Iterator
val_data.reset()

# Loop over the validation Data Iterator.
for batch in val_data:
    # Splits val data and its labels into multiple slices
    data = gluon.utils.split_data(batch.data[0], batch_axis=0, num_slice = 1)
    label = gluon.utils.split_data(batch.label[0], batch_axis=0, num_slice = 1)

    # Initializing the model output var
    outputs = []
    for x in data:
        outputs.append(net(x))

    # Evaluating the accuracy of the model based on val batch datasets
    label = [np.argmax(mx.nd.array(label[0]), axis = 1)]
    metric.update(label, outputs)

    # metric.get ouputs as (label, value), so will use val_acc[1]
    name, val_acc = metric.get()
# assert metric.get()[1] > 0.94

## Metrics export to JSON
metrics = {
    'model_name': 'MLP',
    'framework_name': 'MxNet',
    'dataset': 'MNIST Digits',
    'task': 'classification',
    'total_training_time': np.sum(timer), 
    'average_epoch_training_time': np.average(timer), 
    'average_batch_inference_time': np.average(b_timer)*1000,
    'final_training_loss': cum_loss/num_examples, 
    'final_evaluation_accuracy': val_acc
}

# print(metrics)

with open('m1-mxnet-mlp.json', 'w') as outfile:
    json.dump(metrics, outfile)



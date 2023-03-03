# importing necesary libraries
# Mxnet and numpy imports
import numpy as np

import mxnet as mx
from mxnet import gluon, nd, autograd as ag
from mxnet.gluon import nn

# Libraries for datasets and pre-preprocessing
from mxnet.gluon.data.vision import transforms, CIFAR10
from gluoncv.data import transforms as gcv_transforms
import torch.utils # needed to split the training DS into train_data and cv_data


# json library neded to export metrics 
import json
import time

# Miscellaneous libraries incase I need them for testing
import math

## Class definitions
# Defining the ResNetV2 Class Structure
# ResenetV2 architecture
class BasicBlock(nn.Block):
    def __init__ (self, in_channels, channels, strides = 1 , **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        conv_kwargs = {
            "kernel_size": (3,3),
            "padding": 1,
            "use_bias": False
        }
        self.strides = strides
        self.in_channels = in_channels
        self.channels = channels

        self.bn1 = nn.BatchNorm(in_channels= in_channels)        
        self.conv1 = nn.Conv2D(channels, strides= strides,  in_channels= in_channels, **conv_kwargs) 
        
        self.bn2 = nn.BatchNorm(in_channels= channels)
        self.conv2 = nn.Conv2D(channels, in_channels= channels, **conv_kwargs)
        self.relu = nn.Activation('relu')
        
    def downsample(self,x):
    # Downsample with 'nearest' method (this is striding if dims are divisible by stride)
    # Equivalently x = x[:, :, ::stride, ::stride].contiguous()   
        x = x[:,:, ::self.strides, ::self.strides]
        #creating padding tenspr for extra channels
        (b, c, h, w) = x.shape
        num_pad_channels = self.channels - self.in_channels
        pad = mx.nd.zeros((b, num_pad_channels, h,w))
        # append this padding to the downsampled identity
        x = mx.nd.concat(x , pad, dim = 1)
        return x

    def forward(self, x):
        if self.strides > 1:
            residual = self.downsample(x)
        else:
            residual = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x + residual

# ResnetV2-20
class ResNetV2(nn.Block):
    def __init__(self, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)

        self.input_layer = nn.Conv2D(in_channels=3, channels=16, kernel_size=(3,3), padding=1, use_bias=False)

        self.layer_1 = BasicBlock(16, 16)
        self.layer_2 = BasicBlock(16, 16)
        self.layer_3 = BasicBlock(16, 16)

        self.layer_4 = BasicBlock(16, 32, strides=2)
        self.layer_5 = BasicBlock(32, 32)
        self.layer_6 = BasicBlock(32, 32)

        self.layer_7 = BasicBlock(32, 64, strides=2)
        self.layer_8 = BasicBlock(64, 64)
        self.layer_9 = BasicBlock(64, 64)

        self.flatten = nn.Flatten()

        self.pool = nn.GlobalAvgPool2D(layout='NCHW')
        self.output_layer = nn.Dense(units=10, in_units=64)

    def forward (self, x):
        out = self.input_layer(x)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.layer_6(out)
        out = self.layer_7(out)
        out = self.layer_8(out)
        out = self.layer_9(out)
        out = self.pool(out)
        out = self.flatten(out)
        out = self.output_layer(out)
        return out

# Function definitions
def import_and_transform_data(transform_train, transform_test):
    """Imports the CIFAR-10 dataset and transforms them using the transform functions

    Args:
        transform_train (mxnet.gluon.data.vision.transforms.Compose): the transformation function that will be implemented on the training dataset.
        transform_test (mxnet.gluon.data.vision.transforms.Compose): the transformation function that will be implemented on the testing dataset.
    Outputs:
        full_train_ds: 50,000 transformed image dataset used for training models. Can be further split into train:validation datasets
        test_ds: 10,000 transformed image dataset used for testing models.
    """
    # Creating the train and test DS
    full_train_ds = CIFAR10(train=True).transform_first(transform_train, lazy=False)
    test_ds = CIFAR10(train=False).transform_first(transform_test, lazy=False)
    return full_train_ds, test_ds


def data_split_and_load(train_ds, test_ds, batch_size = 256, train_cv_split_ratio = 0.9):
    """Imports, pre-processes, and loads a dataset into a DataLoader

    Args:
        train_ds (mxnet.gluon.data.dataset.SimpleDataset): Training dataset to be used. Will be split into train and validation datasets using the ratio.
        test_ds (mxnet.gluon.data.dataset.SimpleDataset): Testing dataset to be used.
        batch_size (int): used in gluon.data.Dataloader() to split the datasets into batches of size batch_size
        train_cv_split_ratio (float): ratio of train:cv split. default set to 0.9 (90% train: 10% Cross-validation)
    Outputs:
        train_data: split training data implemented as a DataLoader type
        cv_data: split validation data implemeneted as a DataLoader type
        test_data: hold-out (test) data implemented as a Dataloader type
    
    """
    # Splitting the training datasets into the train_data and cv_data
    train_size = int(train_cv_split_ratio * len(train_ds))
    cv_size = len(train_ds) - train_size
    train_ds, cv_ds = torch.utils.data.random_split(train_ds, [train_size, cv_size]) 
    
    # Loading the datasets into the DataLoader
    train_data = gluon.data.DataLoader(train_ds, batch_size=batch_size,  shuffle=True, last_batch='discard')
    cv_data = gluon.data.DataLoader(cv_ds, batch_size=batch_size,  shuffle=True, last_batch='discard')
    test_data = gluon.data.DataLoader(test_ds, batch_size=batch_size,  shuffle=True, last_batch='discard')
    
    return train_data, cv_data, test_data, batch_size, train_size, cv_size

def net_intialize(net, optimizer, lr, beta_1, beta_2, wd):
    """Initialize the network and load it into the trainer

    Args:
        net (__main__.network): CNN model class. In our cse, this is the ResNetV2 class
        optimizer (string): Optimizer to be used
        lr (float): learning rate of model
        beta_1 (float): beta_1 value
        beta_2 (float): beta_2 value
        wd (float): weight decay to be used
        
    Outputs:
        trainer: Trainer class for forward and backward propogation during runs
        net: CNN network
    """
    # net = ResNetV2()
    net.initialize()
    trainer = gluon.Trainer(
        params = net.collect_params(),
        optimizer= optimizer,
        optimizer_params = {'learning_rate': lr, 'beta1': beta_1, 'beta2': beta_2, 'wd':wd}
    ) # The guidelines state using AdamW optimizer, unsure whether 'adam' is sufficient
    return trainer, net

def train(net, batch_size, num_examples, epochs = 50):
    # Initializing time related variables and lists (to make it easier for metric outputs)

    # initializing the training times
    tic_total_train = time.time()
    epoch_times = []

    # defining the accuracy evaluation metric
    metric = mx.metric.Accuracy()

    # Loss function
    softmax_ce = gluon.loss.SoftmaxCrossEntropyLoss()

    for epoch in range(epochs):
        tic_train_epoch = time.time()
        # creating cumulative loss variable
        cum_loss = 0
        # Resetting train_data iterator

        
        # Looping over train_data iterator
        for data, label in train_data:
            
            # Inside training scope
            with ag.record():
                # Inputting the data into the nn
                outputs = net(data)
                
                # outputs =outputs.argmax(axis=1)
                # label = label.astype('float32').mean().asscalar()
                # # Computing the loss
                loss = softmax_ce(outputs,label)

            # Backpropogating the error
            loss.backward()
        
            # Summation of loss (divided by sample_size in the end)
            cum_loss += nd.sum(loss).asscalar()
            metric.update(label,outputs)

            trainer.step(batch_size)
        
        # Get evaluation results    
        name, acc = metric.get()  
        metric.reset()
        toc_train_epoch = time.time()
        epoch_times.append(toc_train_epoch - tic_train_epoch)
        
        ## CROSS VALIDATION DATASET
        # Looping over cv_data iterator
        tic_val = time.time() # initializing cv timer
        for data, label in cv_data:
            val_outputs = net(data)
            
            metric.update(label, val_outputs)
        
        # Getting evaluation results for cv dataset
        name, val_acc = metric.get()
        metric.reset()
        
        # Evaluating time elapse between the cv dataset
        toc_val = time.time() 
        
        print("Epoch %s | Loss: %.6f, Train_acc: %.6f, Val_acc: %.6f, in %.2fs " %
        (epoch+1, cum_loss/num_examples, acc, val_acc, epoch_times[epoch]))
    toc_total_train = time.time() # total training time
    print("Total_training_time:  %.2fs" %(toc_total_train - tic_total_train))
    print("-"*70)
    
    return acc, val_acc, tic_val, toc_val, tic_total_train, toc_total_train, epoch_times, cum_loss

def test():
    metric = mx.metric.Accuracy()

    # Looping over cv_data iterator
    for data, label in test_data:
        test_outputs = net(data)
        
        metric.update(label, test_outputs)

    # Getting evaluation results for cv dataset
    name, test_acc = metric.get()
    print('Test_acc: ', test_acc)
    print("-"*70)
    return test_acc

def get_metrics(tic_total_train, toc_total_train, tic_val, toc_val, epoch_times, batch_size, train_size, cv_size, cum_loss, val_acc, test_acc):
    metrics = {
        'model_name': 'ResNetV2-20',
        'framework_name': 'MxNet',
        'dataset': 'CIFAR-10',
        'task': 'classification',
        'total_training_time': toc_total_train - tic_total_train, # s
        'average_epoch_training_time': np.average(epoch_times), # s
        'average_batch_inference_time': 1000*np.average(toc_val - tic_val)/math.ceil(cv_size/batch_size), # ms
        'final_training_loss': cum_loss/train_size, 
        'final_evaluation_accuracy': val_acc, 
        'final_test_accuracy': test_acc 
    }

    print("Metrics: ")
    print(metrics)
    # Exporting the metrics file
    with open('m2-mxnet-cnn.json', 'w') as outfile:
        json.dump(metrics, outfile)
    return metrics
 
 

    
## Data Import and pre-processing
# defining the transformation functions for the image data

# As mentioned in the notebook transform_train will be used on both train_data and cv_data, while transform_test will be used on test_data. Since training
# dataset provides more randomized data (and should be more generalizable), I will not be performing the random operations on the testing dataset.

def main():
    transform_train = transforms.Compose([
        gcv_transforms.RandomCrop(32, pad=4), # Randomly crop an area and resize it to be 32x32, then pad it to be 40x40
        transforms.RandomFlipLeftRight(), # Applying a random horizontal flip
        transforms.ToTensor(), # Transpose the image from height*width*num_channels to num_channels*height*width
                                                           # and map values from [0, 255] to [0,1]
        # Normalize the image with mean and standard deviation calculated across all images
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    
    # Importing and transforming data     
    full_train_ds , test_ds =  import_and_transform_data(transform_train, transform_test) 
    
    # Splitting data and Loading into DataLoader
    train_data, cv_data, test_data, batch_size , train_size, cv_size = data_split_and_load(full_train_ds, test_ds) 

    # initializing network and loading into the trainer
    trainer, net = net_intialize(ResNetV2(), 'adam', 0.001, 0.9, 0.999, 0.0001)

    ## Training the Model
    training_results = [] # storing all the variables returned from the train fn.
    training_results = train(net, batch_size , num_examples=train_size)

    ## Runnning Model on Hold-out Dataset
    test_acc = test()

    ## Metrics Acquisition
    get_metrics(training_results[4], training_results[5], training_results[2], training_results[3], training_results[6], batch_size, train_size, cv_size, training_results[7], training_results[1], test_acc)


if __name__ == "__main__":
    main()
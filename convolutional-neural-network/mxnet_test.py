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

# Other imports
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
args = parser.parse_args()
mx.random.seed(args.seed)

# Supress warnings
os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"

# GPU context
ctx = mx.gpu(0)

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
        pad = mx.nd.zeros((b, num_pad_channels, h,w), ctx=ctx)
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
    full_train_ds = CIFAR10(train=True).transform_first(transform_train)
    test_ds = CIFAR10(train=False).transform_first(transform_test)
    return full_train_ds, test_ds


def data_split_and_load(train_ds, test_ds, batch_size=128, train_cv_split_ratio=0.9):
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
    
    return train_data, cv_data, test_data

def net_intialize(net, optimizer, lr, beta_1, beta_2):
    """Initialize the network and load it into the trainer

    Args:
        net (__main__.network): CNN model class. In our cse, this is the ResNetV2 class
        optimizer (string): Optimizer to be used
        lr (float): learning rate of model
        beta_1 (float): beta_1 value
        beta_2 (float): beta_2 value
        
    Outputs:
        trainer: Trainer class for forward and backward propogation during runs
        net: CNN network
    """
    # net = ResNetV2()
    net.initialize()
    params = net.collect_params()
    params.reset_ctx(ctx)  # Put on GPU
    trainer = gluon.Trainer(
        params=params,
        optimizer= optimizer,
        optimizer_params={'learning_rate': lr, 'beta1': beta_1, 'beta2': beta_2}
    )
    return trainer, net

def train(net, trainer, train_data, cv_data, batch_size, epochs=10):
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
        
        # Looping over train_data iterator
        for data, label in train_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            # Inside training scope
            with ag.record():
                # Inputting the data into the nn
                outputs = net(data)
                # Computing the loss
                loss = softmax_ce(outputs,label)

            # Backpropogating the error
            loss.backward()
            trainer.step(batch_size)
        
            # Summation of loss (divided by sample_size in the end)
            cum_loss += loss.mean().asscalar()
            metric.update(label, outputs)

        
        # Get evaluation results    
        name, acc = metric.get()  
        metric.reset()
        toc_train_epoch = time.time()
        epoch_times.append(toc_train_epoch - tic_train_epoch)
        
        ## CROSS VALIDATION DATASET
        # Looping over cv_data iterator
        tic_val = time.time() # initializing cv timer
        for data, label in cv_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            val_outputs = net(data)
            
            metric.update(label, val_outputs)
        
        # Getting evaluation results for cv dataset
        name, val_acc = metric.get()
        metric.reset()
        
        # Evaluating time elapse between the cv dataset
        toc_val = time.time() 
        
        print("Epoch %s | Loss: %.6f, Train_acc: %.6f, Val_acc: %.6f, in %.2fs " %
        (epoch+1, cum_loss/len(train_data), acc, val_acc, epoch_times[epoch]))
    toc_total_train = time.time() # total training time
    print("Total_training_time:  %.2fs" %(toc_total_train - tic_total_train))
    print("-"*70)

    metrics = {
        'model_name': 'ResNetV2-20',
        'framework_name': 'MxNet',
        'dataset': 'CIFAR-10',
        'task': 'classification',
        'total_training_time': toc_total_train - tic_total_train, # s
        'average_epoch_training_time': np.average(epoch_times), # s
        'average_batch_inference_time': 1000 * (toc_val - tic_val) / len(cv_data), # ms
        'final_training_loss': cum_loss/len(train_data), 
        'final_evaluation_accuracy': val_acc, 
    }
    
    return metrics

def test(net, test_data):
    metric = mx.metric.Accuracy()

    # Looping over cv_data iterator
    for data, label in test_data:
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        test_outputs = net(data)
        
        metric.update(label, test_outputs)

    # Getting evaluation results for cv dataset
    name, test_acc = metric.get()
    print('Test_acc: ', test_acc)
    print("-"*70)
    metrics = {"final_test_accuracy": test_acc}
    return metrics


def main():
    # Hyperparameters
    BATCH_SIZE = 128
    LR = 1e-3

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
    full_train_ds , test_ds = import_and_transform_data(transform_train, transform_test) 
    
    # Splitting data and Loading into DataLoader
    train_data, cv_data, test_data = data_split_and_load(full_train_ds, test_ds, batch_size=BATCH_SIZE) 

    # initializing network and loading into the trainer
    trainer, net = net_intialize(ResNetV2(), 'adam', LR, 0.9, 0.999)

    ## Training the Model
    metrics = train(net, trainer, train_data, cv_data, BATCH_SIZE, epochs=10)

    ## Runnning Model on Hold-out Dataset
    test_metrics = test(net, test_data)
    metrics.update(test_metrics)

    print("Metrics: ")
    print(metrics)
    
    # Exporting the metrics file
    date_str = time.strftime("%Y-%m-%d-%H%M%S")
    with open(f'./output/m2-mxnet-cnn-{date_str}.json', 'w') as outfile:
        json.dump(metrics, outfile)


if __name__ == "__main__":
    main()
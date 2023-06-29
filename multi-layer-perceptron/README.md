# WAT.ai Python/Julia framework benchmark comparison Project
## Milestone 1: MLP on MNIST

### Model: Multi-layer Perceptron (MLP)
- Input size: 784 (28 x 28 flattened)
- Hidden layer size: 100
- Hidden activation function: ReLU
- Number of outputs: 10
- Loss function: cross entropy
- Metric: accuracy

### Data: MNIST handwritten digits 
- Train/Test split: Use the MNIST split (60000,10000)
- Pre-processing: normalize by dividing by 255, flatten from (28 x 28 x 60000) to (784 x 60000)
- Pre-processing targets: one hot vectors

### Hyperparameters:
- Optimizer: Adam
- learning rate: 1e-3
- beta_1: 0.9
- beta_2: 0.999
- Number of epochs for training: 10
- Batch size: 128

### Metrics to record:
- Total training time (from start of training script to end of training run)
- Training time per 1 epoch (measure from start to end of each epoch and average over all epochs)
- Inference time per batch (measure per batch and average over all batches)
- Final training loss
- Final evaluation accuracy
'''
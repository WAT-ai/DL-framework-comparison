# WAT.ai Python/Julia framework benchmark comparison Project
## Milestone 2: ResNetV2-20 on CIFAR-10

### Model Specifications

Model: Convolutional Neural Network (CNN)

Using ResNetV2
- Input layer: Input size: (32 x 32) x 3
    - conv2d (3 x 3) x 64
- ResBlock 1: Input size (32 x 32) x 64
     - conv2d (3 x 3) x 16
     - conv2d (3 x 3) x 16
- ResBlock 2: Input size (32 x 32) x 16
     - conv2d (3 x 3) x 16
     - conv2d (3 x 3) x 16
- ResBlock 3: Input size (32 x 32) x 16
     - conv2d (3 x 3) x 16
     - conv2d (3 x 3) x 16  
- ResBlock 4: Input size (32 x 32) x 16
     - conv2d (3 x 3) x 32, stride 2
     - conv2d (3 x 3) x 32
- ResBlock 5: Input size (16 x 16) x 32
     - conv2d (3 x 3) x 32
     - conv2d (3 x 3) x 32
- ResBlock 6: Input size (16 x 16) x 32
     - conv2d (3 x 3) x 32
     - conv2d (3 x 3) x 32
- ResBlock 7: Input size (16 x 16) x 32
     - conv2d (3 x 3) x 64, stride 2
     - conv2d (3 x 3) x 64
- ResBlock 8: Input size (8 x 8) x 64
     - conv2d (3 x 3) x 64
     - conv2d (3 x 3) x 64
- ResBlock 9: Input size (8 x 8) x 64
     - conv2d (3 x 3) x 64
     - conv2d (3 x 3) x 64
- Pooling: input size (8 x 8) x 64
     - GlobalAveragePooling/AdaptiveAveragePooling((1,1))
- Output layer: Input size (64,)
     - Dense/Linear (64,10)
     - Activation: Softmax
     
### Data: CIFAR-10
- 32 x 32 x 3 RGB colour images
- Train/Test split: Use data splits already given (50,000 train, 10,000 test). From the 50,000 train images, use 45,000 for training and 5,000 for validation every epoch inside the training loop. Reserve the 10,000 test set images for final evaluation.
- Pre-processing inputs: 
     - Depending on data source, scale int8 inputs to [0, 1] by dividing by 255
     - ImageNet normalization 
          - From the RGB channels, subtract means [0.485, 0.456, 0.406] and divide by standard deviations [0.229, 0.224, 0.225]
     - 4 pixel padding on EACH side (40x40), then apply 32x32 crop randomly sampled from the padded image or its horizontal flip as in Section 3.2 of [3]
- Preprocessing labels: Use integer indices

### Hyperparameters:
- Optimizer: AdamW
- learning rate: 1e-3 
- beta_1: 0.9
- beta_2: 0.999
- weight decay: 0.0001
- Number of epochs for training: 10
- Batch size: 128


### Metrics to record:
- Total training time (from start of training script to end of training run)
- Training time per 1 epoch (measure from start to end of each epoch and average over all epochs)
- Inference time per batch (measure per batch and average over all batches)
- Last epoch training loss
- Last epoch eval accuracy (from the 5,000 evaluation dataset)
- Held-out test set accuracy (from the 10,000 test dataset)

# Data processing
using MLDatasets;
using MLDataPattern;
using ImageCore;
using Augmentor;
using ImageFiltering;
using MappedArrays;
using Random;
using Flux: DataLoader;
using Functors;
using Optimisers;
using Statistics;
using TimerOutputs;
using Flux;
using JSON;
using Dates;
using ArgParse;

# Seed arg
s = ArgParseSettings()
@add_arg_table s begin
    "--seed", "-s"
        help = "Random seed"
        arg_type = Int
        default = 42
end

BATCH_SIZE = 128

# DATA PRE-PROCESSING
train_data = MLDatasets.CIFAR10(Tx=Float32, split=:train, dir="./data")
test_data = MLDatasets.CIFAR10(Tx=Float32, split=:test, dir="./data")

train_x = train_data.features;
train_y = train_data.targets;

test_x = test_data.features;
test_y = test_data.targets;

# Train-test split
# Copied from https://github.com/JuliaML/MLUtils.jl/blob/v0.2.11/src/splitobs.jl#L65
# obsview doesn't work with this data, so use getobs instead

import MLDataPattern.splitobs;

function splitobs(data; at, shuffle::Bool=false)
    if shuffle
        data = shuffleobs(data)
    end
    n = numobs(data)
    return map(idx -> MLDataPattern.getobs(data, idx), splitobs(n, at))
end

train, val = splitobs((train_x, train_y), at=0.9, shuffle=true);

train_x, train_y = train;
val_x, val_y = val;

# Normalize all the data

means = reshape([0.485, 0.465, 0.406], (1, 1, 3, 1))
stdevs = reshape([0.229, 0.224, 0.225], (1, 1, 3, 1))
normalize(x) = (x .- means) ./ stdevs

# Normalize the data and convert it back to Float32 as normalizing step converts it to Float64
train_x = Array{Float32}(normalize(train_x));
val_x = Array{Float32}(normalize(val_x));
test_x = Array{Float32}(normalize(test_x));


# DATA AUGMENTATION PIPELINE
# Pad the training data for further augmentation
train_x_padded = padarray(train_x, Fill(0, (4, 4, 0, 0)));  
pl = PermuteDims((3, 1, 2)) |> CombineChannels(RGB) |> Either(FlipX(), NoOp()) |> RCropSize(32, 32) |> SplitChannels() |> PermuteDims((2, 3, 1))
# Create an output array for augmented images
outbatch(X) = Array{Float32}(undef, (32, 32, 3, nobs(X)))
# Function that takes a batch (images and targets) and augments the images
augmentbatch((X, y)) = (augmentbatch!(outbatch(X), X, pl), y)

# Shuffled and batched dataset of augmented images
train_batches = mappedarray(augmentbatch, batchview(shuffleobs((train_x_padded, train_y)), size=BATCH_SIZE));

# Test and Validation data
val_loader = DataLoader((val_x, val_y), shuffle=true, batchsize=BATCH_SIZE);
test_loader = DataLoader((test_x, test_y), shuffle=true, batchsize=BATCH_SIZE);

# RESNET LAYER
mutable struct ResNetLayer
    conv1::Flux.Conv
    conv2::Flux.Conv
    bn1::Flux.BatchNorm
    bn2::Flux.BatchNorm
    f::Function
    in_channels::Int
    channels::Int
    stride::Int
end

@functor ResNetLayer (conv1, conv2, bn1, bn2)

function residual_identity(layer::ResNetLayer, x::AbstractArray{T, 4}) where {T<:Number}
    (w, h, c, b) = size(x)
    stride = layer.stride
    if stride > 1
        @assert ((w % stride == 0) & (h % stride == 0)) "Spatial dimensions are not divisible by `stride`"
    
        # Strided downsample
        inds = CartesianIndices((1:stride:w, 1:stride:h))
        x_id = copy(x[inds, :, :])
    else
        x_id = x
    end

    channels = layer.channels
    in_channels = layer.in_channels
    if in_channels < channels
        # Zero padding on extra channels
        (w, h, c, b) = size(x_id)
        pad = zeros(T, w, h, channels - in_channels, b)
        x_id = cat(x_id, pad; dims=3)
    elseif in_channels > channels
        error("in_channels > out_channels not supported")
    end
    return x_id
end

function ResNetLayer(in_channels::Int, channels::Int; stride=1, f=relu)
    bn1 = Flux.BatchNorm(in_channels)
    conv1 = Flux.Conv((3,3), in_channels=>channels; stride=stride, pad=1, init=Flux.kaiming_uniform, bias=false)
    bn2 = Flux.BatchNorm(channels)
    conv2 = Flux.Conv((3,3), channels=>channels; stride=1, pad=1, init=Flux.kaiming_uniform, bias=false)

    return ResNetLayer(conv1, conv2, bn1, bn2, f, in_channels, channels, stride)
end

function (self::ResNetLayer)(x::AbstractArray)
    identity = residual_identity(self, x)
    z = self.bn1(x)
    z = self.f(z)
    z = self.conv1(z)
    z = self.bn2(z)
    z = self.f(z)
    z = self.conv2(z)

    y = z + identity
    return y
end

# RESNET 20 MODEL
mutable struct ResNet20
    input_conv::Flux.Conv
    resnet_blocks::Chain
    pool::GlobalMeanPool
    dense::Flux.Dense
end

@functor ResNet20

function ResNet20(in_channels::Int, num_classes::Int)
    resnet_blocks = Chain(
        block_1 = ResNetLayer(16, 16),
        block_2 = ResNetLayer(16, 16),
        block_3 = ResNetLayer(16, 16),
        block_4 = ResNetLayer(16, 32; stride=2),
        block_5 = ResNetLayer(32, 32),
        block_6 = ResNetLayer(32, 32),
        block_7 = ResNetLayer(32, 64; stride=2),
        block_8 = ResNetLayer(64, 64),
        block_9 = ResNetLayer(64, 64)
    )
    return ResNet20(
        Flux.Conv((3,3), in_channels=>16, init=Flux.kaiming_uniform, pad=1, bias=false),
        resnet_blocks,
        GlobalMeanPool(),
        Dense(64 => num_classes)
    )
end

function (self::ResNet20)(x::AbstractArray)
    z = self.input_conv(x)
    z = self.resnet_blocks(z)
    z = self.pool(z)
    z = dropdims(z, dims=(1, 2))
    y = self.dense(z)
    return y
end

# TRAINING SETUP

"""
    sparse_logit_cross_entropy(logits, labels)

Efficient computation of cross entropy loss with model logits and integer indices as labels.
Integer indices are from [0,  N-1], where N is the number of classes
Similar to TensorFlow SparseCategoricalCrossEntropy

# Arguments
- `logits::AbstractArray`: 2D model logits tensor of shape (classes, batch size)
- `labels::AbstractArray`: 1D integer label indices of shape (batch size,)

# Returns
- `loss::Float32`: Cross entropy loss
"""

function sparse_logit_cross_entropy(logits, labels)
    log_probs = logsoftmax(logits);
    inds = CartesianIndex.(labels .+ 1, axes(log_probs, 2));
    # Select indices of labels for loss
    log_probs = log_probs[inds];
    loss = -mean(log_probs);
    return loss
end

# Create model with 3 input channels and 10 classes
model = ResNet20(3, 10);
# Setup AdamW optimizer
β = (0.9, 0.999);
decay = 1e-4;
state = Optimisers.setup(Optimisers.Adam(1e-3, β, decay), model);

function evaluate(model, test_loader)
    preds = []
    targets = []
    for (x, y) in test_loader
        # Get model predictions
        # Note argmax of nd-array gives CartesianIndex
        # Need to grab the first element of each CartesianIndex to get the true index
        logits = model(x)
        ŷ = map(i -> i[1], argmax(logits, dims=1))
        append!(preds, ŷ)

        # Get true labels
        append!(targets, y)
    end
    accuracy = sum(preds .== targets) / length(targets)
    return accuracy
end

# TRAINING LOOP
# Setup timing output
const to = TimerOutput()

last_loss = 0;
@timeit to "total_training_time" begin
    for epoch in 1:10
        timing_name = epoch > 1 ? "average_epoch_training_time" : "train_jit"

        # Create lazily evaluated augmented training data
        train_batches = mappedarray(augmentbatch, batchview(shuffleobs((train_x_padded, train_y)), size=train_batch_size));

        @timeit to timing_name begin
            losses = []
            for (x, y) in train_batches
                val, grads = Flux.withgradient(model) do m
                    # Any code inside here is differentiated.
                    # Evaluation of the model and loss must be inside!
                    result = m(x)
                    sparse_logit_cross_entropy(result, y)
                end
                
                # Optimiser updates parameters
                Optimisers.update!(state, model, grads[1])
                push!(losses, val)
            end
            last_loss = mean(losses)
            @info "epoch loss" (mean(losses))
        end
        timing_name = epoch > 1 ? "average_inference_time" : "eval_jit"
        @timeit to timing_name begin
            acc = evaluate(model, test_loader)
            @info "epoch" acc
        end
    end
end

# Train time
# Exclude jit time
average_epoch_train_time = TimerOutputs.time(to["total_training_time"]["average_epoch_training_time"]) / (9 * 1e9)  # Outputs in nanoseconds, conver to seconds

# Eval batch time
# Exclude jit time
num_batches = length(test_loader)
average_eval_batch_time = TimerOutputs.time(to["total_training_time"]["average_inference_time"]) / (9 * 1e6 * num_batches)  # Outputs in nanoseconds, conver to milliseconds
     
total_train_time = TimerOutputs.time(to["total_training_time"]) / 1e9  # Convert nanos to seconds
final_eval_accuracy = evaluate(model, test_loader)
     
metrics = Dict(
    "model_name" => "ResNetV2-20",
    "dataset" => "CIFAR-10",
    "framework_name" => "Flux.jl",
    "task" => "classification",
    "total_training_time" => total_train_time,
    "average_epoch_training_time" => average_epoch_train_time,
    "average_batch_inference_time" => average_eval_batch_time,
    "final_training_loss" => last_loss,
    "final_evaluation_accuracy" => final_eval_accuracy
)

date_str = Dates.format(Dates.now(), "yyyy-mm-dd-HHMMSS")
open("./output/m2-flux-mlp-$(date_str).json", "w") do f
    JSON.print(f, json_result)
end

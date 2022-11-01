# Avalon.jl implementation of multi-layer perceptron

# Imports
using Yota;
using MLDatasets;
using NNlib;
using Statistics;
using Distributions;
using Functors;
using Optimisers;
using MLUtils: DataLoader;
using OneHotArrays: onehotbatch
using Metrics;
using TimerOutputs;
using JSON;


# Primitives
# Linear layer
mutable struct Linear
    W::AbstractMatrix{T} where T
    b::AbstractVector{T} where T
end

@functor Linear

# Init
function Linear(in_features::Int, out_features::Int)
    k_sqrt = sqrt(1 / in_features)
    d = Uniform(-k_sqrt, k_sqrt)
    return Linear(rand(d, out_features, in_features), rand(d, out_features))
end
Linear(in_out::Pair{Int, Int}) = Linear(in_out[1], in_out[2])

function Base.show(io::IO, l::Linear)
    o, i = size(l.W)
    print(io, "Linear($i=>$o)")
end

# Forward
(l::Linear)(x::Union{AbstractVector{T}, AbstractMatrix{T}}) where T = l.W * x .+ l.b

# Cross entropy loss
function logitcrossentropy(ŷ, y; dims=1, agg=mean)
    # Compute cross entropy loss from logits
    # Cross entropy computed from NLL loss on logsoftmax of model outputs
      agg(.-sum(y .* logsoftmax(ŷ; dims=dims); dims=dims));
end


# Model definition
mutable struct Net
    fc1::Linear
    fc2::Linear
end

# Need to mark functor for Optimizer to work
@functor Net

# Init
Net() = Net(
    Linear(28*28, 100),
    Linear(100, 10)
)

# Forward
function (model::Net)(x::AbstractArray)
    x = reshape(x, 28*28, :)
    x = model.fc1(x)
    x = relu(x)
    x = model.fc2(x)
    return x
end

# Create objective function to optimize
function loss_function(model::Net, x::AbstractArray, y::AbstractArray)
    ŷ = model(x)
    loss = logitcrossentropy(ŷ, y)
    return loss
end


# Evaluation function
function evaluate(mlp::Net, test_loader::DataLoader)::Number
    preds = []
    targets = []
    for (x, y) in test_loader
        # Get model predictions
        # Note argmax of nd-array gives CartesianIndex
        # Need to grab the first element of each CartesianIndex to get the true index
        logits = mlp(x)
        ŷ = map(i -> i[1], argmax(logits, dims=1))
        append!(preds, ŷ)

        # Get true labels
        true_label = map(i -> i[1], argmax(y, dims=1))
        append!(targets, true_label)
    end
    accuracy = sum(preds .== targets) / length(targets)
    return accuracy
end


# Data loading and processing
function get_data_loaders(; batch_size=128)
    # Data loading
    train_dataset = MNIST(split=:train);
    test_dataset = MNIST(split=:test);

    X_train = train_dataset.features;
    Y_train = train_dataset.targets;

    X_test = test_dataset.features;
    Y_test = test_dataset.targets;

    # Flatten features to be 784 dim
    X_train = reshape(X_train, 784, :);  # (dim x batch)
    X_test = reshape(X_test, 784, :);

    # Convert targets to one-hot vectors
    Y_train = onehotbatch(Y_train, 0:9);
    Y_test = onehotbatch(Y_test, 0:9);  # (dim x batch)

    train_loader = DataLoader((X_train, Y_train), shuffle=true, batchsize=batch_size);
    test_loader = DataLoader((X_test, Y_test), shuffle=false, batchsize=batch_size);
    return train_loader, test_loader
end

# Setup timing
const to = TimerOutput()


function main()
    train_loader, test_loader = get_data_loaders(batch_size=128)
    
    # Setup model and optimizer
    mlp = Net()

    # Default Β is (0.9, 0.999)
    state = Optimisers.setup(Optimisers.Adam(1e-3), mlp);

    # Training loop
    last_loss = 0;
    @timeit to "total_training_time" begin
        for epoch in 1:10
            # Store the timing in the first epoch into a separate timer for jit
            timing_name = epoch > 1 ? "train_epoch" : "train_jit"
            @timeit to timing_name begin
                losses = []
                for (x, y) in train_loader
                    # loss_function does forward pass
                    # Yota.jl grad function computes model parameter gradients in g[2]
                    loss, g = grad(loss_function, mlp, x, y)
                    
                    # Optimiser updates parameters
                    Optimisers.update!(state, mlp, g[2])
                    push!(losses, loss)
                end
                last_loss = mean(losses)
                @info("epoch $epoch loss = $(mean(losses))")
            end
            timing_name = epoch > 1 ? "eval_epoch" : "eval_jit"
            @timeit to timing_name begin
                acc = evaluate(mlp, test_loader)
                @info("epoch $epoch eval accuracy = $(acc)")
            end
        end
    end

    # Compute timing metrics
    # Outputs in nanoseconds, convert to seconds
    average_epoch_train_time = TimerOutputs.time(to["total_training_time"]["train_epoch"]) / (9 * 1e9)
    total_train_time = TimerOutputs.time(to["total_training_time"]) / 1e9


    num_batches = length(test_loader)
    # Outputs in nanoseconds, conver to milliseconds
    average_eval_batch_time = TimerOutputs.time(to["total_training_time"]["eval_epoch"]) / (9 * 1e6 * num_batches)

    final_eval_accuracy = evaluate(mlp, test_loader)

    metrics = Dict(
        "model_name" => "MLP",
        "dataset" => "MNIST Digits",
        "framework_name" => "Avalon.jl",
        "task" => "classification",
        "total_training_time" => total_train_time,
        "average_epoch_training_time" => average_epoch_train_time,
        "average_batch_inference_time" => average_eval_batch_time,
        "final_trianing_loss" => last_loss,
        "final_evaluation_accuracy" => final_eval_accuracy
    )
    open("m1-avalon-mlp.json","w") do f
        JSON.print(f, metrics)
    end
end


# Run main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
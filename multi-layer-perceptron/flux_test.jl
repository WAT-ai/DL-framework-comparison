# Flux.jl implementation of multi-layer perceptron
using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, logitcrossentropy
using MLDatasets
using JSON
using TimerOutputs
using Random
using ArgParse
using Dates

s = ArgParseSettings()
@add_arg_table s begin
    "--seed", "-s"
        help = "Random seed"
        arg_type = Int
        default = 42
end

function accuracy(data_loader, model)
    acc = 0
    for (x,y) in data_loader
        acc += sum(onecold(cpu(model(x))) .== onecold(cpu(y)))*1 / size(x,2)
    end
    acc/length(data_loader)
end
    
function main()
    args = parse_args(s)
    Random.seed!(args["seed"])

    time_outputs = TimerOutput()
    json_result = Dict()
    
    xtrain, ytrain = MLDatasets.MNIST(split=:train, dir="./data/MNIST/raw")[:]
    xtest, ytest = MLDatasets.MNIST(split=:test, dir="./data/MNIST/raw")[:]
    
    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)
    
    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)
    
    # Load the data
    train_data = DataLoader((xtrain, ytrain), batchsize=128, shuffle=true)
    test_data = DataLoader((xtest, ytest), batchsize=128, shuffle=true)
    
    imgsize=(28,28,1)
    nclasses=10
    model = Chain(Dense(prod(imgsize), 100, relu),
        Dense(100, nclasses))
    
    ## Default is already (0.9, 0.999)
    opt = ADAM(1e-3)
    
    loss_func(x, y)= logitcrossentropy(model(x), y)
    
    params = Flux.params(model)
    
    @timeit time_outputs "train_time" begin
        for epoch in 1:10
            # Store the timing in the first epoch into a separate timer for jit
            timing_name = epoch > 1 ? "train_epoch" : "train_jit"
            @timeit time_outputs timing_name begin
                Flux.train!(loss_func, params, train_data, opt) 
            end
            timing_name = epoch > 1 ? "eval_epoch" : "eval_jit"
            @timeit time_outputs timing_name begin
                acc = accuracy(test_data, model)
                @info("epoch $epoch eval accuracy = $(acc)")
            end
        end
    end
    json_result["model_name"] = "MLP"
    json_result["framework_name"] = "Flux"
    json_result["dataset"] = "MNIST Digits"
    json_result["task"] = "classification"
    json_result["total_training_time"] = TimerOutputs.time(time_outputs["train_time"]) / 1e9
    json_result["average_epoch_training_time"] = TimerOutputs.time(time_outputs["train_time"]["train_epoch"]) / (9 * 1e9)
    num_batches = length(test_data)
    json_result["average_batch_inference_time"] = TimerOutputs.time(time_outputs["train_time"]["eval_epoch"]) / (9 * 1e6 * num_batches)
    json_result["final_training_loss"] = loss_func(xtest, ytest)
    json_result["final_evaluation_accuracy"] = accuracy(test_data, model)
    
    date_str = Dates.format(Dates.now(), "yyyy-mm-dd-HHMMSS")
    open("./output/m1-flux-mlp-$(date_str).json", "w") do f
        JSON.print(f, json_result)
    end
end

# Run main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

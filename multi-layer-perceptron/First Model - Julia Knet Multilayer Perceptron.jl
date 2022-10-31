using MLDatasets: MNIST
using Knet, IterTools, MLDatasets
using Dictionaries
using TimerOutputs
using TimerOutputs
using JSON
using Printf
using Knet:minibatch
using Knet:minimize
using Knet
using Knet: Param
using Knet: Knet, dir, accuracy, progress, sgd, gc, Data, nll, relu
using Flatten
using Flux.Data;
using Flux, Statistics

# This loads the MNIST handwritten digit recognition dataset. This code is based off the Knet Tutorial Notebook. 
xtrn,ytrn = MNIST.traindata(Float32)
xtst,ytst = MNIST.testdata(Float32)
println.(summary.((xtrn,ytrn,xtst,ytst)));

xtrn = reshape(xtrn, 784, 60000 ) 
xtst = reshape(xtst, 784, 10000 )
println(summary.((xtrn, xtst))) # can see the data that is flattened 

#Preprocessing targets: one hot vectors
# ytrn = onehotbatch(ytrn, 0:9)
# ytst = onehotbatch(ytst, 0:9)

train_loader = DataLoader((xtrn, ytrn), batchsize=128);
test_loader = DataLoader((xtst, ytst), batchsize = 128)

length(test_loader)



(x,y) = first(train_loader) #gives the first minibatch from training dataset
println.(summary.((x,y)));





struct Dense1; w; b; f; end
Dense1(i,o; f=relu) = Dense1(param(o,i), param0(o), f)
(d::Dense1)(x) = d.f.(d.w * mat(x) .+ d.b)

# Define a chain of layers and a loss function:
struct Chain; layers; end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)

model = Chain((Dense1(784, 100), Dense1(100, 10)))

model(x) #checking if training is working


loss(xtst, ytst) = nll(model(xtst), ytst)
evalcb = () -> (loss(xtst, ytst)) #function that will be called to get the loss 
const to = TimerOutput() # creating a TimerOutput, keeps track of everything


@timeit to "Train Total" begin
    for epoch in 1:10
        @timeit to "train_epoch" begin
            progress!(adam(model, train_loader; lr = 1e-3))
        end
    
        @timeit to "evaluation" begin
            accuracy(model, test_loader)
        end 
        @printf("epoch is %d, loss is %f, accuracy is %f", epoch, (evalcb()), accuracy(model, test_loader))
    end 
end 




final_train_loss = evalcb()
final_eval_accuracy = accuracy(model, test_loader)

# see the overall loss
println("Overall Loss: ", final_train_loss) 
println("Overall Accuracy: ", final_eval_accuracy ) #see the overall accuracy




show(to, allocations = true, compact = true) #see the time it took for training and evaluating the model



#average epoch training time converted to seconds from nanoseconds
average_train_epoch_time = (mean(TimerOutputs.time(to["Train Total"]["train_epoch"])))/(1e+9 *10)
total_train_time = TimerOutputs.time(to["Train Total"])/(1e+9)
average_batch_inference_time = TimerOutputs.time(to["Train Total"]["evaluation"])/(length(test_loader)*1e+9)

average_train_epoch_time



#getting dictionary to format the metrics
metrics = Dict("model_name" => "MLP",
 "framework_name"=>"Knet",
  "dataset" => "MNIST Digits", 
    "task" => "classifcation",
    "average_epoch_training_time" => average_train_epoch_time,
    "total_training_time" => total_train_time,
    "average_batch_inference_time" => average_batch_inference_time,
    "final_training_loss" => final_train_loss,
    "final_evaluation_accuracy" => final_eval_accuracy
)


stringdata = JSON.json(metrics)

#will allow the metrics to be entered into a file 

open("M1-Knet-mlp.json", "w") do f
    write(f, stringdata)
    end 

dict2 = Dict()
open("M1-Knet-mlp.json", "r") do f
    global dict2
    dict2 = JSON.parse(f)
end

pwd() #checking directory 











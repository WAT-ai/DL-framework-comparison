import torch 
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import json
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        nn.Flatten(start_dim=0),
    ])

    train_ds = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_ds = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

    train_dl = torch.utils.data.DataLoader(dataset=train_ds, batch_size=128, shuffle=True)
    test_dl = torch.utils.data.DataLoader(dataset=test_ds, batch_size=128, shuffle=False)
    return train_dl, test_dl


class MultiLayerPerceptron(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1 = nn.Linear(784, 100)
    self.fc2 = nn.Linear(100, 10)

  def forward(self, x):
    x = self.fc1(x)
    x = torch.relu(x)
    x = self.fc2(x)
    return x


def train_epoch(model, optimizer, criterion, train_dl, epoch):
    model.train()
    train_loss = 0
    tic_train_epoch = time.time()

    for data, target in train_dl:
    
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = criterion(output, target.to(device))
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item()
            
    train_loss = train_loss / len(train_dl)
    toc_train_epoch = time.time()
    time_per_epoch = toc_train_epoch - tic_train_epoch
    print('\nEpoch: {} \tTraining Loss: {:.6f} \tEpoch Time: {:.2f}s'.format(
    epoch+1, 
    train_loss,
    time_per_epoch,
    ))
    metrics = {
        "epoch_train_time": time_per_epoch,
        "epoch_train_loss": train_loss
    }
    return metrics


def eval_epoch(model, criterion, test_dl, epoch):
    model.eval()
    tic_eval = time.time()

    test_loss = 0    
    preds = []
    labels = []
    for data, target in test_dl:
        with torch.no_grad():
            output = model(data.to(device))
            loss = criterion(output, target.to(device))

            test_loss += loss.item()

            pred = torch.argmax(output, dim=-1).tolist()
            preds.extend(pred)
            labels.extend(target.tolist())
    toc_eval = time.time()

    acc = accuracy_score(labels, preds)
    print(f'Epoch: {epoch + 1 } \t Test Loss: {test_loss / len(test_dl):.6f} \tEpoch Time: {toc_eval - tic_eval:.2f}s')

    num_eval_batches = len(test_dl)

    average_batch_inference_time = (1000 * (toc_eval - tic_eval) / num_eval_batches) # in ms

    print("Average Batch Inference Time: {:.2f}ms".format(average_batch_inference_time))
    metrics = {
        "eval_accuracy": acc,
        "average_batch_inference_time": average_batch_inference_time
    }
    return metrics


def main():
    train_loader, test_loader = get_dataloaders()

    model = MultiLayerPerceptron().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001, betas=[0.9, 0.999])

    tic_total_train = time.time()
    train_times = []
    for epoch in range(10):
        train_metrics = train_epoch(model, optimizer, criterion, train_loader, epoch)
        eval_metrics = eval_epoch(model, criterion, test_loader, epoch)
        train_times.append(train_metrics["epoch_train_time"])
    toc_total_train = time.time()
    total_train_time = toc_total_train - tic_total_train

    print("\nTotal Training Time: {:.2f}s".format(total_train_time))
    print(f"Average epoch train time: {np.mean(train_times):.2f} s")

    metrics = {
        "model_name": "MLP",
        "framework_name": "PyTorch",
        "dataset": "MNIST Digits",
        "task": "classification",
        "total_training_time": total_train_time,         # in seconds
        "average_epoch_training_time": np.mean(train_times),  # in seconds
        "average_batch_inference_time": eval_metrics["average_batch_inference_time"],  # in milliseconds
        "final_training_loss": train_metrics["epoch_train_loss"],
        "final_evaluation_accuracy": eval_metrics["eval_accuracy"],
    }

    with open("m1-pytorch-mlp.json", "w") as outfile:
        json.dump(metrics, outfile)

if __name__ == "__main__":
    main()

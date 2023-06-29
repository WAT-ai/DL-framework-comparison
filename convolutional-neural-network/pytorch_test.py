import torch
import torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import random_split
from sklearn.metrics import accuracy_score
import time
import json
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
args = parser.parse_args()

torch.manual_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using torch with {device}")


def get_datasets(batch_size=32):
    # Transforms
    augment_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Pad((4, 4)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32))
    ])

    evaluate_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_size = 45_000
    val_size = 5_000

    train_data = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        transform=augment_pipeline,
        download=True
    )
    test_ds = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        transform=evaluate_pipeline,
        download=True
    )
    
    train_ds, val_ds = random_split(train_data, [train_size, val_size])
    val_ds.transform = evaluate_pipeline
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    return train_loader, val_loader, test_loader


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super(ResidualBlock, self).__init__()
        conv_kwargs = {
            "kernel_size": (3, 3),
            "padding": 1,  # To ensure 3x3 conv does not reduce image size. padding=1 also works
            "bias": False
        }
        self.stride = stride
        self.in_channels = in_channels
        self.channels = channels
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        # This conv is in_channels -> channels and applies stride
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, stride=stride, **conv_kwargs)
        self.bn2 = nn.BatchNorm2d(channels)
        # This conv is channels -> channels
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, **conv_kwargs)
    
    def strided_identity(self, x):
        # Downsample with 'nearest' method (this is striding if dims are divisible by stride)
        # Equivalently x = x[:, :, ::stride, ::stride].contiguous()
        if self.stride != 1:
            x = F.interpolate(x, mode='nearest', scale_factor=(1/self.stride))
        # Create padding tensor for extra channels
        if self.channels != self.in_channels:
            (b, c, h, w) = x.shape
            num_pad_channels = self.channels - self.in_channels
            pad = torch.zeros((b, num_pad_channels, h, w), device=x.device)
            # Append padding to the downsampled identity
            x = torch.cat((x, pad), dim=1)
        return x

    def forward(self, x):
        identity = self.strided_identity(x)
        z = self.bn1(x)
        z = self.relu(z)
        z = self.conv1(z)
        z = self.bn2(z)
        z = self.relu(z)
        z = self.conv2(z)
        out = identity + z
        return out
      

class ResNetV2(nn.Module):
  def __init__(self):
    super(ResNetV2, self).__init__()

    self.input_layer = nn.Conv2d(3, 16, (3,3), padding=1)

    self.layer_1 = ResidualBlock(16,16)
    self.layer_2 = ResidualBlock(16,16)
    self.layer_3 = ResidualBlock(16,16)

    self.layer_4 = ResidualBlock(16,32, stride=2)
    self.layer_5 = ResidualBlock(32,32)
    self.layer_6 = ResidualBlock(32,32)

    self.layer_7 = ResidualBlock(32,64, stride=2)
    self.layer_8 = ResidualBlock(64,64)
    self.layer_9 = ResidualBlock(64,64)

    self.pool = nn.AdaptiveAvgPool2d((1,1))
    self.output_layer = nn.Linear(64,10)


  def forward(self, x):
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
    out = out.flatten(1)
    out = out.view(out.size(0), -1)
    out = self.output_layer(out)
    return out


def train_epoch(model, criterion, optimizer, train_loader, epoch):
    metrics = {}
    running_loss = 0
    running_acc = 0
    
    start_epoch = time.time()
    model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()

        inputs = inputs.to(device)
        labels = labels.to(device)

        logits = model(inputs)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = logits.detach().cpu().numpy().argmax(axis=-1)
        running_acc += accuracy_score(labels.cpu(), preds)
    
    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_acc / len(train_loader)

    metrics["epoch_train_time"] = end_epoch - start_epoch
    metrics["loss"] = epoch_loss
    metrics["accuracy"] = epoch_acc
    
    print(f"Epoch {epoch + 1}: train loss = {epoch_loss:.4f}, train accuracy = {epoch_acc * 100:.2f}%, epoch time = {epoch_time:.2f} s")
    return metrics


def eval_epoch(model, criterion, val_loader, epoch):
    metrics = {}
    running_loss = 0
    all_preds = []
    all_labels = []
    
    start_epoch = time.time()
    model.eval()
    for inputs, labels in val_loader:
        with torch.no_grad():
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)

            running_loss += loss.item()
            preds = logits.detach().cpu().numpy().argmax(axis=-1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
    end_epoch = time.time()
    epoch_time = end_epoch - start_epoch
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    metrics["average_batch_inference_time"] = epoch_time / len(val_loader) * 1000  # In milliseconds
    metrics["loss"] = epoch_loss
    metrics["accuracy"] = epoch_acc
    
    print(f"Epoch {epoch + 1}: val loss = {epoch_loss:.4f}, val accuracy = {epoch_acc * 100:.2f}%, val time = {epoch_time:.2f} s")
    return metrics


def main():
    model = ResNetV2().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loader, val_loader, test_loader = get_datasets(batch_size=128)

    start_train = time.time()
    train_epoch_times = []
    eval_batch_times = []

    for epoch in range(10):
        train_metrics = train_epoch(model, loss_fn, optimizer, train_loader, epoch)
        val_metrics = eval_epoch(model, loss_fn, val_loader, epoch)

        train_epoch_times.append(train_metrics["epoch_train_time"])
        eval_batch_times.append(val_metrics["average_batch_inference_time"])
    end_train = time.time()

    test_metrics = eval_epoch(model, loss_fn, test_loader, 0)

    metrics = {
        "model_name": "ResNetV2-20",
        "framework_name": "PyTorch",
        "dataset": "CIFAR-10",
        "task": "classification",
        "total_training_time": end_train - start_train,
        "average_epoch_training_time": np.mean(train_epoch_times),
        "average_batch_inference_time": np.mean(eval_batch_times),
        "final_training_loss": train_metrics["loss"],
        "final_evaluation_accuracy": val_metrics["accuracy"],
        "final_test_accuracy": test_metrics["accuracy"]
    }

    print(metrics)

    date_str = time.strftime("%Y-%m-%d-%H%M%S")
    with open(f"./output/m2-pytorch-mlp-{date_str}.json", "w") as outfile:
        json.dump(metrics, outfile)


if __name__ == "__main__":
    main()
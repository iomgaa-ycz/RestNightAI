import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM
from FastAPI.Utils.utils import *

def train_Pose(model, train_loader, optimizer, arg):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    for i, (data, target) in enumerate(train_loader):
        data = data.to(arg["device"])
        target = target.to(arg["device"])
        data = data.float()
        optimizer.zero_grad()
        output = model(data)
        # Calculate cross entropy loss
        loss = cross_entropy_loss(output, target)

        loss.backward()
        optimizer.step()
        train_loss += loss.item() * output.size()[0]
        total += output.size()[0]

        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()

        if i % 10 == 0:
            print("train_loss: ", train_loss / total, "accuracy: ", 100 * correct / total)

    return train_loss / total, 100 * correct / total

def val_Pose(model, val_loader, arg):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(arg["device"])
        target = target.to(arg["device"])
        data = data.float()
        output = model(data)
        # Calculate cross entropy loss
        loss = cross_entropy_loss(output, target)

        val_loss += loss.item() * output.size()[0]
        total += output.size()[0]

        _, predicted = torch.max(output.data, 1)
        correct += (predicted == target).sum().item()

        if i % 10 == 0:
            print("val_loss: ", val_loss / total, "accuracy: ", 100 * correct / total)

    return val_loss / total, 100 * correct / total

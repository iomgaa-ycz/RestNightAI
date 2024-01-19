import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM
from FastAPI.Utils.utils import *

def train(model, train_loader, optimizer, arg):
    model.train()
    train_loss_l2 = 0
    train_loss_ssim = 0
    num = 0
    mse_loss = torch.nn.MSELoss()
    ssim_loss = SSIM(data_range=1, size_average=True, channel=1)
    for i, (data, target) in enumerate(train_loader):
        data = data.to(arg["device"])
        target = target.to(arg["device"])
        data = data.float()
        optimizer.zero_grad()
        output = model(data)
        # Calculate L2 loss
        l2_loss = mse_loss(output, data)

        # Calculate SSIM loss
        loss_ssim = 1 - ssim_loss(output, data)

        # Compare the weights of the two losses and scale them to a unified level
        loss = calculate_loss(l2_loss, loss_ssim)

        loss.backward()
        optimizer.step()
        train_loss_l2 += l2_loss.item() * output.size()[0]
        train_loss_ssim += loss_ssim.item() * output.size()[0]
        num += output.size()[0]
        if i % 10 == 0:
            print("train_loss_l2: ", train_loss_l2 / num, "train_loss_ssim: ", train_loss_ssim / num)
    return train_loss_l2 / num, train_loss_ssim / num,  (train_loss_l2 / num) *1.5 /2

def val(model, val_loader, optimizer, arg, rate):
    model.eval()
    val_loss_l2 = 0
    val_loss_ssim = 0
    accuracy_num = 0
    num = 0
    mse_loss = torch.nn.MSELoss()
    ssim_loss = SSIM(data_range=1, size_average=True, channel=1)
    for i, (data, target) in enumerate(val_loader):
        data = data.to(arg["device"])
        target = target.to(arg["device"])
        data = data.float()
        output = model(data)
        # Calculate L2 loss
        l2_loss = torch.zeros(data.size()[0], device=arg["device"])
        # Calculate SSIM loss
        loss_ssim = torch.zeros(data.size()[0], device=arg["device"])
        for j in range(data.size()[0]):
            l2_loss[j] = mse_loss(output[j].unsqueeze(0), data[j].unsqueeze(0))
            loss_ssim[j] = 1 - ssim_loss(output[j].unsqueeze(0), data[j].unsqueeze(0))

        val_loss_l2 += l2_loss.sum().item()
        val_loss_ssim += loss_ssim.sum().item()
        accuracy_num += calculate_acc(l2_loss, loss_ssim, target, rate).sum().item()
        num += output.size()[0]
        if i % 10 == 0:
            print("val_loss_l2: ", val_loss_l2 / num, "val_loss_ssim: ", val_loss_ssim / num, "accuracy: ", accuracy_num / num)
    return val_loss_l2 / num, val_loss_ssim / num , accuracy_num / num

import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM

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
        rate = l2_loss / loss_ssim
        if rate > 1000:
            loss = l2_loss + loss_ssim * 1000
        elif rate > 100:
            loss = l2_loss + loss_ssim * 100
        elif rate > 10:
            loss = l2_loss + loss_ssim * 10
        elif rate > 1:
            loss = l2_loss + loss_ssim
        elif rate > 0.1:
            loss = l2_loss * 10 + loss_ssim
        elif rate > 0.01:
            loss = l2_loss * 100 + loss_ssim
        else:
            loss = l2_loss * 1000 + loss_ssim

        loss.backward()
        optimizer.step()
        train_loss_l2 += l2_loss.item()
        train_loss_ssim += loss_ssim.item()
        num += output.size()[0]
        if i % 10 == 0:
            print("train_loss_l2: ", train_loss_l2 / num, "train_loss_ssim: ", train_loss_ssim / num)
    return train_loss_l2 / num, train_loss_ssim / num

def val(model, val_loader, optimizer, arg):
    model.eval()
    val_loss_l2 = 0
    val_loss_ssim = 0
    num = 0
    mse_loss = torch.nn.MSELoss()
    ssim_loss = SSIM(data_range=1, size_average=True, channel=1)
    for i, (data, target) in enumerate(val_loader):
        data = data.to(arg["device"])
        target = target.to(arg["device"])
        data = data.float()
        output = model(data)
        # Calculate L2 loss
        l2_loss = mse_loss(output, data)

        # Calculate SSIM loss
        loss_ssim = 1 - ssim_loss(output, data)

        val_loss_l2 += l2_loss.item()
        val_loss_ssim += loss_ssim.item()
        num += output.size()[0]
        if i % 10 == 0:
            print("val_loss_l2: ", val_loss_l2 / num, "val_loss_ssim: ", val_loss_ssim)
    return val_loss_l2 / num, val_loss_ssim / num

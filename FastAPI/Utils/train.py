import torch
import torch.nn.functional as F
from pytorch_msssim import SSIM
from FastAPI.Utils.loss import EntropyLossEncap

def train(model, train_loader, optimizer, arg):
    model.train()
    train_loss_l2 = 0
    train_loss_ssim = 0
    train_loss_entropy = 0  # Added variable for entropy loss
    num = 0
    mse_loss = torch.nn.MSELoss()
    ssim_loss = SSIM(data_range=1, size_average=True, channel=1)
    tr_entropy_loss = EntropyLossEncap()
    for i, (data, target) in enumerate(train_loader):
        data = data.to(arg["device"])
        target = target.to(arg["device"])
        data = data.float()
        optimizer.zero_grad()
        output,att = model(data)
        # Calculate L2 loss
        l2_loss = mse_loss(output, data)

        # Calculate entropy loss
        entropy_loss = tr_entropy_loss(att)

        # Compare the weights of the two losses and scale them to a unified level
        rate = l2_loss / entropy_loss
        if rate > 1000:
            loss = l2_loss + entropy_loss * 1000
        elif rate > 100:
            loss = l2_loss + entropy_loss * 100
        elif rate > 10:
            loss = l2_loss + entropy_loss * 10
        elif rate > 1:
            loss = l2_loss + entropy_loss
        elif rate > 0.1:
            loss = l2_loss * 10 + entropy_loss
        elif rate > 0.01:
            loss = l2_loss * 100 + entropy_loss
        else:
            loss = l2_loss * 1000 + entropy_loss

        loss.backward()
        optimizer.step()
        train_loss_l2 += l2_loss.item()
        train_loss_entropy += entropy_loss.item()  # Accumulate entropy loss
        num += output.size()[0]
        if i % 10 == 0:
            print("train_loss_l2: ", train_loss_l2 / num, "train_loss_entropy: ", train_loss_entropy / num)
    return train_loss_l2 / num, train_loss_entropy / num

def val(model, val_loader, optimizer, arg):
    model.eval()
    val_loss_l2 = 0
    val_loss_entropy = 0  # Added variable for entropy loss
    num = 0
    mse_loss = torch.nn.MSELoss()
    ssim_loss = SSIM(data_range=1, size_average=True, channel=1)
    tr_entropy_loss = EntropyLossEncap()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(arg["device"])
        target = target.to(arg["device"])
        data = data.float()
        output,att = model(data)
        # Calculate L2 loss
        l2_loss = mse_loss(output, data)

        # Calculate entropy loss
        entropy_loss = tr_entropy_loss(att)

        val_loss_l2 += l2_loss.item()
        val_loss_entropy += entropy_loss.item()  # Accumulate entropy loss
        num += output.size()[0]
        if i % 10 == 0:
            print("val_loss_l2: ", val_loss_l2 / num, "val_loss_entropy: ", val_loss_entropy)
    return val_loss_l2 / num, val_loss_entropy / num

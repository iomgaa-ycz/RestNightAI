import torch
import torch.nn.functional as F

def train(model, train_loader, optimizer, arg):
    model.train()
    train_loss = 0
    num = 0
    mse_loss = torch.nn.MSELoss()
    for i, (data,target) in enumerate(train_loader):
        data = data.float()
        optimizer.zero_grad()
        output = model(data)
        # Calculate L2 loss
        l2_loss = mse_loss(output, data)
        l2_loss.backward()
        optimizer.step()
        train_loss += l2_loss.item()
        num += output.size()[0]
        if i%10 == 0:
            print("train_loss: ", train_loss / num)
    return train_loss / num

def val(model, val_loader, optimizer, arg):
    model.eval()
    val_loss = 0
    num = 0
    mse_loss = torch.nn.MSELoss()
    for i, (data,target) in enumerate(val_loader):
        data = data.float()
        output = model(data)
        # Calculate L2 loss
        l2_loss = mse_loss(output, data)
        val_loss += l2_loss.item()
        num += output.size()[0]
        if i%10 == 0:
            print("val_loss: ", val_loss / num)
    return val_loss / num
import torch
import torch.nn as nn

def split_list(lst, rate):
    index = int(len(lst) * rate)
    list1 = lst[:index]
    list2 = lst[index:]
    
    if not list1:
        list1 = list2[-1:]
        list2 = list2[:-1]
    elif not list2:
        list2 = list1[-1:]
        list1 = list1[:-1]
    
    return list1, list2


def initialize_layers(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    return model

def calculate_loss(l2_loss, loss_ssim):
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
    return loss

def calculate_acc(l2_loss, loss_ssim, pose, rate):
    positive_samples = torch.zeros_like(l2_loss)
    loss_samples = torch.zeros_like(l2_loss)
    # for i in range(l2_loss.size(0)):
    #     loss_samples[i] = calculate_loss(l2_loss[i], loss_ssim[i])
    
    positive_samples += torch.where((l2_loss > rate) & (pose == 1), torch.ones_like(positive_samples), torch.zeros_like(positive_samples))
    positive_samples += torch.where((l2_loss < rate) & (pose == 0), torch.ones_like(positive_samples), torch.zeros_like(positive_samples))
    
    return positive_samples
    

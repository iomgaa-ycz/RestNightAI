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


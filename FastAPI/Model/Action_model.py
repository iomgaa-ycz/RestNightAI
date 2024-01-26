import torch
import torch.nn as nn

from FastAPI.Model.ResNet import *
from FastAPI.Model.Decoder import *

class Action_model(nn.Module):
    def __init__(self, hypers=None):
        super(Action_model, self).__init__()

        encoder_params = hypers['encoder']
        decoder_params = hypers['decoder']
        encoder_resnet_params = encoder_params['resnet1_params']
        decoder_resnet_params = decoder_params['resnet1_params']

        self.action = ResNet(
            channels=encoder_resnet_params['channels'],
            in_channel=encoder_resnet_params['in_channel'],
            num_blocks=encoder_resnet_params['num_blocks'],
            strides=encoder_resnet_params['strides'])
        
        self.lstm = nn.LSTM(48*5*10, 256, batch_first=True)
        self.fc = nn.Linear(256, 3)
        
    def forward(self, x):
        n,c,w,h = x.shape
        x = x.reshape(n*c,1,w,h)
        x = self.action(x)
        x = x.view(n,c, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

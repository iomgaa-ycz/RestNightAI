import torch
import torch.nn as nn

from FastAPI.Model.ResNet import *
from FastAPI.Model.Decoder import *

class Pose_model(nn.Module):
    def __init__(self, hypers=None):
        super(Pose_model, self).__init__()

        encoder_params = hypers['encoder']
        decoder_params = hypers['decoder']
        encoder_resnet_params = encoder_params['resnet1_params']
        decoder_resnet_params = decoder_params['resnet1_params']

        self.pose = ResNet(
            channels=encoder_resnet_params['channels'],
            in_channel=encoder_resnet_params['in_channel'],
            num_blocks=encoder_resnet_params['num_blocks'],
            strides=encoder_resnet_params['strides'])
        
        self.head = nn.Sequential(
            nn.Linear(48*5*10, 256),
            nn.ReLU(),
            nn.Dropout(0.7),  # Added dropout with coefficient 0.5
            nn.Linear(256, 4)
        )
        self.flat = nn.Flatten()
        
    def forward(self, x):
        x = self.pose(x)
        x = self.flat(x)
        x = self.head(x)
        return x

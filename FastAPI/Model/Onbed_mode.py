import torch
import torch.nn as nn

from FastAPI.Model.ResNet import *
from FastAPI.Model.Decoder import *

class Onbed_model(nn.Module):
    def __init__(self, hypers=None):
        super(Onbed_model, self).__init__()

        pose_params = hypers['pose']
        resnet_params = pose_params['resnet_params']
        linear_params = pose_params['linear_params']
        decoder_params = pose_params['decoder_params']

        self.pose = ResNet(
            channels=resnet_params['channels'],
            in_channel=resnet_params['in_channel'],
            num_blocks=resnet_params['num_blocks'],
            strides=resnet_params['strides'])
        
        self.decoder = ResNetDecoder(
            channels=decoder_params['channels'],
            out_channel=decoder_params['out_channel'],
            num_blocks=decoder_params['num_blocks'],
            strides=decoder_params['strides'])
    def forward(self, x):
        x = self.pose(x)
        x = self.decoder(x)
        return x

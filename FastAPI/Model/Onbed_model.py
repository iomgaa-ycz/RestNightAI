import torch
import torch.nn as nn

from FastAPI.Model.ResNet import *
from FastAPI.Model.Decoder import *

class Onbed_model(nn.Module):
    def __init__(self, hypers=None):
        super(Onbed_model, self).__init__()

        encoder_params = hypers['encoder']
        decoder_params = hypers['decoder']
        encoder_resnet_params = encoder_params['resnet1_params']
        decoder_resnet_params = decoder_params['resnet1_params']

        self.pose = ResNet(
            channels=encoder_resnet_params['channels'],
            in_channel=encoder_resnet_params['in_channel'],
            num_blocks=encoder_resnet_params['num_blocks'],
            strides=encoder_resnet_params['strides'])
        
        self.decoder = ResNetDecoder(
            channels=decoder_resnet_params['channels'],
            out_channel=decoder_resnet_params['out_channel'],
            num_blocks=decoder_resnet_params['num_blocks'],
            strides=decoder_resnet_params['strides'])
    def forward(self, x):
        x = self.pose(x)
        x = self.decoder(x)
        return x

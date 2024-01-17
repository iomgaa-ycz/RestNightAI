import torch.nn as nn

class BasicBlockDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, output_padding=0):
        super(BasicBlockDecoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=2, output_padding=output_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        return out

class ResNetDecoder(nn.Module):
    def __init__(self, num_blocks=None, channels=None, out_channel=None, strides=None):
        super(ResNetDecoder, self).__init__()

        self.in_channels = channels[-1]

        self.layer4 = self._make_layer(channels[2], num_blocks[3], stride=strides[4], output_padding=1)
        self.layer3 = self._make_layer(channels[1], num_blocks[2], stride=strides[3], output_padding=1)
        self.layer2 = self._make_layer(channels[0], num_blocks[1], stride=strides[2], output_padding=1)
        self.layer1 = self._make_layer(out_channel, num_blocks[0], stride=strides[1], output_padding=1)

        self.conv1 = nn.ConvTranspose2d(out_channel, 1, kernel_size=4, stride=strides[0], padding=1, output_padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)

    def _make_layer(self, out_channels, num_blocks, stride, output_padding):
        layers = []
        for _ in range(num_blocks):
            layers.append(BasicBlockDecoder(self.in_channels, out_channels, stride, output_padding))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = self.relu(self.bn1(self.conv1(x)))
        return x

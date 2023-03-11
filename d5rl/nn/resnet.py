import torch
import torch.nn as nn

# simplified version from torchvision: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
# removed support for groups, dilations, wide variants, last mapping to classes


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, blocks, img_channels=3):
        super().__init__()
        self._in_channels = 64

        self.conv1 = nn.Conv2d(img_channels, self._in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self._in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(block, 64, blocks[0])
        self.layer2 = self.__make_layer(block, 128, blocks[1], stride=2)
        self.layer3 = self.__make_layer(block, 256, blocks[2], stride=2)
        self.layer4 = self.__make_layer(block, 512, blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_layer(self, block, hidden_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self._in_channels != hidden_channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self._in_channels, hidden_channels * block.expansion, stride),
                nn.BatchNorm2d(hidden_channels * block.expansion)
            )

        layers = []
        layers.append(
            block(self._in_channels, hidden_channels, stride, downsample)
        )
        self._in_channels = hidden_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self._in_channels, hidden_channels)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # now sure that we really need first conv with filter size 7 + maxpool
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        return out


class ResNet18(ResNet):
    def __init__(self, img_channels=3):
        super().__init__(BasicBlock, [2, 2, 2, 2], img_channels=img_channels)


class ResNet34(ResNet):
    def __init__(self, img_channels=3):
        super().__init__(BasicBlock, [3, 4, 6, 3], img_channels=img_channels)


class ResNet50(ResNet):
    def __init__(self, img_channels=3):
        super().__init__(BottleneckBlock, [3, 4, 6, 3], img_channels=img_channels)


class ResNet101(ResNet):
    def __init__(self, img_channels=3):
        super().__init__(BottleneckBlock, [3, 4, 23, 3], img_channels=img_channels)


class ResNet152(ResNet):
    def __init__(self, img_channels=3):
        super().__init__(BottleneckBlock, [3, 8, 36, 3], img_channels=img_channels)
import torch
import torch.nn as nn
import torch.nn.functional as F

# ResNet models for CIFAR. Adapted from
# 1. https://github.com/elephantmipt/compressors/blob/master/compressors/models/cv/resnet_cifar.py
# 2. https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
# 3. https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py


# TODO: add support for GroupNorm normalization (as we might use small batches on V100)
def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_channels, hidden_channels, stride)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = conv3x3(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.downsample = nn.Identity()
        if stride != 1 or in_channels != hidden_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, hidden_channels, stride),
                nn.BatchNorm2d(hidden_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(x)

        out += residual
        out = F.relu(out, inplace=True)

        return out


class ResNet(nn.Module):
    def __init__(self, depth, num_filters, img_channels=3, out_dim=1024):
        super().__init__()
        assert (depth - 2) % 9 == 0, "Depth should be 9n+2, e.g. 11, 20, 29, 38, 47, 56, 110, 1199"
        n = (depth - 2) // 9
        # TODO: maybe in the future we should add support for the Bottleneck style block,
        #  but in the original work only Basic block is used for cifar
        self.__in_channels = num_filters[0]

        self.conv1 = nn.Conv2d(img_channels, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])

        self.layer1 = self.__make_layer(BasicBlock, num_filters[1], n)
        self.layer2 = self.__make_layer(BasicBlock, num_filters[2], n, stride=2)
        self.layer3 = self.__make_layer(BasicBlock, num_filters[3], n, stride=2)
        # this layer is an addition to the standard CIFAR type model,
        # as we need higher downsampling for 24x80 nethack img -> 3x10
        self.layer4 = self.__make_layer(BasicBlock, num_filters[4], n, stride=2)

        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_filters[4] * 3 * 10, out_dim)

        # TODO: add optional identity init for residual as in original pytorch resnet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def __make_layer(self, block, hidden_channels, blocks, stride=1):
        layers = [
            block(self.__in_channels, hidden_channels, stride)
        ]
        self.__in_channels = hidden_channels
        for i in range(1, blocks):
            layers.append(block(self.__in_channels, hidden_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        assert x.shape[-2:] == (24, 80), "for now, this encoder only for full nethack img"
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # for 24x80 final would be 64x3x10
        # x = self.avgpool(x)

        # would be of shape [batch_size, 3840 * k],
        # if this is too much, we should allow stride=2 for first block
        x = x.flatten(1)
        x = self.fc(x)

        return x


# Additionally support wide variants from "Wide Residual Networks": https://arxiv.org/pdf/1605.07146.pdf
class ResNet11(ResNet):
    def __init__(self, img_channels, out_dim, k=1):
        super().__init__(11, [16, 16 * k, 32 * k, 64 * k, 128 * k], img_channels, out_dim)


class ResNet20(ResNet):
    def __init__(self, img_channels, out_dim, k=1):
        super().__init__(20, [16, 16 * k, 32 * k, 64 * k, 128 * k], img_channels, out_dim)


class ResNet38(ResNet):
    def __init__(self, img_channels, out_dim, k=1):
        super().__init__(38, [16, 16 * k, 32 * k, 64 * k, 128 * k], img_channels, out_dim)


class ResNet56(ResNet):
    def __init__(self, img_channels, out_dim, k=1):
        super().__init__(56, [16, 16 * k, 32 * k, 64 * k, 128 * k], img_channels, out_dim)


class ResNet110(ResNet):
    def __init__(self, img_channels, out_dim, k=1):
        super().__init__(110, [16, 16 * k, 32 * k, 64 * k, 128 * k], img_channels, out_dim)


if __name__ == "__main__":
    test_img = torch.randn(8, 3, 24, 80)
    model = ResNet11(3, 1024, k=1)

    print(model(test_img).shape)





import torch.nn as nn

__all__ = ['ResNeXt', 'resnext29']


class BottleneckC(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, cardinality=16):
        super(BottleneckC, self).__init__()
        self.mult = 4
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, self.mult * out_channels,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.mult * out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != self.mult * out_channels:
            self.downsample.add_module('conv_downsample',
                                       nn.Conv2d(in_channels,
                                                 self.mult * out_channels,
                                                 kernel_size=1,
                                                 stride=stride,
                                                 bias=False))
            self.downsample.add_module('bn_downsample',
                                       nn.BatchNorm2d(self.mult * out_channels))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        residual = self.downsample(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        return self.relu(out + residual)


class ResNeXt(nn.Module):
    '''
    ResNeXt for CIFAR-10 and CIFAR-100, as described in
    "Aggregated Residual Transformations for Deep Neural Networks" by Xie et al.
    '''

    def __init__(self, layers, num_classes=10):
        super(ResNeXt, self).__init__()
        self.mult = 4
        self.in_channels = 64
        self.relu = nn.ReLU(inplace=True)
        self.conv0 = nn.Conv2d(3, self.in_channels, 3, stride=1, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(self.in_channels)
        self.layer1 = self._create_layer(64, layers[0])
        self.layer2 = self._create_layer(128, layers[1], stride=2)
        self.layer3 = self._create_layer(256, layers[2], stride=2)
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(self.mult * 256, num_classes)

    def _create_layer(self, channels, num_blocks, stride=1):
        layer = nn.Sequential()
        for i in range(num_blocks):
            if i == 0:
                layer.add_module(f'bottleneck{i}', BottleneckC(self.in_channels,
                                                               channels, stride=stride))
                self.in_channels = self.mult * channels
            else:
                layer.add_module(f'bottleneck{i}', BottleneckC(self.in_channels,
                                                               channels))
        return layer

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnext29(**kwargs):
    model = ResNeXt(layers=[3, 3, 3], **kwargs)
    return model

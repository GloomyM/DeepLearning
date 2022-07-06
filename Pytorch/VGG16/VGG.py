import torch
import json
import torch.nn as nn
from torchvision import transforms


class VGG(nn.Module):
    def __init__(self, netType='vgg16', num_classes=10):
        super(VGG, self).__init__()
        config = parse(netType)
        self.layers = self.init_layers(config=config)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        # x = torch.flatten(x)
        x = x.view(x.size(0), -1)  # 注意这里是按照batch_size个数展开
        x = self.fc(x)
        return x

    def init_layers(self, config: dict):
        input_channel = 1
        layers = nn.Sequential()
        for item in config:
            if item == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(
                    nn.Conv2d(in_channels=input_channel, out_channels=int(item), kernel_size=(3, 3), padding=1))
                layers.append(nn.ReLU(True))
                input_channel = int(item)
        return layers


def parse(netType):
    with open('config.json') as conf:
        data = json.load(conf)
    return data[netType]

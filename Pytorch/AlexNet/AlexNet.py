import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # 卷积层
        self.layers = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 96, kernel_size=(3, 3)),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第二层
            nn.Conv2d(96, 256, kernel_size=(3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # 第三层
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(True),

            # 第四层
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(True),

            # 第五层
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.Dropout(0.5),
            nn.Linear(2048, 10)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

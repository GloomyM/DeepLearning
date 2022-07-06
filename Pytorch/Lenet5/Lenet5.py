import torch.nn as nn


class Lenet5(nn.Module):
    def __init__(self):
        super(Lenet5, self).__init__()
        self.layers = nn.Sequential(
            # 第一层
            nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二层
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 16, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)  # 展开为一维
        x = self.fc(x)
        return x

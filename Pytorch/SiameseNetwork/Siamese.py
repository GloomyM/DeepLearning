import torch.nn as nn


class Siamese(nn.Module):

    def __init__(self):
        super(Siamese, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=(5, 5), stride=(3, 3), padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=4, stride=2),
            nn.Dropout(0.5)
        )

        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Linear(128, 2)
        )

    def forward_once(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, first_img, second_img):
        y1 = self.forward_once(first_img)
        y2 = self.forward_once(second_img)
        return y1, y2

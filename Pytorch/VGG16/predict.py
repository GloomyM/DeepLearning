import torch.utils.data
import torch.nn as nn
import numpy as np
import os
from VGG import VGG
from torchvision import datasets
from torchvision import transforms

data_transformer = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()
])
device = torch.device("mps")

if __name__ == '__main__':
    if torch.has_mps:
        print("Yes")
    import platform
    print(platform.platform())
    download = False
    if 'res' not in os.listdir():
        download = True
    train_set = datasets.MNIST(root='./res/data', transform=data_transformer, train=True, download=download)
    test_set = datasets.MNIST(root='./res/data', transform=data_transformer, train=False, download=download)

    train_set = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
    test_set = torch.utils.data.DataLoader(dataset=test_set, batch_size=64)
    print(test_set.dataset)
    vgg = VGG('vgg16').to(device)
    print(vgg)

    optimizer = torch.optim.Adam(vgg.parameters(), lr=0.05)
    loss_function = nn.CrossEntropyLoss()
    epochs = 10

    for epoch in range(epochs):
        sum_loss = 0.0
        for i, data in enumerate(train_set):
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # 将数据梯度归零
            outputs = vgg(inputs)  # 前向传播预测值
            loss = loss_function(outputs, labels)
            loss.backward()  # 反向传播
            optimizer.step()  # 根据梯度下降进行数据更新
            sum_loss += loss.item()  # 求总的loss
            if i % 100 == 99:
                print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch + 1, i + 1, sum_loss / 100))


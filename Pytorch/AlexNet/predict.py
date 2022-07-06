import os

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets
import torch.utils.data
from AlexNet import AlexNet

device = torch.device("mps")

if __name__ == '__main__':
    download = False

    if 'res' not in os.listdir():
        download = True

    train_set = datasets.CIFAR10(root='./res/data', transform=transforms.ToTensor(), train=True, download=download)
    test_set = datasets.CIFAR10(root='./res/data', transform=transforms.ToTensor(), train=False, download=download)
    train_set = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True, num_workers=4)

    test_set = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=4)
    alexNet = AlexNet().to(device)

    optimize = torch.optim.Adam(alexNet.parameters(), lr=0.01)
    loss_function = nn.CrossEntropyLoss()
    epochs = 5
    for epoch in range(epochs):
        loss_sum = 0.0
        for i, data in enumerate(train_set):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimize.zero_grad()
            outputs = alexNet(inputs)
            loss_per = loss_function(outputs, labels)
            loss_per.backward()
            optimize.step()
            loss_sum += loss_per.item()

            if i % 100 == 99:
                print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch + 1, i + 1, loss_sum / 100))
                loss_sum = 0.0

        total, right = 0, 0
        for i, data in enumerate(test_set):
            test_inputs, test_labels = data
            test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
            test_outputs = alexNet(test_inputs)
            test_outputs = torch.max(test_outputs.data, 1)[1]
            total += test_outputs.size(0)
            right += (test_labels == test_outputs).sum()
        print("第{}轮的准确率为:{:.2f}%".format(epoch + 1, 100.0 * right.item() / total))


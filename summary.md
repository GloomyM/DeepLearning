# 总结

### 关于为什么卷积可以实现对输入的矩阵进行降维：

在$$Pytorch$$中，卷积函数往往都有一个参数叫$$kernal\_size$$也就是卷积核的大小，举一个最常见的例子：

输入的图片大小为$$3*224*224$$，这时如果我们使用一个$$kernal\_size$$为3，步长为2，$$padding$$为1的卷积核对输入的图像进行卷积运算。根据公式$$outSize=(inputSize + 2*p - kernal\_size + 1) / stride$$，因此输出后的大小变为$$112*122$$，但是通道数是可以任意设定的，因为前面定义的卷积核大小实际是$$3*3*3$$，第一个3表示通道数，必须保证与输入的通道数一致，后面的两个3分别表示宽和高。假设我们需要64个$$3*3*3$$的卷积核那么它的实际尺寸是$$64 * 3 * 3 * 3$$，64表示卷积核个数，第一个3表示输入的通道数，后面两个3表示宽和高。因此我们在经过64个卷积核卷积后得到的就是$$64*112*112$$，64即位通道数，本质上输出的通道数就是卷积核的大小。

### 关于Pytorch中卷积层以及池化层的相关操作

在Pytorch中已经实现了卷积以及池化的相关操作，分别是：

- **二维卷积**

  ```python
  torch.nn.Conv2d(in_channels: int, # 输入的通道数
  				out_channels: int, # 输出通道数
          kernel_size: _size_2_t, # 卷积核的大小
          stride: _size_2_t = 1, # 步长（默认为1）
  				padding: _size_2_t = 0, # 填充（默认为0）
          groups: int = 1,
          bias: bool = True,
          padding_mode: str = 'zeros')
  ```

- **最大池化**

  ```python
  torch.nn.MaxPool2d(kernel_size: _size_any_t, # 池化窗口的大小
  				stride: Optional[_size_any_t] = None, # 步长
          padding: _size_any_t = 0) # 填充（默认为0）
  ```

- **平均池化**

  ```python
  torch.nn.AvgPool2d(kernel_size: _size_any_t, # 池化窗口的大小
  				stride: Optional[_size_any_t] = None, # 步长
          padding: _size_any_t = 0) # 填充（默认为0）
  ```

### 关于如何利用Pytorch自定义模型

​        在实际应用中，往往需要自定义各种网络模型来满足实际需要，举最简单的Lenet5网络模型为例，下面利用Pytorch建立Lenet5网络，并对MNIST手写数字识别数据集进行训练以及测试。        

​		第一步，定义网络模型，通过继承nn.Module，然后重写forward函数进行前向传播，具体代码如下：

```python
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

```

​		第二步，加载数据集，如果使用的是一些常用的数据集例如MNIST，CIFAR10等，可以直接使用pytorch进行加载，通过dataset加载数据集再通过dataloader加载dataset，相关代码如下：

```python
train_set = datasets.CIFAR10(root='./res/data', transform=transforms.ToTensor(), train=True, download=download)
test_set = datasets.CIFAR10(root='./res/data', transform=transforms.ToTensor(), train=False, download=download)
train_set = torch.utils.data.DataLoader(dataset=train_set, batch_size=64, shuffle=True)
test_set = torch.utils.data.DataLoader(dataset=test_set, batch_size=64, shuffle=True)
```

​		第三步，训练+测试模型，对于训练模型首先要定义前面生成的模型，以及损失函数和优化器，常用的损失函数有交叉熵等，常用的优化器有梯度下降和Adam等(Adam效果更好)，然后就是针对每次训练的结果进行反向传播最后测试，相关代码如下：

```python
device = torch.device("mps")  # mac选择gpu加速
lenet5 = Lenet5().to(device)
loss_function = nn.CrossEntropyLoss()  # 定义损失函数（交叉熵）
optimizer = torch.optim.Adam(lenet5.parameters(), lr=0.01)  # 使用adam进行反向传播
epochs = 5
for epoch in range(epochs):

    # 训练训练集
    loss = 0
    for i, data in enumerate(train_set):
        inputs, labels = data  # 获取参数和对应的标签
        inputs, labels = inputs.to(device), labels.to(device)  # gpu
        optimizer.zero_grad()  # 重置优化器为0

        output = lenet5(inputs)  # 计算预测值
        loss_per = loss_function(output, labels)  # 计算损失值
        loss_per.backward()  # 反向传播
        optimizer.step()

        loss += loss_per.item()  # 计算每次的损失和

        if i % 100 == 99: # 每100个batch输出一次，每个batch包含64张图片
            print('[Epoch:%d, batch:%d] train loss: %.03f' % (epoch + 1, i + 1, loss / 100))
            loss = 0.0

    # 对测试集进行测试
    total, right = 0, 0
    for data in test_set:
        test_inputs, test_labels = data
        test_inputs, test_labels = test_inputs.to(device), test_labels.to(device)
        output_test = lenet5(test_inputs)
        output_predict = torch.max(output_test.data, 1)[1]
        total += test_labels.size(0)
        temp = (output_predict == test_labels).sum()
        right += temp
    print("第{}轮的准确率为:{:.2f}%".format(epoch + 1, 100.0 * right.item() / total))
```

### 关于对HighWay Network的理解
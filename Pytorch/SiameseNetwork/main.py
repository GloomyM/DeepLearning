import torch
from torch.utils.data import DataLoader
from Minst import Minst
from Siamese import Siamese
from ContrastiveLoss import ContrastiveLoss
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text is not None:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(text)
    plt.show()


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


if __name__ == '__main__':
    train_set = DataLoader(Minst(path='dataset/', isTrain=True), batch_size=64, num_workers=4, shuffle=True)
    test_set = DataLoader(Minst(path='dataset/', isTrain=False), shuffle=True)
    device = torch.device('mps')
    siamese = Siamese().to(device)
    optim = torch.optim.Adam(siamese.parameters(), lr=1e-3)
    loss_func = ContrastiveLoss()
    epochs = 20
    loss_list = []
    for epoch in range(epochs):
        loss_sum = 0.0
        for i, data in enumerate(train_set, 0):
            first_img, second_img, label = data
            first_img = first_img.to(device)
            second_img = second_img.to(device)
            label = label.to(device)
            optim.zero_grad()
            y0, y1 = siamese(first_img, second_img)
            loss = loss_func(y0, y1, label)
            loss.backward()
            loss_sum += loss
            optim.step()
            if i % 100 == 0:
                print("Epoch {}: Batch: {} Current Loss:{}".format(epoch + 1, i, loss.item()))
                loss_list.append(loss.item)

    siamese.eval()
    cnt_positive, cnt_negative = 0, 0
    for i, data in enumerate(test_set, 0):
        first_img, second_img, label = data
        concat_imgs = torch.cat((first_img, second_img), dim=0)
        first_img, second_img = first_img.to(device), second_img.to(device)
        y1, y2 = siamese(first_img, second_img)
        if label == torch.FloatTensor([[0]]):
            cnt_negative += 1
        else:
            cnt_positive += 1
        dist = F.pairwise_distance(y1, y2)
        print(dist.item())
        imshow(make_grid(concat_imgs), f'Dissimilarity:{dist.item():.4f}')
        if cnt_negative >= 2 and cnt_positive >= 2:
            break

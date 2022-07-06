import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision import datasets


class Minst(Dataset):
    def __init__(self, path, isTrain=True, columns=['first_img', 'second_img', 'label'],
                 transform=transforms.ToTensor()):
        self.data = datasets.ImageFolder(root="./data/faces/training/")
        self.mode = 'train' if isTrain else 'test'
        self.path = path
        self.transform = transform
        self.columns = columns
        self.data = pd.read_csv(self.path + "/" + str(self.mode) + ".csv", names=self.columns)

    def __getitem__(self, index):
        first_img_path = str(self.path) + str(self.data.at[index, 'first_img'])
        second_img_path = str(self.path) + str(self.data.at[index, 'second_img'])
        first_img = Image.open(first_img_path).convert('L')
        second_img = Image.open(second_img_path).convert('L')
        first_img = self.transform(first_img)
        second_img = self.transform(second_img)
        return first_img, second_img, torch.tensor([self.data.at[index, 'label']], dtype=torch.float32)

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    data = Minst(path='dataset', isTrain=True)
    print(len(data))

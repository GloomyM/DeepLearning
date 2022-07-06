import os
import random


class DataGenerator:

    def __init__(self, train_size=0.8, path='dataset'):
        self.path = path
        self.train_size = train_size

    def generate(self):
        data = os.listdir(self.path + "/images")
        countN = {}
        for i in range(10):
            countN[i] = 0
        for item in data:
            countN[int(item[0])] += 1
        print(countN)
        size_split = int(len(data) * self.train_size)
        train_set, test_set = data[:size_split], data[size_split:]
        # print(len(train_set))
        # print(len(test_set))
        with open(self.path + "/train.csv", 'w') as train_write:
            cnt_positive = 0
            cnt_negative = 0
            while cnt_positive < 200:
                first_img, second_img = random.sample(train_set, 2)
                if first_img[0] == second_img[0]:
                    label = 1
                    cnt_positive += 1
                    train_write.write("images/" + first_img + ",images/" + second_img + "," + str(label) + "\n")
            while cnt_negative < 200:
                first_img, second_img = random.sample(train_set, 2)
                if first_img[0] != second_img[0]:
                    label = 0
                    cnt_negative += 1
                    train_write.write("images/" + first_img + ",images/" + second_img + "," + str(label) + "\n")

        with open(self.path + "/test.csv", 'w') as test_write:
            for _ in range(100):
                first_img, second_img = random.sample(test_set, 2)
                label = 1 if first_img[0] == second_img[0] else 0
                test_write.write("images/" + first_img + ",images/" + second_img + "," + str(label) + "\n")


if __name__ == '__main__':
    g = DataGenerator(train_size=0.8)
    g.generate()

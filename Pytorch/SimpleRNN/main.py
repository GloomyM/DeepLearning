import random

import torch
from torchtext.datasets import IMDB
from torchtext import data
from SimpleRNN import SimpleRNN


def binary_accuracy(preds, y):

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in iterator:
        optimizer.zero_grad()

        predictions = model(batch.text).squeeze(1)

        loss = criterion(predictions, batch.label)

        acc = binary_accuracy(predictions, batch.label)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)

            loss = criterion(predictions, batch.label)

            acc = binary_accuracy(predictions, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


batch_size = 64
device = torch.device('mps')
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype=torch.float)

train_set, test_set = IMDB.splits(TEXT, LABEL)
train_set, valid_set = train_set.split(random_state=random.seed(2022))
print("训练集大小: {}".format(len(train_set)))
print("测试集大小: {}".format(len(test_set)))
MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_set, max_size=MAX_VOCAB_SIZE)  # 建立vocabulary，最多25000个token
LABEL.build_vocab(train_set)
print(TEXT.vocab.freqs.most_common(20))  # 输出词频最高的前20个token
simpleRnn = SimpleRNN(len(TEXT.vocab), 100, 256, 1).to(device)
optimizer = torch.optim.Adam(simpleRnn.parameters(), lr=1e-3)
loss_func = torch.nn.BCEWithLogitsLoss().to(device)

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((train_set, valid_set, test_set),
                                                                           batch_size=64,
                                                                           device=device)

epochs = 5

best_valid_loss = float('inf')

for epoch in range(epochs):
    train_loss, train_acc = train(simpleRnn, train_iterator, optimizer, loss_func)
    valid_loss, valid_acc = evaluate(simpleRnn, valid_iterator, loss_func)

    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

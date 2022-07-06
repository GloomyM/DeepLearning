import torch
from torchtext.datasets import IMDB
from torchtext import data

batch_size = 64
device = torch.device('mps')
TEXT = data.Field(tokenize='spacy', tokenizer_language='en_core_web_sm')
LABEL = data.LabelField(dtype=torch.float)

train_set, test_set = IMDB.splits(TEXT, LABEL)
print("训练集大小: {}".format(len(train_set)))
print("测试集大小: {}".format(len(test_set)))
MAX_VOCAB_SIZE = 25000

TEXT.build_vocab(train_set, max_size=MAX_VOCAB_SIZE)  # 建立vocabulary，最多25000个token
LABEL.build_vocab(train_set)
print(TEXT.vocab.freqs.most_common(20))  # 输出词频最高的前20个token

import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, input_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, text):
        embedded = self.embedding(text)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))
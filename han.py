import torch
from gensim.models import KeyedVectors


class WordEncoder(torch.nn.Module):
    def __init__(self, embedding_weights, hidden_size):
        super(WordEncoder, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(
            embedding_weights, freeze=True
        )
        self.gru = torch.nn.GRU(
            input_size=self.embedding.embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True,
        )

    def forward(self, input, hidden):
        embedding = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedding, hidden)
        return output, hidden

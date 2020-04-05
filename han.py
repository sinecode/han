import torch


class WordEncoder(torch.nn.Module):
    def __init__(self, embedding_matrix, hidden_size=50):
        super(WordEncoder, self).__init__()
        embedding_dim = embedding_matrix.shape[1]
        self.embedding = torch.nn.Embedding.from_pretrained(
            embeddings=torch.FloatTensor(embedding_matrix), freeze=True,
        )
        self.gru = torch.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True,
        )
        self.word_context_vector = torch.nn.Parameter(
            torch.Tensor(2 * hidden_size, 2 * hidden_size)
        )

    def forward(self, input, hidden_state):
        output = self.embedding(input)
        return self.gru(output.float(), hidden_state)

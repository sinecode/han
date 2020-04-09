import torch


class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.gru = torch.nn.GRU(
            input_size=input_size, hidden_size=hidden_size, bidirectional=True,
        )

    def forward(self, input, hidden_state):
        return self.gru(input, hidden_state)


class Attention(torch.nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.context_vector = torch.nn.Parameter(torch.Tensor(input_size, 1))
        self.linear = torch.nn.Linear(
            in_features=input_size, out_features=input_size
        )

    def forward(self, input):
        output = torch.tanh(self.linear(input))
        output = torch.matmul(output, self.context_vector).squeeze(dim=-1)
        output = torch.nn.functional.softmax(output, dim=-1)
        outs = []
        for alpha, h in zip(output, input):
            alpha = alpha.unsqueeze(1).expand_as(h)
            outs.append(alpha * h)
        return sum(outs)


class Han(torch.nn.Module):
    def __init__(
        self,
        embedding_matrix,
        num_classes,
        batch_size=64,
        word_hidden_size=50,
        sent_hidden_size=50,
    ):
        super(Han, self).__init__()
        embedding_dim = embedding_matrix.shape[1]
        self.embedding = torch.nn.Embedding.from_pretrained(
            embeddings=torch.FloatTensor(embedding_matrix), freeze=True,
        )
        self.batch_size = batch_size
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.init_hidden_state()
        self.word_encoder = Encoder(embedding_dim, word_hidden_size)
        self.word_attention = Attention(input_size=word_hidden_size * 2)
        self.sent_encoder = Encoder(word_hidden_size * 2, sent_hidden_size)
        self.sent_attention = Attention(input_size=sent_hidden_size * 2)
        self.last_layer = torch.nn.Linear(
            in_features=sent_hidden_size * 2, out_features=num_classes
        )

    def init_hidden_state(self):
        self.word_hidden_state = torch.zeros(
            2, self.batch_size, self.word_hidden_size
        )
        self.sent_hidden_state = torch.zeros(
            2, self.batch_size, self.sent_hidden_size
        )

    def forward(self, input):
        output_list = []
        # reshape as [num_sent_per_doc, batch_size, num_words_per_doc]
        input = input.permute(1, 0, 2)
        # for each sentence in input..
        #
        # i is all the first sentences for each doc,
        # then i is all the sencond sentences for each doc,
        # then i is all the third sentences for each doc,
        # etc...
        for i in input:
            # reshape as [num_words_per_doc, batch_size]
            i = i.permute(1, 0)
            output = self.embedding(i)
            output, self.word_hidden_state = self.word_encoder(
                output.float(), self.word_hidden_state
            )
            output = self.word_attention(output).unsqueeze(0)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_encoder(
            output, self.sent_hidden_state
        )
        output = self.sent_attention(output)
        output = torch.nn.functional.softmax(
            self.last_layer(output), dim=0
        ).squeeze(1)
        return output

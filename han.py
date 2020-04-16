import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE, BATCH_SIZE


class WordAttention(nn.Module):
    def __init__(self, embedding_matrix, hidden_size):
        super(WordAttention, self).__init__()
        embedding_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.FloatTensor(embedding_matrix), freeze=True,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True,
        )
        word_vector_size = hidden_size * 2
        self.fc = nn.Linear(
            in_features=word_vector_size, out_features=word_vector_size
        )
        self.context_vector = nn.Parameter(torch.Tensor(word_vector_size, 1))

    def forward(self, input, hidden_state):
        # print(f"a {input.shape}")
        output = self.embedding(input)
        # print(f"b {output.shape}")
        f_output, h_output = self.gru(output.float(), hidden_state)
        # print(f"c {output.shape}")
        # print(f"d {h_output.shape}")
        output = torch.tanh(self.fc(f_output))
        # print(f"e {output.shape}")
        output = torch.matmul(output, self.context_vector).squeeze(dim=-1)
        # print(f"f {output.shape}")
        output = F.softmax(output, dim=0)
        # print(f"g {output.shape}")
        outs = []
        for alpha, h in zip(output, f_output):
            alpha = alpha.unsqueeze(1).expand_as(h)
            outs.append(alpha * h)
        return sum(outs).unsqueeze(0), h_output


class SentAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SentAttention, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, bidirectional=True,
        )
        self.context_vector = nn.Parameter(torch.Tensor(input_size, 1))
        self.fc = nn.Linear(in_features=input_size, out_features=input_size)

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)
        output = torch.tanh(self.fc(f_output))
        output = torch.matmul(output, self.context_vector).squeeze(dim=-1)
        output = F.softmax(output, dim=0)
        outs = []
        for alpha, h in zip(output, f_output):
            alpha = alpha.unsqueeze(1).expand_as(h)
            outs.append(alpha * h)
        return sum(outs), h_output


class Han(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        num_classes,
        word_hidden_size,
        sent_hidden_size,
    ):
        super(Han, self).__init__()
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.init_hidden_state(BATCH_SIZE)
        self.word_attention = WordAttention(embedding_matrix, word_hidden_size)
        self.sent_attention = SentAttention(
            input_size=word_hidden_size * 2, hidden_size=sent_hidden_size,
        )
        self.fc = nn.Linear(
            in_features=sent_hidden_size * 2, out_features=num_classes
        )

    def init_hidden_state(self, batch_size):
        self.word_hidden_state = torch.zeros(
            2, batch_size, self.word_hidden_size
        ).to(DEVICE)
        self.sent_hidden_state = torch.zeros(
            2, batch_size, self.sent_hidden_size
        ).to(DEVICE)

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
            i = i.permute(1, 0).to(DEVICE)
            output, self.word_hidden_state = self.word_attention(
                i, self.word_hidden_state
            )
            # print(f"h {output.shape}")
            # exit()
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.sent_hidden_state = self.sent_attention(
            output, self.sent_hidden_state
        )
        output = self.fc(output)
        output = F.log_softmax(output, dim=1)
        return output

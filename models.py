import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE, BIDIRECTIONAL


class WordEncoder(nn.Module):
    def __init__(self, embedding_matrix, hidden_size):
        super(WordEncoder, self).__init__()
        embedding_dim = embedding_matrix.shape[1]
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=torch.FloatTensor(embedding_matrix), freeze=True,
        )
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bidirectional=(BIDIRECTIONAL == 2),
        )

    def forward(self, input, hidden_state):
        output = self.embedding(input)
        f_output, h_output = self.gru(output, hidden_state)
        return f_output, h_output


class SentEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SentEncoder, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bidirectional=(BIDIRECTIONAL == 2),
        )

    def forward(self, input, hidden_state):
        f_output, h_output = self.gru(input, hidden_state)
        return f_output, h_output


class Attention(nn.Module):
    def __init__(self, input_size):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.fc = nn.Linear(self.input_size, self.input_size)
        self.context_vector = nn.Parameter(torch.randn(self.input_size))

    def forward(self, input):
        output = torch.tanh(self.fc(input))
        output = torch.matmul(output, self.context_vector)
        output = F.softmax(output, dim=1)
        output = output.permute(1, 0)
        input = input.permute(1, 0, 2)
        batch_size = input.shape[1]
        weighted_sum = torch.zeros(batch_size, self.input_size).to(DEVICE)
        for alpha, h in zip(output, input):
            alpha = alpha.unsqueeze(1).expand_as(h)
            weighted_sum += alpha * h
        return weighted_sum


class Fan(nn.Module):
    "Flat Attention Network"

    def __init__(
        self, embedding_matrix, word_hidden_size, num_classes, batch_size
    ):
        super(Fan, self).__init__()
        self.word_hidden_size = word_hidden_size
        self.word_encoder = WordEncoder(embedding_matrix, word_hidden_size)
        self.word_attention = Attention(word_hidden_size * BIDIRECTIONAL)
        self.fc = nn.Linear(word_hidden_size * BIDIRECTIONAL, num_classes)
        self.init_hidden_state(batch_size)

    def init_hidden_state(self, batch_size):
        self.word_hidden_state = torch.zeros(
            BIDIRECTIONAL, batch_size, self.word_hidden_size
        ).to(DEVICE)

    def forward(self, input):
        # Move the batch size in the last position because
        # we have to iterate over the document dimension,
        # that is over all the words of the document.
        input = input.permute(1, 0)
        self.word_hidden_state = torch.zeros_like(self.word_hidden_state).to(
            DEVICE
        )
        word_encoder_outputs = []
        for word in input:
            # Add an empty dimension because the GRU needs a 3D input,
            # moreover this is the dimension where all the encoder
            # outputs will be concatenated
            word = word.unsqueeze(0)
            output, self.word_hidden_state = self.word_encoder(
                word, self.word_hidden_state
            )
            word_encoder_outputs.append(output)
        word_attn_input = torch.cat(word_encoder_outputs, dim=0)
        word_attn_input = word_attn_input.permute(1, 0, 2)
        output = self.word_attention(word_attn_input)
        output = self.fc(output)
        output = F.log_softmax(output, dim=1)
        return output


class Han(nn.Module):
    "Hierachical Attention Network"

    def __init__(
        self,
        embedding_matrix,
        word_hidden_size,
        sent_hidden_size,
        num_classes,
        batch_size,
    ):
        super(Han, self).__init__()
        self.word_hidden_size = word_hidden_size
        self.word_encoder = WordEncoder(embedding_matrix, word_hidden_size)
        self.word_attention = Attention(word_hidden_size * BIDIRECTIONAL)
        self.sent_hidden_size = sent_hidden_size
        self.sent_encoder = SentEncoder(
            word_hidden_size * BIDIRECTIONAL, sent_hidden_size
        )
        self.sent_attention = Attention(sent_hidden_size * BIDIRECTIONAL)
        self.fc = nn.Linear(sent_hidden_size * BIDIRECTIONAL, num_classes)
        self.init_hidden_state(batch_size)

    def init_hidden_state(self, batch_size):
        self.word_hidden_state = torch.zeros(
            BIDIRECTIONAL, batch_size, self.word_hidden_size
        ).to(DEVICE)
        self.sent_hidden_state = torch.zeros(
            BIDIRECTIONAL, batch_size, self.sent_hidden_size
        ).to(DEVICE)

    def forward(self, input):
        # Move the batch size in the last position because
        # we have to iterate over the document dimensions,
        # that is over all the words and all the sentences.
        input = input.permute(1, 2, 0)
        self.sent_hidden_state = torch.zeros_like(self.sent_hidden_state).to(
            DEVICE
        )
        sent_encoder_outputs = []
        for sentence in input:
            self.word_hidden_state = torch.zeros_like(
                self.word_hidden_state
            ).to(DEVICE)
            word_encoder_outputs = []
            for word in sentence:
                # Add an empty dimension because the GRU needs a 3D input,
                # moreover this is the dimension where all the encoder
                # outputs will be concatenated
                word = word.unsqueeze(0)
                output, self.word_hidden_state = self.word_encoder(
                    word, self.word_hidden_state
                )
                word_encoder_outputs.append(output)
            word_attn_input = torch.cat(word_encoder_outputs, dim=0)
            word_attn_input = word_attn_input.permute(1, 0, 2)
            output = self.word_attention(word_attn_input)
            # Add an empty dimension (as before)
            output = output.unsqueeze(0)
            output, self.sent_hidden_state = self.sent_encoder(
                output, self.sent_hidden_state
            )
            sent_encoder_outputs.append(output)
        sent_attn_input = torch.cat(sent_encoder_outputs, dim=0)
        sent_attn_input = sent_attn_input.permute(1, 0, 2)
        output = self.sent_attention(sent_attn_input)
        output = self.fc(output)
        output = F.log_softmax(output, dim=1)
        return output

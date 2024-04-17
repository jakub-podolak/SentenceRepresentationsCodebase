import torch
import numpy as np
import torch.nn as nn


class UnidirectionalLSTMEncoder(nn.Module):
    def __init__(self, batch_size = 64, word_embedding_dim = 300, encoding_lstm_dim = 2048):
        super().__init__()
        self.batch_size = batch_size
        self.word_embedding_dim = word_embedding_dim
        self.encoding_lstm_dim = encoding_lstm_dim
        self.lstm = nn.LSTM(self.word_embedding_dim, self.encoding_lstm_dim, 1,
                            bidirectional=False, batch_first=True) # todo: check dropout

    def forward(self, sentences_length_tuple):
        padded_batch_of_sentences, true_sentence_lengths = sentences_length_tuple

        # convert lengths to torch
        sentence_lengths = torch.tensor(true_sentence_lengths, dtype=torch.long, device='cpu')
        sorted_lengths, idx_sort = torch.sort(sentence_lengths, descending=True)

        sorted_sentences = padded_batch_of_sentences.index_select(0, idx_sort)

        # pack sentences
        packed_sentences = nn.utils.rnn.pack_padded_sequence(sorted_sentences, sorted_lengths, batch_first=True)

        hidden_states = self.lstm(packed_sentences)[1][0]
        hidden_states = hidden_states.squeeze(0)

        _, idx_unsort = torch.sort(idx_sort)
        hidden_states = hidden_states.index_select(0, idx_unsort)

        return hidden_states
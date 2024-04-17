import torch
import numpy as np
import torch.nn as nn


class BidirectionalLSTMEncoder(nn.Module):
    def __init__(self, batch_size=64, word_embedding_dim=300, encoding_lstm_dim=2048, pooling_type=None):
        super().__init__()
        self.batch_size = batch_size
        self.word_embedding_dim = word_embedding_dim
        self.encoding_lstm_dim = encoding_lstm_dim
        self.lstm = nn.LSTM(input_size=self.word_embedding_dim,
                            hidden_size=self.encoding_lstm_dim,
                            num_layers=1,
                            bidirectional=True)
        self.pooling_type = pooling_type # None or max

    def forward(self, sentences_length_tuple):
        padded_batch_of_sentences, true_sentence_lengths = sentences_length_tuple
        
        # convert lengths to torch
        sentence_lengths = torch.tensor(true_sentence_lengths, dtype=torch.long, device=padded_batch_of_sentences.device)
        sorted_lengths, idx_sort = torch.sort(sentence_lengths, descending=True)

        sorted_sentences = padded_batch_of_sentences.index_select(0, idx_sort)

        packed_sentences = nn.utils.rnn.pack_padded_sequence(sorted_sentences, sorted_lengths, batch_first=True)

        lstm_output, (hidden_states, _) = self.lstm(packed_sentences)
        lstm_output, _ = nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)

        # unsort lstm output
        _, idx_unsort = torch.sort(idx_sort)
        lstm_output = lstm_output.index_select(0, idx_unsort)

        if self.pooling_type == 'max':
            # use only tokens that are not padded by adding -inf
            mask = torch.arange(lstm_output.size(1))[None, :] < sentence_lengths[:, None]
            mask = mask.to(lstm_output.device).unsqueeze(2)

            masked_output = lstm_output.masked_fill(~mask, float('-inf'))
            # max pooling across tokens
            sentence_embeddings, _ = torch.max(masked_output, dim=1)
        else:
            hidden_states = hidden_states.squeeze(0)
            # (num_layers * num_directions, batch, hidden_size)
            forward_hidden = hidden_states[-2,:,:]  # last layer's forward hidden state
            backward_hidden = hidden_states[-1,:,:]  # last layer's backward hidden state
            sentence_embeddings = torch.cat((forward_hidden, backward_hidden), dim=1)
            sentence_embeddings = sentence_embeddings.index_select(0, idx_unsort)

        return sentence_embeddings

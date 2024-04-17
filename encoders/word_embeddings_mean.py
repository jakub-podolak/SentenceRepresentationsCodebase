import torch
import torch.nn as nn
from .base_encoder import SentenceEncoder

class WordEmbeddingsMeanEncoder(SentenceEncoder):
    """
    Just takes the mean of the provided token embeddings.
    """
    def forward(self, sentences_length_tuple):
        x, sentence_lengths = sentences_length_tuple
        # x is of shape [batch_size, num_tokens, embedding_dimension]
        sentence_lengths = torch.Tensor(sentence_lengths).unsqueeze(-1)

        return x.sum(dim=1) / sentence_lengths
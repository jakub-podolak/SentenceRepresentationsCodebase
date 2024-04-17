import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class SentenceEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        # x is a tensor of shape [batch, num_tokens, embedding_dimension]
        pass
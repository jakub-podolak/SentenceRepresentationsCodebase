import torch
import torch.nn as nn
import torch.nn.functional as F

class SNLIClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_dim=300, hidden_dim=512):
        super(SNLIClassifier, self).__init__()

        self.encoder = encoder
        self.fn1 = nn.Linear(4 * embedding_dim, hidden_dim)
        self.fn2 = nn.Linear(hidden_dim, hidden_dim)
        self.fn3 = nn.Linear(hidden_dim, 3)    

    def forward(self, sentence_a, sentence_b):
        """
        sentence_a: word embeddings for first sentence
        sentence_b: word embeddings of the second sentence
        """
        # encode both sentences using the encoder
        sentence_a = self.encoder(sentence_a)
        sentence_b = self.encoder(sentence_b)

        f1 = (sentence_a - sentence_b).abs()
        f2 = sentence_a * sentence_b

        # sentence_a and sentence_b are 2D (b, e)
        x = torch.cat([sentence_a, sentence_b, f1, f2], dim=1)

        x = self.fn1(x)
        x = self.fn2(x)
        x = self.fn3(x)

        out = x
        return out
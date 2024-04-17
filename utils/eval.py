from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score
import numpy as np

from utils.utils import batch_data
from utils.word_embeddings import get_word_embeddings

def evaluate_model(model, data_to_evaluate, word2vec, batch_size=64, tuple_input=True):
    model.eval()

    sentence1 = list(data_to_evaluate['sentence1'])
    sentence2 = list(data_to_evaluate['sentence2'])
    target = list(data_to_evaluate['target'])

    predicted_labels = []

    for (s1_batch, s2_batch) in tqdm(zip(batch_data(sentence1, batch_size), batch_data(sentence2, batch_size)), total=(len(sentence1) // batch_size)):
        if tuple_input:
            s1_encoded = (torch.Tensor(np.array([get_word_embeddings(word2vec, s1) for s1 in s1_batch])), [len(s1) for s1 in s1_batch])
            s2_encoded = (torch.Tensor(np.array([get_word_embeddings(word2vec, s2) for s2 in s2_batch])), [len(s1) for s1 in s1_batch])
        else:
            s1_encoded = torch.Tensor(np.array([get_word_embeddings(word2vec, s1) for s1 in s1_batch]))
            s2_encoded = torch.Tensor(np.array([get_word_embeddings(word2vec, s2) for s2 in s2_batch]))
        output = model.forward(s1_encoded, s2_encoded)
        predicted_classes = torch.argmax(output, dim=-1).detach().numpy()
        predicted_labels += list(predicted_classes)
            
    return {
        'accuracy': accuracy_score(target, predicted_labels)
    }
import io
import logging

import numpy as np

def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
        for word in s:
            words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id


def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    return word_vec


def get_word_embeddings(word_vec, sent, pad_with_zeroes_to_length=100, embedding_size=300):
    word_embeddings = []
    for word in sent:
        word = str(word)
        if word in word_vec:
            word_embeddings.append(word_vec[word])
        else:
            # Handling unknown tokens
            word_embeddings.append(np.zeros(embedding_size))
        
    if pad_with_zeroes_to_length:
        # Add zero padding
        while len(word_embeddings) < pad_with_zeroes_to_length:
            word_embeddings.append(np.zeros(embedding_size))
    
    return word_embeddings
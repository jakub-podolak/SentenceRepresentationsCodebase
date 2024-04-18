import os
import pickle
from tqdm import tqdm
import string
import nltk
from nltk.tokenize import word_tokenize
import string
from datasets import load_dataset

nltk.download('punkt')

def preprocess_text(text):
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    tokens = word_tokenize(text)    
    return tokens

def get_snli_data(split='train', sample=None):
    loaded_dataset = load_dataset('snli')
    data = loaded_dataset[split].to_pandas()
    data.columns = ['sentence1', 'sentence2', 'target']

    # remove data for which we don't have gold label
    data = data[data['target'] != -1]

    if sample:
        print('sampling...')
        data = data.sample(sample)

    # tokenize and lowercase the data
    print('tokenizing sentence 1')
    tokenized = []
    for sentence1 in tqdm(data['sentence1']):
        tokenized.append(preprocess_text(sentence1))
    data['sentence1'] = tokenized
    
    print('tokenizing sentence 2')
    tokenized = []
    for sentence2 in tqdm(data['sentence2']):
        tokenized.append(preprocess_text(sentence2))
    data['sentence2'] = tokenized
        
    return data[['sentence1', 'sentence2', 'target']].dropna()

def save_data(data, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_data(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def check_and_load_or_save(split):
    file_path = f'tokenized/{split}.pickle'
    
    if not os.path.exists(file_path):
        data = get_snli_data(split)
        save_data(data, file_path)
    else:
        data = load_data(file_path)
    
    return data
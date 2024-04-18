import os
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime

import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

from utils.word_embeddings import get_word_embeddings, create_dictionary, get_wordvec
from utils.utils import batch_data
from utils.eval import evaluate_model
from utils.snli_data import check_and_load_or_save
from heads.snli_model import SNLIClassifier
from encoders.word_embeddings_mean import WordEmbeddingsMeanEncoder
from encoders.lstm.bidirectional_lstm import BidirectionalLSTMEncoder
from encoders.lstm.unidirectional_lstm import UnidirectionalLSTMEncoder


def train_one_epoch(nli_model, train, optimizer, loss_fn, word2vec, batch_size, writer, epoch, device, tuple_input=True):
    nli_model.train()

    all_costs = []

    train_shuffled = train.sample(len(train))
    
    sentence1 = list(train_shuffled['sentence1'])
    sentence2 = list(train_shuffled['sentence2'])
    target = list(train_shuffled['target'])

    i = 0
    running_loss = 0.0
    trainloader_len = len(list(batch_data(sentence1, batch_size)))
    print(trainloader_len)
    for (s1_batch, s2_batch, target_batch) in tqdm(zip(batch_data(sentence1, batch_size),\
                                                       batch_data(sentence2, batch_size),\
                                                       batch_data(target, batch_size)), total=(len(sentence1) // batch_size)):
        if tuple_input:
            s1_encoded = (torch.Tensor(np.array([get_word_embeddings(word2vec, s1) for s1 in s1_batch])).to(device), [len(s1) for s1 in s1_batch])
            s2_encoded = (torch.Tensor(np.array([get_word_embeddings(word2vec, s2) for s2 in s2_batch])).to(device), [len(s1) for s1 in s1_batch])
        else:
            s1_encoded = torch.Tensor(np.array([get_word_embeddings(word2vec, s1) for s1 in s1_batch])).to(device)
            s2_encoded = torch.Tensor(np.array([get_word_embeddings(word2vec, s2) for s2 in s2_batch])).to(device)

        tgt_batch = torch.Tensor(target_batch).long().to(device)

        optimizer.zero_grad()
        output = nli_model.forward(s1_encoded, s2_encoded)

        loss = loss_fn(output, tgt_batch)
        all_costs.append(loss.item())

        # backward
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            writer.add_scalar('training loss', running_loss / 100, epoch * trainloader_len + i)

            running_loss = 0.0
        i += 1

    return all_costs


def parse_option():
    parser = argparse.ArgumentParser("Sentence Representations Models")
    parser.add_argument(
        "--encoder", type=str, default="mean_embeddings", help="one of the implemented models"
    )
    parser.add_argument(
        "--vec_path", type=str, default="pretrained/glove.840B.300d.txt", help="path to vector embeddings txt"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=50, help="max number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch size"
    )
    parser.add_argument(
        "--optimizer_lr", type=float, default=0.1, help="initial learning rate"
    )
    parser.add_argument(
        "--encoding_dim", type=int, default=256, help="lstm encoding dimension"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.99, help="weight decay"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_option()
    print(args)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"runs/exp_{current_time}_{args.encoder}_{args.encoding_dim}"
    writer = SummaryWriter(run_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    print("using device", device)

    if not os.path.exists('tokenized'):
        os.makedirs('tokenized')
    
    train = check_and_load_or_save('train')
    valid = check_and_load_or_save('validation')
    test = check_and_load_or_save('test')
    
    print('Length of train data:', len(train))
    print('Length of validation data:', len(valid))
    print('Length of test data:', len(test))

    all_sentences = list(train['sentence1']) + list(train['sentence2']) +\
                    list(valid['sentence1']) + list(valid['sentence2']) +\
                    list(test['sentence1']) + list(test['sentence2'])
        
    id2word, word2id = create_dictionary(all_sentences)

    word2vec = get_wordvec(args.vec_path, word2id)

    encoder = None
    if args.encoder == 'mean_embeddings':
        encoder = WordEmbeddingsMeanEncoder()
        nli_model = SNLIClassifier(encoder=encoder, embedding_dim=300)
    elif args.encoder == 'lstm':
        encoder = UnidirectionalLSTMEncoder(encoding_lstm_dim=args.encoding_dim, batch_size=args.batch_size)
        nli_model = SNLIClassifier(encoder=encoder, embedding_dim=args.encoding_dim)
    elif args.encoder == 'bilstm':
        encoder = BidirectionalLSTMEncoder(pooling_type=None, encoding_lstm_dim=args.encoding_dim, batch_size=args.batch_size)
        nli_model = SNLIClassifier(encoder=encoder, embedding_dim=2*args.encoding_dim)
    elif args.encoder == 'bilstm_max':
        encoder = BidirectionalLSTMEncoder(pooling_type='max', encoding_lstm_dim=args.encoding_dim, batch_size=args.batch_size)
        nli_model = SNLIClassifier(encoder=encoder, embedding_dim=2*args.encoding_dim)
    
    encoder.to(device)
    nli_model.to(device)

    print(encoder, nli_model)
    try:
        print("encoder device", next(encoder.parameters()).device)
    except Exception:
        pass
    print("nli_model device", next(nli_model.parameters()).device)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=nli_model.parameters(), lr=args.optimizer_lr, weight_decay=args.weight_decay)

    best_validation_accuracy = 0.0

    for epoch in range(args.max_epochs):
        print('Epoch:', epoch)
        loss_batches = train_one_epoch(nli_model, train, optimizer, loss_fn, word2vec, args.batch_size, writer, epoch, device)
        print('Mean loss in epoch', np.mean(loss_batches))
        eval_score = evaluate_model(nli_model, valid, word2vec, args.batch_size, device)
        eval_accuracy = eval_score['accuracy']
        print('Validation accuracy', eval_accuracy)
        writer.add_scalar('validation accuracy', eval_accuracy, epoch)

        if eval_accuracy > best_validation_accuracy:
            best_validation_accuracy = eval_accuracy
            print("New high score! saving a model")
            torch.save(nli_model, f"{run_name}/model_{epoch}_checkpoint.pickle")
        else:
            # shrink lr
            optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 5
            print("Shrinking lr to", optimizer.param_groups[0]['lr'])
        
        # early stopping
        if optimizer.param_groups[0]['lr'] < 1e-5:
            print("Stopping!")
            break

if __name__ == "__main__":
    main()

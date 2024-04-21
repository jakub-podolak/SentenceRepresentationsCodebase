import logging
import sys
import argparse
import torch
import numpy as np
from utils.snli_data import check_and_load_or_save
from utils.word_embeddings import create_dictionary, get_wordvec, get_word_embeddings
from utils.eval import evaluate_model  

PATH_TO_SENTEVAL = 'SentEval/'
PATH_TO_DATA = 'SentEval/data/'
PATH_TO_VEC = 'pretrained/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def parse_option():
    parser = argparse.ArgumentParser("Evaluate Sentence Embedding Models")
    parser.add_argument(
        "--path", type=str, default="None", help="path to pretrained model"
    )
    parser.add_argument(
        "--transfer", action=argparse.BooleanOptionalAction
    )
    parser.add_argument(
        "--snli", action=argparse.BooleanOptionalAction
    )
    args = parser.parse_args()
    return args


def evaluate_transfer_tasks(sentence_encoder, word2id, word2vec):
     # SentEval prepare and batcher
    def prepare(params, samples):
        params.word2id = word2id
        params.word_vec = word2vec
        params.wvec_dim = 300
        params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        return

    def batcher(params, batch):
        batch = [sent if len(sent) > 0 else ['.'] for sent in batch]
        embeddings = []

        for sent in batch:
            word_embeddings = torch.Tensor(np.array([get_word_embeddings(params.word_vec, sent)])).to(params.device)

            sentvec = sentence_encoder.forward((word_embeddings, [len(sent)]))[0].detach().cpu().numpy()
            embeddings.append(sentvec)

        embeddings = np.vstack(embeddings)
        return embeddings

    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5, 'seed': 1111}

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.INFO)

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    # define transfer tasks
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14']
    # transfer_tasks = ['MR']

    results = se.eval(transfer_tasks)
    return results


def main():
    args = parse_option()
    print(args)

    assert args.path is not None

    with open(args.path, 'rb') as f:
        model = torch.load(f)
        sentence_encoder = model.encoder

    # Initialize dictionary from SNLI
    train = check_and_load_or_save('train')
    valid = check_and_load_or_save('validation')
    test = check_and_load_or_save('test')

    all_sentences = list(train['sentence1']) + list(train['sentence2']) +\
                    list(valid['sentence1']) + list(valid['sentence2']) +\
                    list(test['sentence1']) + list(test['sentence2'])

    _, word2id = create_dictionary(all_sentences)
    word2vec = get_wordvec(PATH_TO_VEC, word2id)

    if args.transfer:
        transfer_results = evaluate_transfer_tasks(sentence_encoder, word2id, word2vec)
    else:
        transfer_results = None

    if args.snli:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dev_acc = evaluate_model(model, valid, word2vec, batch_size=64, device=device)
        test_acc = evaluate_model(model, test, word2vec, batch_size=64, device=device)
    else:
        dev_acc = None
        test_acc = None
    print(sentence_encoder)
    print('transfer results:')
    print(transfer_results)
    print('dev acc:')
    print(dev_acc)
    print('test_acc:')
    print(test_acc)

if __name__ == "__main__":
    main()
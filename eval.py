import logging
import sys
import argparse
import torch
import numpy as np
from utils.word_embeddings import create_dictionary, get_wordvec, get_word_embeddings

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
    args = parser.parse_args()
    return args


def main():
    args = parse_option()
    print(args)

    assert args.path is not None

    with open(args.path, 'rb') as f:
        model = torch.load(f)
        sentence_encoder = model.encoder

    # SentEval prepare and batcher
    def prepare(params, samples):
        _, params.word2id = create_dictionary(samples)
        params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
        params.wvec_dim = 300
        params.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
        return

    def batcher(params, batch):
        batch = [sent if sent != [] else ['.'] for sent in batch]
        embeddings = []

        for sent in batch:
            word_embeddings = torch.Tensor(np.array([get_word_embeddings(params.word_vec, sent)])).to(params.device)

            sentvec = sentence_encoder.forward((word_embeddings, [len(sent)]))[0].detach().cpu().numpy()
            embeddings.append(sentvec)

        embeddings = np.vstack(embeddings)
        return embeddings

    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 5, 'seed': 1111}

    # Set up logger
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    se = senteval.engine.SE(params_senteval, batcher, prepare)

    # define transfer tasks
    transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14']

    results = se.eval(transfer_tasks)
    print(results)

if __name__ == "__main__":
    main()

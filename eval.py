import os
import argparse
from learner import Learner





def parse_option():
    parser = argparse.ArgumentParser("Evaluate Sentence Embedding Models")
    parser.add_argument(
        "--model", type=str, default="word_embeddings", help="one of the implemented models"
    )
    parser.add_argument(
        "--task", type=str, default="senteval", help="one of the implemented tasks"
    )


def main():
    args = parse_option()
    print(args)

if __name__ == "__main__":
    main()

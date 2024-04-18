#!/bin/bash

mkdir -p pretrained
cd pretrained

url="https://nlp.stanford.edu/data/glove.840B.300d.zip"

# Download the GloVe embeddings zip file
echo "Downloading GloVe embeddings..."
wget -c "$url"

# Unpack the zip file
echo "Unpacking..."
unzip -o glove.840B.300d.zip
echo "Done"
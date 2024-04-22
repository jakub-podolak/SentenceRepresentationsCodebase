# SentenceRepresentationsCodebase
My own codebase for experimenting with Sentence Represenantations for Advanced Topics in Computational Semantics

## 0. Organization of repository:

```
encoders/ 		# sentence encoders: including lstms and mean of embeddings
heads/ 			# classification heads that take embeddings for specific datasets (e.g. SNLI)
utils/ 			# various auxiliary functions with mixed use
train.py 		# main script for training (see section 3)
eval.py 		# main script for evaluation (see section 4)
```

Extra directories which are created during setup:

```
runs/ 			# directory with all models and tensorboard runs
pretrained/		# directory with glove embeddings
tokenized/		# directory where tokenized data is stored for efficiency
```



You can **download my runs [here](https://drive.google.com/drive/folders/1jwklBzEyv8po4n0PDaHQ2fZMrgBtb7gP?usp=sharing)**



## 1. Setup
##### 1.1 Install environment

```
conda env create --name acts_gpu --file=acts_gpu.yaml
source activate acts_gpu
```

##### 1.2 Download glove embeddings

```
bash download_glove.sh
```

##### 1.3 Install SentEval

Clone repo from FAIR github to this directory

```
git clone https://github.com/facebookresearch/SentEval.git
cd SentEval/
```

Install SentEval

```
python setup.py install
```

Download datasets

```
cd data/downstream/
./get_transfer_data.bash
```

## 2. Running on Lisa / Snellius

Install environment using
```bash
sbatch install_environment.job
```

Run interactive session
```bash
srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18 --time=04:00:00 --pty bash -i
```

And later
```bash
module purge
module load 2022
module load Anaconda3/2022.05

source activate acts_gpu
```



## 3. Training Models

Example for LSTM:

```
python3 train_snli.py --encoder lstm --max_epochs 50 --batch_size 64 --optimizer_lr 0.1 --encoding_dim 2048 --lr_decay 0.99
```

Currently supported encoders: `mean_embeddings` `lstm` `bilstm` `bilstm_max`

## 4. Evaluating Models

Example for LSTM:

```
python3 eval.py --transfer --snli --path runs/exp_20240418_145108_lstm_2048/model_13_checkpoint.pickle
```


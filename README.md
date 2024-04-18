# SentenceRepresentationsCodebase
My own codebase for experimenting with Sentence Represenantations for Advanced Topics in Computational Semantics

## 1. Setup on Lisa / Snellius
Download glove embeddings by running 
```
bash download_glove.sh
```

Install environment using
```
sbatch install_environment.job
```

Run interactive session
```
srun --partition=gpu --gpus=1 --ntasks=1 --cpus-per-task=18 --time=04:00:00 --pty bash -i
```

And later
```
module purge
module load 2022
module load Anaconda3/2022.05

source activate acts_gpu
```
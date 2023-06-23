# Benchmark Falcon LLM

## Installation on Lambda Cloud

From a clean [Lambda cloud instance](https://cloud.lambdalabs.com/):

**Install miniconda**

```
# Download latest miniconda.
wget -nc https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install. -b is used to skip prompt
bash Miniconda3-latest-Linux-x86_64.sh -b

# Activate.
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"

# (optional) Add activation cmd to bashrc so you don't have to run the above every time.
printf '\neval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc
```

**Setup env**

```
# Create and activate env. -y skips confirmation prompt.
conda create -n falcon-env python=3.9 -y
conda activate falcon-env

# newest torch with cuda 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -U accelerate einops sentencepiece git+https://github.com/huggingface/transformers.git && \
pip install -U trl git+https://github.com/huggingface/peft.git && \
pip install scipy datasets bitsandbytes wandb
```

## Usage

```
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
conda activate falcon-env
python ft.py
```

# Results

TODO: table / graph for training throughput (`seq/sec`)

# Credits

The fine-tuning script is based on this [Colab notebook](https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing) from Huggingface's blog: [The Falcon has landed in the Hugging Face ecosystem](https://huggingface.co/blog/falcon#fine-tuning-with-peft).

The installation steps are based on naterw's [instructions](https://huggingface.co/tiiuae/falcon-40b/discussions/18#647939c2c68a021fbba88182).
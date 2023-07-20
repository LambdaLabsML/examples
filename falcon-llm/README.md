# Fine-tuning Falcon LLM 7B/40B

Running state-of-the-art language models (LLMs) on a single GPU with [LoRA](https://arxiv.org/abs/2106.09685) and [quantization](https://github.com/TimDettmers/bitsandbytes) is extremely impressive. This is especially true with the recent emergence of commercially viable models like [Faclon](https://falconllm.tii.ae/) and [MPT](https://www.mosaicml.com/blog/mpt-30b). For instance, you can perform inference using the Falcon 40B model in 4-bit mode with approximately 27 GB of GPU RAM, making a single A100 or A6000 GPU sufficient. Additionally, you can fine-tune the same model using PEFT (Parameter-Efficient Fine-Tuning) in 4-bit mode with around 62 GB of GPU RAM, requiring a single A100 80GB card.

The ability to fine-tune these models on a single GPU allows us to employ traditional data parallelism and linearly scale training throughput with more GPUs. This blog post provides instructions on how to achieve that on Lambda Cloud. The same instructions can be applied to multi-GPU Linux workstations or servers, assuming they have the latest NVIDIA driver installed (which can be done using [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software)).

## Installation on Lambda Cloud

As of the writing of this tutorial (June 28, 2023), Lambda Cloud provides different versions of PyTorch for different instances (e.g., version 2.0.1 for H100 and version 1.3.1 for other instances). During our tests, we encountered some issues when running `Falcon`/`bitsandbytes` on specific GPUs within these environments. To maintain a clean setup, we have decided to write this tutorial using a conda environment and install PyTorch 2.0.1, built with CUDA 11.8, across all types of GPU instances, including H100, A100, A6000, and A10.

Here are the steps to set up the conda environment on a fresh new Lambda Cloud instance, based on the [instructions](https://huggingface.co/tiiuae/falcon-40b/discussions/18#647939c2c68a021fbba88182) provided by Huggingface community contributors:

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

Install using the yaml file:

```
conda env create -f falcon-env.yml
conda activate falcon-env
```

or manually:

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

We provide a fine-tuning script that is written based on this [Colab notebook](https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing). We modified the original script so it is data parallelized for better scaling across multiple GPUs. In particular, we launch the script with `torchrun` and use `device_map={"": Accelerator().process_index}` to allocate each replicate of the model on the correct device.

Now, we are ready to fine-tune the models. Here are some example commands:

```
eval "$(/home/ubuntu/miniconda3/bin/conda shell.bash hook)"
conda activate falcon-env

# Single GPU, falcon 7B, 4bit quantization
torchrun --nnodes 1 --nproc_per_node 1 \
ft.py \
-m ybelkada/falcon-7b-sharded-bf16 \
-q 4bit

# 8x GPUs, falcon 40B, 8bit quantization
torchrun --nnodes 1 --nproc_per_node 8 \
ft.py \
-m tiiuae/falcon-40b \
-q 8bit
```

# Results

We benchmarked the training throughput across multiple popular GPU instances. Here is the training throughputs measured by samples/sec:

## Falcon 7B

| Config           | samples/sec 4bit | samples/sec 8bit |
| ---------------- | ---------------- | ---------------- |
| 1xA100 80GB SXM4 | 5.024            | 1.923            |
| 8xA100 80GB SXM4 | 39.535           | 15.241           |
| 1xH100 80GB SXM5 | 6.7              | -                |
| 8xH100 80GB SXM5 | 55.423           | -                |
| 1xA100 40GB SXM4 | 3.9              | 1.688            |
| 1xA6000          | 1.895            | 1.513            |
| 1xA10            | 1.45             | OOM              |

## Falcon 40B

| Config           | samples/sec 4bit | samples/sec 8bit |
| ---------------- | ---------------- | ---------------- |
| 1xA100 80GB SXM4 | 1.111            | 0.36             |
| 8xA100 80GB SXM4 | 8.705            | 2.85             |

As the table shows, training throughput scales nearly perfectly (over 7.8x speedup from 1x to 8x GPUs). However, at the moment we are still experiencing some CUDA errors with H100 for some specific configurations. Here are some potentially related [discussions](https://github.com/search?q=repo%3ATimDettmers%2Fbitsandbytes+h100&type=issues) that we follow up on closely.

# Credits

Tim Dettmers's [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library and HuggingFace's [PEFT](https://github.com/huggingface/peft) library make it possible to fine-tune these models on a single GPU.

The fine-tuning script is based on this [Colab notebook](https://colab.research.google.com/drive/1BiQiw31DT7-cDp1-0ySXvvhzqomTdI-o?usp=sharing) from Huggingface's blog: [The Falcon has landed in the Hugging Face ecosystem](https://huggingface.co/blog/falcon#fine-tuning-with-peft). We modified the original script so it is data parallelized for better scaling across multiple GPUs.

The installation steps are based on the [instructions](https://huggingface.co/tiiuae/falcon-40b/discussions/18#647939c2c68a021fbba88182) from Huggingface community contributors.

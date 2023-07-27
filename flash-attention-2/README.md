# FlashAttention 2 on Lambda Cloud

FlashAttention, the game-changing algorithm designed to accelerate attention modules and minimize memory usage without any approximation, took the world by storm after its release in 2022. It quickly found its way into machine learning [frameworks](https://github.com/Dao-AILab/flash-attention/blob/main/usage.md#integrated-into-machine-learning-frameworks) and became a staple in industry-standard [benchmarks](https://spectrum.ieee.org/mlperf-rankings-2022), leaving a trail of awe-inspiring results in its wake.

Now, brace yourself for the next level of innovation as FlashAttention 2 has been released! Building upon its predecessor's success, FlashAttention 2 delivers an astounding 2× speedup, achieved through improved parallelism and work partitioning. In this blog post, we'll show you how to use FlashAttention 2 on Lambda Cloud and share benchmark results for training GPT3-style models using A100 and H100 GPUs. Let's dive in!

![record](imgs/record.gif)

## Installation

The authors of FlasAttention have provided a [Dockerfile](https://github.com/Dao-AILab/flash-attention/blob/main/training/Dockerfile) that contains the latest FlashAttention 2. Our fork of the repo contains configuration to train GPT3-style models with the OpenWebText dataset.

```
git clone https://github.com/LambdaLabsML/flash-attention.git && \
cd flash-attention && \
docker build -t flash-attention:latest ./training
```

## Run

From the `flash-attention` folder:

```
# Prepare openwebtext dataset

docker run --gpus all --shm-size=1024g \
-v ${PWD}:/workspace \
flash-attention:latest \
sh -c 'cd  /workspace/training && export PYTHONPATH=$PWD:$PYTHONPATH && pytest -q -s tests/datamodules/test_language_modeling_hf.py -k "openwebtext"'


# Train GPT-3
# Change trainer.devices so it matches the number of GPUs on your machine

docker run --gpus all --shm-size=1024g \
-v ${PWD}:/workspace \
flash-attention:latest \
sh -c 'cd  /workspace/training && export PYTHONPATH=$PWD:$PYTHONPATH && python run.py experiment=owt/gpt3-2.7B-flash trainer.devices=8'
```

## Results

Here is the training speed of the `gpt3-2.7B-flash` model with FlashAttention 2. Overall H100 80GB SXM5 produces more than 2x tokens/sec compared to A100 80GB SXM4. And both cards scale well from 1x to 8x GPUs.

| Configurations   | Iter/Sec | Tokens/Sec | BS_per_GPU | Memory_per_GPU (GB) | Time to 300B Tokens GPT3-2.7B (Days) | Extrapolated Time to 300B Tokens GPT3-175B (Days) |
| ---------------- | -------- | ---------- | ---------- | ------------------- | ------------------------------------ | ------------------------------------------------- |
| A100 80GB SXM4   | 2.6      | 10649.6    | 4          | 73                  | 326                                  | 21132                                             |
| H100 80GB SXM5   | 5.44     | 22282.24   | 4          | 73                  | 156                                  | 10100                                             |
| 8xA100 80GB SXM4 | 2.5      | 81920      | 4          | 56                  | 42                                   | 2747                                              |
| 8xH100 80GB SXM5 | 5.34     | 174981.12  | 4          | 56                  | 20                                   | 1286                                              |

We also estimated the time to solution (process 300 Billion tokens) for GPT3 175B model by linearly scaling the time to solution of the GPT3 2.7B model by 65 folds (`175/2.7`). The result suggests that with FlashAttention 2 one can expect to reproduce GPT3 175B training in just about 10 days with 1024 H100 80GB SXM5 GPUs.

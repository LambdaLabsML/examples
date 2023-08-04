# FlashAttention 2 on Lambda Cloud A100 v.s. H100

FlashAttention, the game-changing algorithm designed to accelerate attention modules and minimize memory usage without any approximation, took the world by storm after its release in 2022. It quickly found its way into machine learning [frameworks](https://github.com/Dao-AILab/flash-attention/blob/main/usage.md#integrated-into-machine-learning-frameworks) and became a staple in industry-standard [benchmarks](https://spectrum.ieee.org/mlperf-rankings-2022), leaving a trail of awe-inspiring results in its wake.

Now, brace yourself for the next level of innovation as FlashAttention 2 has been released! Building upon its predecessor's success, FlashAttention 2 delivers an astounding 2× speedup, achieved through improved parallelism and work partitioning. In this blog post, we'll show you how to use FlashAttention 2 on Lambda Cloud and share benchmark results for training GPT3-style models using A100 and H100 GPUs. The key findings are:

- FlashAttention 2 achieved 3x or higher speedups over the baseline Huggingface implementation.
- H100 80GB SXM5 is 2x faster than A100 80GB SXM4 when running FlashAttention 2.

To give you a taste of its real-world impact, FlashAttention 2 enables replicating `GPT3-175B` training with "just" 242,400 GPU hours (H100 80GB SXM5). On [Lambda Cloud](<(https://lambdalabs.com/service/gpu-cloud/reserved)>), this translates to $1,175,640 using the Sprint cluster ($4.85/H100/Hour) or $458,136 using the three-year reserved cluster ($1.89/H100/Hour). This represents a remarkable 75% or 90% cost reduction compared to our [earlier blog's](https://lambdalabs.com/blog/demystifying-gpt-3) $4,600,000 estimation.

Without further ado, let's dive into the details of the benchmark and results.

![record](imgs/record.gif)

## Benchmark

The FlashAttention repo have provided a [Dockerfile](https://github.com/Dao-AILab/flash-attention/blob/main/training/Dockerfile) that contains the latest FlashAttention 2. Our fork of the repo contains configuration to train GPT3-style models with the OpenWebText dataset.

```
git clone https://github.com/LambdaLabsML/flash-attention.git && \
cd flash-attention && \
docker build -t flash-attention:latest ./training
```

Now you can launch the data preparation script and the benchmark script from the `flash-attention` folder:

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

There are multiple configurations in the [experiment](https://github.com/LambdaLabsML/flash-attention/tree/main/training/configs/experiment) folder. The rest of the blog will focus on the `GPT3-2.7B` configuration.

## Results

Let's begin by comparing the training speed of the [baseline](https://github.com/LambdaLabsML/flash-attention/blob/main/training/configs/experiment/owt/gpt3-2.7B-hf.yaml) implementation and the [FlashAttention 2](https://github.com/LambdaLabsML/flash-attention/blob/main/training/configs/experiment/owt/gpt3-2.7B-flash.yaml) implementation. The table below shows the FlashAttention 2 implementation achieved `3x` or higher `Tokens/Sec`, which is calculated as `Iter/Sec` x `max_length` x `batch_size`. Additionally, FlashAttention 2 optimizes memory usage, enabling an increase in affordable `batch_size` from `1` to `4`. In our benchmark, we set `headdim` to `80` for the FlashAttention 2 implementation. While setting `headdim` to `128` ([calculated](https://github.com/LambdaLabsML/flash-attention/blob/main/training/configs/experiment/owt/gpt3-2.7B-flash-hdim128.yaml#L8-L9) as `n_embd` divided by `n_head`) may yield slightly improved performance, the exact difference varies depending on the model.

| Configurations          | Iter/Sec | Tokens/Sec | BS_per_GPU | Memory_per_GPU (GB) | Time to 300B Tokens GPT3-2.7B (Days) | Extrapolated Time to 300B Tokens GPT3-175B (Days) |
| ----------------------- | -------- | ---------- | ---------- | ------------------- | ------------------------------------ | ------------------------------------------------- |
| A100 80GB SXM4 Baseline | 3.63     | 3717.1     | 1          | 73                  | 934                                  | 60544                                             |
| A100 80GB SXM4 FA2      | 2.6      | 10649.6    | 4          | 73                  | 326                                  | 21132                                             |
| H100 80GB SXM5 Baseline | 6.12     | 6266.88    | 1          | 73                  | 156                                  | 10100                                             |
| H100 80GB SXM5 FA2      | 5.44     | 22282.24   | 4          | 73                  | 555                                  | 35911                                             |

It is nice to see that H100 80GB SXM5 produces more than 2x `Tokens/Sec` compared to A100 80GB SXM4 (`22282.24` v.s. `10649.6`), and both GPUs scaled very well from 1x to 8x GPUs (96% and 98% for A100 and H100 respectively, as shown in the table below).

| Configurations       | Iter/Sec | Tokens/Sec | BS_per_GPU | Memory_per_GPU (GB) | Time to 300B Tokens GPT3-2.7B (Days) | Extrapolated Time to 300B Tokens GPT3-175B (Days) |
| -------------------- | -------- | ---------- | ---------- | ------------------- | ------------------------------------ | ------------------------------------------------- |
| A100 80GB SXM4 FA2   | 2.6      | 10649.6    | 4          | 73                  | 326                                  | 21132                                             |
| H100 80GB SXM5 FA2   | 5.44     | 22282.24   | 4          | 73                  | 156                                  | 10100                                             |
| 8xA100 80GB SXM4 FA2 | 2.5      | 81920      | 4          | 56                  | 42                                   | 2747                                              |
| 8xH100 80GB SXM5 FA2 | 5.34     | 174981.12  | 4          | 56                  | 20                                   | 1286                                              |

Last but not least, we estimated the time to solution (process 300 Billion tokens) for `GPT3-175B` model by linearly scaling the time to solution of the GPT3 2.7B model by 65 folds (`175/2.7`). The result suggests that with FlashAttention 2 one can expect to reproduce `GPT3-175B` training in just about 10 days with 1024 H100 80GB SXM5 GPUs. On [Lambda Cloud](<(https://lambdalabs.com/service/gpu-cloud/reserved)>), this translates to $1,175,640 using the Sprint cluster ($4.85/H100/Hour) or $458,136 using the three-year reserved cluster ($1.89/H100/Hour). This represents a remarkable 75% or 90% reduction compared to our [earlier blog's](https://lambdalabs.com/blog/demystifying-gpt-3) estimated cost of $4,600,000.

# Acknowledgement

We thank Tri Dao (the first author of [FlashAttention](https://github.com/Dao-AILab/flash-attention)) for valuable feedback on our benchmark results.

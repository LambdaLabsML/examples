# Flash-Attention 2 on Lambda Cloud

![record](imgs/record.gif)

## Installation

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
# Change trainer.devices accordingly to match the number of GPUs on your instance

docker run --gpus all --shm-size=1024g \
-v ${PWD}:/workspace \
flash-attention:latest \
sh -c 'cd  /workspace/training && export PYTHONPATH=$PWD:$PYTHONPATH && python run.py experiment=owt/gpt3-2.7B-flash trainer.devices=8'
```

## Results

Here is the training speed of the `gpt3-2.7B-flash` model with flash attention 2. Overall H100 80GB SXM5 produces more than 2x tokens/sec compared to A100 80GB SXM4. And cards scale well from 1x to 8x GPUs.

| Configurations   | Iter/Sec | Tokens/Sec | BS_per_GPU | Memory_per_GPU (GB) | Time to 300B Tokens GPT3-2.7B (Days) | Extrapolated Time to 300B Tokens GPT3-175B (Days) |
| ---------------- | -------- | ---------- | ---------- | ------------------- | ------------------------------------ | ------------------------------------------------- |
| A100 80GB SXM4   | 2.6      | 10649.6    | 4          | 73                  | 326                                  | 21132                                             |
| H100 80GB SXM5   | 5.44     | 22282.24   | 4          | 73                  | 156                                  | 10100                                             |
| 8xA100 80GB SXM4 | 2.5      | 81920      | 4          | 56                  | 42                                   | 2747                                              |
| 8xH100 80GB SXM5 | 5.34     | 174981.12  | 4          | 56                  | 20                                   | 1286                                              |

We also estimated the time to solution (process 300 Billion tokens) for GPT3 175B model by linearly scaling the time to solution of the GPT3 2.7B model by 65 folds (`175/2.7`). The result suggests that with Flash Attention 2 one can expect to reproduce GPT3 175B training in just about 10 days with 1024 H100 80GB SXM5 GPUs.

# Fine tuning Meta's Llama 2 on Lambda GPU Cloud

Meta [recently released](https://ai.meta.com/llama/) the next generation of the Llama models (Llama 2), trained on 40% more data! Given the explosive popularity of open source large language models (LLMs) like Llama, which spawned popular models like vicuna and falcon, new versions of these models are very exciting.

Along with the development of these models, the open source community has released a ton of utilities for fine tuning and deploying language models. Libraries like [peft](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/TimDettmers/bitsandbytes), and [trl](https://github.com/lvwerra/trl/tree/main) make it possible to fine tune LLMs on machines that can't hold the full precision model in GPU ram.

This blog post provides instructions on how to fine tune Llama 2 models on [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud). The same instructions can be applied to multi-GPU Linux workstations or servers, assuming they have the latest NVIDIA driver installed (which can be done using [Lambda Stack](https://lambdalabs.com/lambda-stack-deep-learning-software)).

## Getting access to the models

Both meta and huggingface need to grant you access to download the models:

1. Request access from Meta here: https://ai.meta.com/resources/models-and-libraries/llama-downloads/
2. Request access from huggingface on any of the model pages: https://huggingface.co/meta-llama/Llama-2-7b
3. Set up an auth token with huggingface here: https://huggingface.co/settings/tokens. You'll use this later.

## Spin up a GPU machine (you can use an A10!)

You can go to https://cloud.lambdalabs.com/instances and select whatever instance type you want to start fine tuning the llama models. For the 7b parameter variant, you can go as small as an A10 (24GB GPU ram).

Assuming you've also set up an ssh key at https://cloud.lambdalabs.com/ssh-keys, once the machine starts up you can just copy/paste the ssh command associated with the machine and run that in a terminal:

![Alt text](image.png)

## Set up environment

1. Install python packages

```bash
pip install transformers peft trl bitsandbytes
```

2. Clone the `trl` repo for the training script (https://github.com/lvwerra/trl/blob/main/examples/scripts/sft_trainer.py)

```bash
git clone https://github.com/lvwerra/trl
```

3. Log into huggingface on CLI

```bash
huggingface-cli login
```

Copy the auth token you created earlier (from https://huggingface.co/settings/tokens) and paste it into the prompt.

## Fine tune!

```bash
python trl/examples/scripts/sft_trainer.py \
    --model_name meta-llama/Llama-2-7b-hf \
    --dataset_name timdettmers/openassistant-guanaco \
    --load_in_8bit \
    --use_peft \
    --batch_size 8 \
    --gradient_accumulation_steps 1
```

This will download the model weights automatically, so the first time you run, it will take a bit to actually start training.

You should end up seeing output like this:

```bash
...
{'loss': 1.6493, 'learning_rate': 1.4096181965881397e-05, 'epoch': 0.0}                                                                        
{'loss': 1.3571, 'learning_rate': 1.4092363931762796e-05, 'epoch': 0.0}                                                                        
{'loss': 1.5853, 'learning_rate': 1.4088545897644193e-05, 'epoch': 0.0}                                                                        
{'loss': 1.4237, 'learning_rate': 1.408472786352559e-05, 'epoch': 0.0}                                                                         
{'loss': 1.7098, 'learning_rate': 1.4080909829406987e-05, 'epoch': 0.0}                                                                        
{'loss': 1.4348, 'learning_rate': 1.4077091795288384e-05, 'epoch': 0.0}                                                                        
{'loss': 1.6022, 'learning_rate': 1.407327376116978e-05, 'epoch': 0.01}                                                                        
{'loss': 1.3352, 'learning_rate': 1.4069455727051177e-05, 'epoch': 0.01}
...
```

## Summary

We've shown how easy it is to spin up a low cost ($0.60 per hour) GPU machine to fine tune the Llama 2 7b models. Spinning up the machine and setting up the environment takes only a few minutes, and the downloading model weights takes ~2 minutes at the beginning of training. This means you start fine tuning within 5 minutes using really simple commands!

If you request a larger GPU like an A100, you can up the batch size you use in the training command, which will increase the samples/sec. Here are some simple benchmarks on different GPUs and batch sizes:

| Model | GPU    | Batch Size | 4bit samples/sec[1] | 8bit samples/sec[1] |
| ----- | ------ | ---------- | ------------------- | ------------------- |
| 7b    | 1xA10  | 8          | 0.13                | 0.5                 |
| 7b    | 1xA100 | 32         | 0.4                 | 0.7                 |

[1] Here samples/sec is calculated by multiplying batch size by the inverse of `s/iter` that the sft_trainer script reports. All training runs had gradient accumulation steps equal to 1.

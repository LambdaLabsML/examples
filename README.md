# Deep Learning Examples

A repository of deep learning examples.

## Getting Started

- Pick an example from the list below
- Use your [local GPU machine](https://lambdalabs.com/gpu-workstations/vector), or get the GPUs you need in [Lambda Cloud](https://lambdalabs.com/service/gpu-cloud)
- Follow the instructions in the examples README.md

## Examples

- [YoloV5](yolov5/)
Example workflow involving YoloV5. See this blog post for more instructions: XX. The main entry point is `yolov5/prepare_dataset.py`, which downloads a dataset, preps it for YoloV5, and downloads pretrained weights files.
Roughly, you'll need to follow these steps:
  - `cd yolov5`
  - Clone YoloV5 from within the YoloV5 repository, i.e. `git clone git@github.com:ultralytics/yolov5.git`
  - Install YoloV5 requirements.txt
  - Run `python prepare_dataset.py` to download WiderFace data from HuggingFace. You can then use YoloV5 training/inference CLI as expected.

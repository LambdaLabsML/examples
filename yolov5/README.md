Example workflow involving YoloV5. See this blog post for more instructions: XX. The main entry point is `yolov5/prepare_dataset.py`, which downloads a dataset, preps it for YoloV5, and downloads pretrained weights files.
Roughly, you'll need to follow these steps:
  - Clone YoloV5 from within the YoloV5 repository, i.e. `git clone git@github.com:ultralytics/yolov5.git`
  - Install libs from YoloV5's requirements.txt
  - Run `python prepare_dataset.py` to download WiderFace data from HuggingFace. You can then use YoloV5 training/inference CLI as expected.

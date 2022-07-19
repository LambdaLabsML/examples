#!/bin/sh
pip install min-dalle streamlit 'basicsr>=1.3.3.11' opencv-python

git clone https://github.com/xinntao/Real-ESRGAN.git
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
pip install -e Real-ESRGAN
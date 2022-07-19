#!/bin/sh

git clone https://github.com/kuprel/min-dalle.git
cd min-dalle
git clone https://github.com/xinntao/Real-ESRGAN.git

pip install streamlit 'basicsr>=1.3.3.11' opencv-python
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

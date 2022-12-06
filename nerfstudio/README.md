# Hosting Nerfstudio Publically (use your machine or on Lambda Cloud)

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Example](#example)


## Introduction

![Alt Text](img/deyoung.gif)



[Nerfstudio](https://docs.nerf.studio/en/latest/) is an excellent tool that enables people to create and view [NeRF](https://arxiv.org/abs/2003.08934) (Neural Radiance Fields) models. It contains several APIs that simplify [data preprocessing](https://docs.nerf.studio/en/latest/quickstart/custom_dataset.html) (e.g., running `colmap`) and [model training](https://docs.nerf.studio/en/latest/quickstart/first_nerf.html). It also comes with a web-based interactive [viewer](https://docs.nerf.studio/en/latest/quickstart/viewer_quickstart.html) that visualizes the input data and the output model.

It is quite easy to get nerfstudio up and running -- as long as you have a machine with a decent GPU to work with. However, what if that is not the case and you have to run the job on a remote server? In particular, how do you monitor a remote NeRF job and share it with other people?

This tutorial describes the steps of setting up nerfstudio on a remote cloud instance and sharing your work as as a public app. For sharing the app, we will choose a method thats avoid SSH port forwarding -- this is helpful when sharing SSH credentials is not a viable option. 

We will use Lambda Cloud instance as an example. You can find how to spin up a Lambda Cloud instance in this [YouTube tutorial](https://www.youtube.com/watch?v=CKxR6ClKstU). The workflow has been tested with Lambda's 1xA100 instance, which we found to be a good fit for NeRF workloads. The same workflow should work for other cloud options and local machines as long as they come with a decent GPU and have Ubuntu and the NVIDIA driver pre-installed.


## Installation

We use Docker to set up the environment for nerfstudio. Lambda cloud has docker pre-installed with GPU support. You can also install docker on your own machine using these commands:

```
sudo apt-get install docker.io nvidia-container-toolkit && sudo systemctl restart docker
```

It is also useful to remove the `sudo` requirement for running docker:

```
sudo groupadd docker
sudo usermod -aG docker $USER
```

You need to ssh into the server with a new session to let `usermod` take effect.  


Now you are ready to build the image for nerfstudio. Here are the commands:

```
export username=YOUR_DOCKER_HUB_ACCOUNT_NAME

git clone --branch lambda https://github.com/LambdaLabsML/nerfstudio.git && \
cd nerfstudio && \
docker build . -t $username/nerfstudio:latest
```

The image takes about 50 mins to build. You can also directly pull our pre-built image from the docker hub:

```
docker pull chuanli11/nerfstudio:latest
```

The dockerfile is based on the official [nerfstudio dockerfile](https://github.com/nerfstudio-project/nerfstudio/blob/main/Dockerfile), with a few tweaks:

* Use colmap's `dev` branch instead of the `3.7` branch so that colmap can work with __GPU__ on a __headless__ server. Otherwise running colmap will cause `qt.qpa.xcb: could not connect to display` error on a headless server.
* Set `ENV TCNN_CUDA_ARCHITECTURES="80;86"` so that tinycudann works A100 card.
* Install [localtunnel](https://github.com/localtunnel/localtunnel) inside of the image for sharing nerfstudio viewer publically.   


## Usage

### Step One: Start an interactive docker session

```
docker run --gpus all \
-v <host-data-dir>:/workspace/ \
-v /home/ubuntu/.cache/:/home/user/.cache/ \
-p 7007:7007 \
--rm \
-it \
<username>/nerfstudio:latest
```

This command binds port `7007` of the container to port `7007` of the host machine. This is the default port used by the nerfstudio viewer. It also mounts the data directory on the host machine to the `/workspace` folder inside the container.

### Step two: Data Process

Here we use video input as an example. Assume the input video is located at `/workspace/<path-to-video.MP4>`. We can use the following command to prepare input data (extra frames and run colmap with them):

```
ns-process-data video \
--data <path-to-video.MP4> \
--output-dir <input-processed-dir> \
--verbose \
--num_frames_target 400 \
--matching_method sequential
```

We set `num_frames_target` to `400` -- this is an approximated number of frames extracted from the video footage. We set `matching_method` to `sequential` so to avoid an exhaustive `colmap` feature matching process. These settings can be changed according to the scene's complexity, the time budget, and the computing budget. In particular, `sequential` matching is only recommended when input images are from video footage.


### Step Three: Expose Your Localhost To The World

Before we launch a NeRF training job, we should use `localtunnel` to generate a public URL for interacting with nerfstudio remotely. There are many options for serving a self-hosted apps publically (e.g. ngrok or Cloudflare tunnel). We choose `localtunnel` since it does not require any registration or auth token. It is also the method used in this official nerfstudio [colab notebook](https://github.com/nerfstudio-project/nerfstudio/blob/main/colab/demo.ipynb)

Since `localtunnel` is pre-installed in our docker image, you can directly launch a service via the following command:

```
lt --port 7007 &
```

The `port` should be consistent with what nerfstudio viewer uses (default 7007). Remember to hit the enter key so that `localtunnel` runs in the background. It should output an URL for the service. You need to keep a note for the `<app-name>` part of the URL -- that is, everything after `https://`. For example, if the URL is 

```
https://huge-owls-reply-104-171-203-21.loca.lt
```

The `<app-name>` will be

```
huge-owls-reply-104-171-203-21.loca.lt
```

### Step Four: Training

This is the command to run NeRF training:

```
ns-train nerfacto \
--data <input-processed-dir>
```

Once the training starts, you can use the following __public__ URL to view the result:

```
https://viewer.nerf.studio/?websocket_url=wss://<app-name>
```

![Alt Text](img/viewer_ui.png)

By default `ns-train` will train a model from scratch. If you want to start from a pre-trained model, simply point `trainer.load-dir` to the pre-trained model directory. For loading visualization without training, you can set `--viewer.start-train` to `False`:

```
ns-train nerfacto \
--data <input-processed-dir> \
--trainer.load-dir <path-to-model-dir> \
--viewer.start-train False
```

### Step Five: Rendering 

Once the model is trained, you can render a video of the scene. This [YouTube tutorial](https://www.youtube.com/watch?v=nSFsugarWzk&t=136s) is a very useful guide to creating a camera path and rendering a video for your model.

![Alt Text](img/viewer_camera_path.png)



## Example

You can download our [preprocessed dataset](https://drive.google.com/file/d/1Nq6lQ8yoS_XiNXA1Zny1mHJqroWcYcT-/view?usp=sharing) and directly train a NeRF model with it. 

Assuming the data is unzipped to `/home/ubuntu/data/deyoung_frames`. These are the command to run the training job.

```
# Launch docker with data folder mounted
docker run --gpus all \
-v /home/ubuntu/data:/workspace/ \
-v /home/ubuntu/.cache/:/home/user/.cache/ \
-p 7007:7007 \
--rm \
-it \
<username>/nerfstudio:latest


# Take a note on the https://<app-name> created by localtunnel
lt --port 7007 &

# Launch training job
ns-train nerfacto \
--data deyoung_frames

# View your job at https://viewer.nerf.studio/?websocket_url=wss://<app-name>
```

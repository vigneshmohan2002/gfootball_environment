ARG DOCKER_BASE
FROM $DOCKER_BASE
ARG DEVICE

ENV DEBIAN_FRONTEND=noninteractive


# Import the GPG key for the NVIDIA CUDA repository

RUN dpkg --configure -a
    
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update
RUN apt-get --no-install-recommends install -yq git cmake build-essential \
  libgl1-mesa-dev libsdl2-dev \
  cuda \
  libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
  libdirectfb-dev libst-dev mesa-utils xvfb x11vnc \
  cmake libopenmpi-dev python3-dev zlib1g-dev \
  python3-pip neovim vim

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install --no-cache-dir psutil dm-sonnet==1.*
RUN python3 -m pip install --user nvidia-pyindex
RUN python3 -m pip install --user nvidia-tensorflow[horovod]
RUN python3 -m pip install 'gast==0.2.2'
RUN python3 -m pip install stable-baselines
RUN python3 -m pip install tqdm


RUN python3 -m pip install --no-cache-dir git+https://github.com/openai/baselines.git@master
COPY . /gfootball
RUN cd /gfootball && python3 -m pip install .
WORKDIR '/gfootball'

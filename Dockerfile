FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
WORKDIR /workspace

# python, dependencies for mujoco-py (and in general useful for RL research),
# from https://github.com/openai/mujoco-py
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    build-essential \
    patchelf \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

### NetHack dependencies
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update && apt-get install -yq \
        bison \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        flex \
        git \
        gpg \
        libbz2-dev \
        ninja-build \
        software-properties-common \
        wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null \
        | gpg --dearmor - \
        > /usr/share/keyrings/kitware-archive-keyring.gpg
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ bionic main' \
        > /etc/apt/sources.list.d/kitware.list
RUN apt-get update && apt-get install -yq \
    cmake \
    kitware-archive-keyring
COPY . /opt/nle

COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip setuptools wheel
RUN pip3 install -r requirements.txt

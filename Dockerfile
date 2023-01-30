FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
WORKDIR /workspace

RUN #rm /etc/apt/sources.list.d/cuda.list
RUN #rm /etc/apt/sources.list.d/nvidia-ml.list
# python, dependencies for mujoco-py, from https://github.com/openai/mujoco-py
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    build-essential \
    patchelf \
    curl \
    git \
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

# RUN ln -s /usr/bin/python3 /usr/bin/python
# installing mujoco distr
RUN mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}

# installing dependencies, optional mujoco_py compilation
COPY requirements.txt requirements.txt
RUN pip install --pre torch --force-reinstall --index-url https://download.pytorch.org/whl/nightly/cu117
RUN pip install -r requirements.txt

RUN python3 -c "import mujoco_py"

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

# Install package
RUN pip install nle
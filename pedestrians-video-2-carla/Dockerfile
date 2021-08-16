FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libboost-python-dev \
    libjpeg-dev \
    libjpeg-turbo8-dev \
    libpng-dev \
    python3-pip \
    screen \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -f https://download.pytorch.org/whl/torch_stable.html \
    torch==1.9.0+cu111 \
    torchvision==0.10.0+cu111 \
    torchaudio==0.9.0 \
    && pip install --no-cache-dir \
    autopep8 \
    pylint

# Let's pretend we've installed CARLA via easy_install
# It's client for Python 3.7 and in Ubuntu 20.04 there's Python 3.8 but hopefully this will work
# TODO: update it to installable, official CARLA package once we make a switch to 0.9.12
COPY --from=carlasim/carla:0.9.11 /home/carla/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg /usr/local/lib/python3.8/dist-packages/carla-0.9.11-py3.7-linux-x86_64.egg
RUN echo "import sys; sys.__plen = len(sys.path)\n./carla-0.9.11-py3.7-linux-x86_64.egg\nimport sys; new=sys.path[sys.__plen:]; del sys.path[sys.__plen:]; p=getattr(sys,'__egginsert',0); sys.path[p:p]=new; sys.__egginsert = p+len(new)" > /usr/local/lib/python3.8/dist-packages/easy_install.pth

# Direct project dependencied are defined in pedestrians-video-2-carla/setup.cfg
# However, we want to leverage the cache, so we're going to specify at least basic ones with versions here
RUN pip install --no-cache-dir \
    cameratransform==1.1 \
    numpy==1.21 \
    Pillow==8.3 \
    scipy==1.7

# Create non-root user
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=carla-pedestrians-client
ENV HOME /home/${USERNAME}

RUN groupadd -g ${GROUP_ID} ${USERNAME} \
    && useradd -ms /bin/bash -u ${USER_ID} -g ${GROUP_ID} ${USERNAME} \
    && echo "${USERNAME}:${USERNAME}" | chpasswd \
    && mkdir ${HOME}/.vscode-server ${HOME}/.vscode-server-insiders /outputs \
    && chown ${USERNAME}:${USERNAME} ${HOME}/.vscode-server ${HOME}/.vscode-server-insiders /outputs
USER carla-pedestrians-client

# Run infinite loop to allow easily attach to container
CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]
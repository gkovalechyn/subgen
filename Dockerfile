ARG CUDA_VER=11.8.0-cudnn8-runtime-ubuntu20.04

FROM nvidia/cuda:${CUDA_VER}

WORKDIR /subgen

ADD requirements.txt /subgen/requirements.txt

RUN apt-get update \
    && apt-get install -y \
        python3 \
        python3-pip \
        ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install -r requirements.txt

ENV PYTHONUNBUFFERED=1

ADD launcher.py /subgen/launcher.py
ADD subgen.py /subgen/subgen.py

CMD [ "bash", "-c", "python3 -u launcher.py" ]

ARG CUDA_VER=12.2.2-runtime-ubuntu22.04

FROM nvidia/cuda:${CUDA_VER}
WORKDIR /subgen

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
        python3 \
        python3-pip \
        ffmpeg \
        pkg-config \
        libavformat-dev \
        libavcodec-dev \
        libavdevice-dev \
        libavutil-dev \
        libswscale-dev \
        libswresample-dev \
        libavfilter-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/

ADD requirements.txt /subgen/requirements.txt

RUN pip3 install --upgrade pip setuptools \
    && pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-build-isolation openai-whisper \
    && pip3 install -r requirements.txt

ENV PYTHONUNBUFFERED=1

ADD subgen.py /subgen/subgen.py

ENTRYPOINT [ "/bin/bash", "-c" ]
CMD [ "python3 -u subgen.py" ]

#
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04


# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list


RUN apt-get update && apt-get install -y \
	curl \
    ca-certificates \
	sudo \
	git \
	bzip2 \
	libx11-6 \
	libpng-dev \
	wget \
	ffmpeg \
	python3-pip \
	python3-dev \
	cmake \
	build-essential \
	pkg-config \
   && rm -rf /var/lib/apt/lists/*

RUN pip install uv

# Create a working directory.
RUN mkdir /app
WORKDIR /app

## download & compile VTM-23.3
RUN mkdir /app/vtm
WORKDIR /app/vtm
RUN wget https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/archive/VTM-23.3/VVCSoftware_VTM-VTM-23.3.tar.gz
RUN tar xvf VVCSoftware_VTM-VTM-23.3.tar.gz
WORKDIR /app/vtm/VVCSoftware_VTM-VTM-23.3/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN make -j

WORKDIR /app
RUN git clone https://github.com/InterDigitalInc/CompressAI-Vision compressai_vision_repo && \
    cd compressai_vision_repo && \
    git submodule update --init --recursive

WORKDIR /app/compressai_vision_repo
RUN uv python pin 3.8.20
RUN uv venv
RUN uv sync --only-group=dev
RUN uv run -- bash scripts/install_uv.sh --no-weights

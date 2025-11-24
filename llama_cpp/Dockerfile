FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y && \
    apt install -y --no-install-recommends \
        ca-certificates \
        git \
        build-essential \
        cmake \
        wget \
        curl \
        libcurl4-openssl-dev && \
    apt clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /opt/llama.cpp

RUN cmake -B build -DLLAMA_SERVER=ON
RUN cmake --build build --config Release -j$(nproc)

RUN wget -O /opt/model.gguf \
    https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf

WORKDIR /opt/llama.cpp/build/bin

EXPOSE 11343

ENTRYPOINT ["./llama-server"]
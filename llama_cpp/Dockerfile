FROM nvidia/cuda:13.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /srv

RUN apt-get update && apt-get install -y \
    make git cmake clang libomp-dev \
    curl ca-certificates libssl-dev zlib1g-dev xz-utils \
 && rm -rf /var/lib/apt/lists/*

ENV CC=clang
ENV CXX=clang++
ENV CURL_VERSION=8.17.0


RUN curl -LO https://curl.se/download/curl-${CURL_VERSION}.tar.xz \
 && tar -xf curl-${CURL_VERSION}.tar.xz \
 && cd curl-${CURL_VERSION} \
 && ./configure --prefix=/usr/local --disable-shared --enable-static \
    --with-openssl --with-zlib --disable-manual \
    --without-libpsl --without-nghttp2 --without-brotli --without-zstd \
    --disable-dict --disable-file --disable-ftp --disable-gopher \
    --disable-imap --disable-ipfs --disable-ldap --disable-mqtt --disable-pop3 \
    --disable-rtsp --disable-smb --disable-smtp --disable-telnet --disable-tftp \
 && make -j$(nproc) \
 && make install

ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV LDFLAGS="-L/usr/local/lib -lssl -lcrypto -lz"

RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /srv/llama.cpp

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_CURL=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
 && cmake --build build -j6


RUN curl -L -o /srv/model.gguf \
    https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q6_K.gguf


FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ca-certificates libomp5 \
 && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/lib

COPY --from=builder /usr/local/cuda/lib64/libcublas.so.13 /usr/local/lib/
COPY --from=builder /usr/local/cuda/lib64/libcublasLt.so.13 /usr/local/lib/
COPY --from=builder /usr/local/cuda/lib64/libcudart.so.13 /usr/local/lib/

COPY --from=builder /srv/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /srv/llama.cpp/build/bin/llama-cli /usr/local/bin/llama-cli

COPY --from=builder /srv/model.gguf /opt/model.gguf

WORKDIR /opt
EXPOSE 11343

ENTRYPOINT ["llama-server"]
CMD ["--model", "/opt/model.gguf", "--host", "0.0.0.0", "--port", "11343", "--n-gpu-layers", "999"]

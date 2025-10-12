ARG CUDA_VERSION=12.8.0
ARG from=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

FROM ${from} AS base

ARG DEBIAN_FRONTEND=noninteractive

# 安裝系統依賴 (分層優化 - 不常變動的基礎包)
RUN apt update -y && apt upgrade -y && apt install -y --no-install-recommends \
    git \
    git-lfs \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    curl \
    vim \
    libsndfile1 \
    ccache \
    software-properties-common \
    ffmpeg \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt clean

# 安裝 CMake (單獨層,避免每次重建)
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.1/cmake-3.26.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.26.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.26.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.26.1/bin/* /usr/local/bin

# 創建 Python 符號鏈接
RUN ln -s /usr/bin/python3 /usr/bin/python

# 初始化 Git LFS
RUN git lfs install

FROM base AS dev

WORKDIR /app

# ==================== Python 依賴安裝 (分層優化) ====================

# 第一層: 基礎依賴 (最不常變動)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir networkx==3.1

# 第二層: PyTorch 生態 (大型依賴,單獨層)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    torch==2.8.0 \
    torchvision==0.23.0 \
    torchaudio==2.8.0 \
    xformers==0.0.32.post2

# 第三層: Transformers 相關
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    transformers==4.52.3 \
    accelerate \
    qwen-omni-utils

# 第四層: Flash Attention (可選,耗時編譯)
ARG BUNDLE_FLASH_ATTENTION=true
ENV MAX_JOBS=32
ENV NVCC_THREADS=2
ENV CCACHE_DIR=/root/.cache/ccache

RUN --mount=type=cache,target=/root/.cache/ccache \
    --mount=type=cache,target=/root/.cache/pip \
    if [ "$BUNDLE_FLASH_ATTENTION" = "true" ]; then \
        pip install --no-cache-dir flash-attn --no-build-isolation; \
    fi

# 第五層: 應用依賴
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    setuptools_scm \
    torchdiffeq \
    resampy \
    x_transformers

# 第六層: Web 和音頻處理
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    gradio==5.23.1 \
    gradio_client==1.8.0 \
    librosa==0.11.0 \
    ffmpeg==1.4 \
    ffmpeg-python==0.2.0 \
    soundfile==0.13.1 \
    Flask==3.0.3 \
    modelscope_studio

# 第七層: 量化庫
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    autoawq==0.2.9 \
    gptqmodel==2.0.0 \
    numpy==2.2.2

# 第八層: 簡繁轉換和安全工具
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    opencc-python-reimplemented \
    python-magic

# 第九層: 其他應用依賴
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir \
    scikit-build-core[pyproject] pathspec pybind11

# ==================== 應用程式安裝 ====================

FROM dev AS bundle_app

# 創建應用目錄結構
RUN mkdir -p /app/serve /app/inputs /app/temp /app/outputs && \
    chmod -R 777 /app/inputs /app/temp /app/outputs

# 複製應用代碼 (放在最後,因為經常變動)
COPY app/serve ./serve
COPY low-VRAM-mode ./serve/low-VRAM-mode
COPY qwen-omni-utils ./qwen-omni-utils

# 安裝 qwen-omni-utils
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir -e ./qwen-omni-utils

# 設置環境變數
ENV PYTHONPATH=/app:/app/low-VRAM-mode:/app/serve/low-VRAM-mode
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 80 5000 5001

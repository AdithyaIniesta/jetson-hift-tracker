# =============================================================================
# JetsonTracker — JetPack 5 (L4T R35.4.1) | Orin NX
# Stack: Ubuntu 20.04 | CUDA 11.4 | OpenCV 4.5.4 | GStreamer 1.16.3
#        nvv4l2h264enc (HW) | nvv4l2decoder (HW) | nvvidconv
# Note: NVIDIA multimedia stack already included in l4t-jetpack base image
# Stages:
#   base    — L4T + system libs + OpenCV built from source
#   dev     — base + build tools
#   runtime — base + binary + assets
# =============================================================================

ARG L4T_BASE=nvcr.io/nvidia/l4t-jetpack:r35.4.1
ARG OPENCV_VERSION=4.5.4
ARG OPENCV_CUDA_ARCH=8.7
ARG MAKE_JOBS=2

# =============================================================================
# STAGE 1 — base
# =============================================================================
FROM ${L4T_BASE} AS base

ARG OPENCV_VERSION
ARG OPENCV_CUDA_ARCH
ARG MAKE_JOBS

ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH} \
    PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:/usr/lib/aarch64-linux-gnu/pkgconfig:${PKG_CONFIG_PATH:-} \
    GST_PLUGIN_PATH=/usr/lib/aarch64-linux-gnu/gstreamer-1.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    v4l-utils \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    libgtk-3-dev \
    libeigen3-dev \
    libatlas-base-dev \
    libopenblas-dev \
    libusb-1.0-0-dev \
    libomp-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ldconfig

# Build OpenCV from source with CUDA
RUN mkdir -p /tmp/ocv \
    && wget -q "https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.tar.gz" \
            -O /tmp/ocv/opencv.tar.gz \
    && wget -q "https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.tar.gz" \
            -O /tmp/ocv/contrib.tar.gz \
    && tar xf /tmp/ocv/opencv.tar.gz -C /tmp/ocv \
    && tar xf /tmp/ocv/contrib.tar.gz -C /tmp/ocv \
    && cmake -G Ninja \
        -S /tmp/ocv/opencv-${OPENCV_VERSION} \
        -B /tmp/ocv/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DOPENCV_EXTRA_MODULES_PATH=/tmp/ocv/opencv_contrib-${OPENCV_VERSION}/modules \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DWITH_CUDA=ON \
        -DWITH_CUDNN=ON \
        -DOPENCV_DNN_CUDA=ON \
        -DENABLE_FAST_MATH=ON \
        -DCUDA_FAST_MATH=ON \
        -DWITH_CUBLAS=ON \
        -DCUDA_ARCH_BIN=${OPENCV_CUDA_ARCH} \
        -DCUDA_ARCH_PTX="" \
        -DWITH_GSTREAMER=ON \
        -DWITH_V4L=ON \
        -DWITH_FFMPEG=ON \
        -DWITH_EIGEN=ON \
        -DENABLE_NEON=ON \
        -DBUILD_opencv_python3=OFF \
        -DBUILD_opencv_python2=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_DOCS=OFF \
    && ninja -C /tmp/ocv/build -j${MAKE_JOBS} \
    && ninja -C /tmp/ocv/build install \
    && ldconfig \
    && rm -rf /tmp/ocv \
    && pkg-config --modversion opencv4 \
    && echo "[OK] OpenCV ${OPENCV_VERSION} installed"

WORKDIR /workspace

# =============================================================================
# STAGE 2 — dev
# =============================================================================
FROM base AS dev

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    git curl gdb \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["bash"]

# =============================================================================
# STAGE 3 — runtime
# =============================================================================
FROM base AS runtime

ENV JVP_ROOT=/opt/jvp \
    LD_LIBRARY_PATH=/opt/jvp/lib:/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}

RUN mkdir -p /opt/jvp/bin /opt/jvp/lib /opt/jvp/config

WORKDIR /opt/jvp
CMD ["/opt/jvp/bin/JetsonTracker"]

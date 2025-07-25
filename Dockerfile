# RTSP Object Detection System - DeepStream Container
# Based on NVIDIA DeepStream SDK with GPU support

# Use NVIDIA DeepStream base image with Ubuntu 20.04
FROM nvcr.io/nvidia/deepstream:6.4-gc-triton-devel AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libglib2.0-dev \
    libgstrtspserver-1.0-dev \
    libgirepository1.0-dev \
    pkg-config \
    cmake \
    build-essential \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Build the Kinesis Video Streams Producer GStreamer Plugin
FROM base AS build-kvssink

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates \
    libssl-dev \
    libcurl4-openssl-dev \
    liblog4cplus-dev \
    byacc \
    make \
    cmake \
    curl \
    g++ \
    git \
    gstreamer1.0-plugins-base-apps \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    m4 \
    pkg-config \
    xz-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone https://github.com/awslabs/amazon-kinesis-video-streams-producer-sdk-cpp.git -b v3.4.2

WORKDIR /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/

RUN cmake .. -DBUILD_DEPENDENCIES=OFF -DBUILD_GSTREAMER_PLUGIN=ON && \
    make -j $(nproc)

# Verify that the KVS Gstreamer plugin was built successfully.
RUN ldd /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/libgstkvssink.so

FROM base

# Install runtime dependencies for the Kinesis Video Streams Producer GStreamer Plugin
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
    liblog4cplus-dev \
    libcurl4 && \
    rm -rf /var/lib/apt/lists/*

# Import the Kinesis Video Streams Producer GStreamer Plugin into the GStreamer plugin path.
COPY --from=build-kvssink /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/libgstkvssink.so /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/
COPY --from=build-kvssink /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/libKinesisVideoProducer.so /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/
COPY --from=build-kvssink /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/dependency/libkvscproducer/kvscproducer-src/libcproducer.so* /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/dependency/libkvscproducer/kvscproducer-src/
COPY --from=build-kvssink /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/dependency/libkvscproducer/kvscproducer-src/libkvsCommonCurl.so* /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/dependency/libkvscproducer/kvscproducer-src/

ENV GST_PLUGIN_PATH=/opt/amazon-kinesis-video-streams-producer-sdk-cpp/build

# Verify that the Kinesis Video Streams Producer GStreamer Plugin has required dependencies.
RUN ldd /opt/amazon-kinesis-video-streams-producer-sdk-cpp/build/libgstkvssink.so

# Verify that the Kinesis Video Streams Producer GStreamer Plugin is available in the GStreamer plugin path.
RUN gst-inspect-1.0 kvssink

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
# Use --ignore-installed to handle distutils installed packages like blinker
RUN pip3 install --no-cache-dir --ignore-installed -r /tmp/requirements.txt

# Create application directory
WORKDIR /app

# Copy application source code
COPY src/ /app/src/
COPY configs/ /app/configs/
COPY tests/ /app/tests/

# Create directories for data and models
RUN mkdir -p /app/models /app/data/frames /app/logs

# Install DeepStream Python bindings
USER root
RUN apt-get update && apt-get install -y python3-gi python3-dev python3-gst-1.0 && \
    echo "Checking DeepStream installation directories..." && \
    ls -la /opt/nvidia/deepstream/ || echo "DeepStream directory not found" && \
    # Install DeepStream Python bindings for Triton container
    cd /opt/nvidia/deepstream/deepstream && \
    if [ -f "/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/install.sh" ]; then \
        cd /opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps && \
        bash install.sh; \
    elif [ -d "/opt/nvidia/deepstream/deepstream/sources/python" ]; then \
        cd /opt/nvidia/deepstream/deepstream/sources/python && \
        python3 setup.py install; \
    else \
        # Find and install any pyds wheel files
        find /opt/nvidia/deepstream -name "pyds*.whl" -exec pip3 install --force-reinstall {} \; || \
        echo "No pyds wheel found, trying to build from source"; \
        # Clone and build the Python bindings if not available
        cd /opt && \
        git clone https://github.com/NVIDIA-AI-IOT/deepstream_python_apps.git && \
        cd deepstream_python_apps && \
        git submodule update --init && \
        cd bindings && \
        mkdir build && \
        cd build && \
        cmake .. -DPYTHON_MAJOR_VERSION=3 -DPYTHON_MINOR_VERSION=8 && \
        make -j$(nproc) && \
        pip3 install ./pyds-*.whl; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# Set up DeepStream Python bindings path with all possible locations
ENV PYTHONPATH="${PYTHONPATH}:/opt/nvidia/deepstream/deepstream/lib:/opt/nvidia/deepstream/deepstream/python:/opt/nvidia/deepstream/deepstream/sources/python:/usr/lib/python3/dist-packages:/opt/nvidia/deepstream/deepstream/sources/deepstream_python_apps/bindings"

# Create a simple test script to verify DeepStream paths
RUN echo '#!/usr/bin/env python3\nimport sys\nprint("Python version:", sys.version)\nprint("Python path:", sys.path)\ntry:\n    import pyds\n    print("pyds imported successfully")\nexcept ImportError as e:\n    print("Failed to import pyds:", e)\n    print("Searching for pyds...")\n    import os\n    for root, dirs, files in os.walk("/opt/nvidia"):\n        for file in files:\n            if "pyds" in file:\n                print(f"Found: {os.path.join(root, file)}")\n' > /tmp/find_pyds.py && \
    chmod +x /tmp/find_pyds.py && \
    python3 /tmp/find_pyds.py || echo "pyds import failed, but continuing build"

# Create non-root user for security
RUN groupadd -r deepstream && useradd -r -g deepstream deepstream
RUN chown -R deepstream:deepstream /app
USER deepstream

# Expose ports
EXPOSE 8554 8555 1883

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import sys; sys.path.append('/app/src'); from config_manager import ConfigManager; print('OK')" || exit 1

# Default command
CMD ["python3", "/app/src/pipeline_manager.py", "/app/configs/deepstream_app_config.txt", "/app/configs/app_config.yaml"]
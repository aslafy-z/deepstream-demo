# Implementation Plan

- [x] 1. Set up Python project structure and DeepStream configuration files
  - Create Python project directory structure with src/, configs/, models/, and tests/
  - Set up requirements.txt with DeepStream Python bindings and dependencies
  - Create DeepStream application config template (deepstream_app_config.txt)
  - Create detection model config template (detection_config.txt)
  - Set up minimal custom configuration file (app_config.yaml)
  - _Requirements: 8.1, 10.1, 10.4_

- [x] 2. Implement minimal DeepStream pipeline wrapper in Python
  - Create Python application using DeepStream Python bindings (pyds)
  - Implement GStreamer pipeline creation and management using Python GStreamer bindings
  - Add basic error handling and logging using Python logging framework
  - Set up DeepStream config file parsing using Python
  - _Requirements: 1.1, 8.1, 8.5_

- [x] 3. Add RTSP input stream handling in Python
  - Configure uridecodebin source in DeepStream config for RTSP input
  - Implement reconnection logic using Python threading and DeepStream callbacks
  - Add stream status monitoring through Python probe functions on DeepStream pipeline
  - Create Python-based stream health monitoring with logging
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 4. Configure object detection with ONNX/TensorRT models
  - Set up nvinfer plugin configuration for ONNX model loading
  - Configure TensorRT engine generation and caching
  - Implement model validation using DeepStream's model loading callbacks
  - Add detection confidence and class filtering through DeepStream config
  - _Requirements: 2.1, 2.2, 2.3, 2.4_

- [x] 5. Enable object tracking with DeepStream nvtracker
  - Configure nvtracker plugin with appropriate tracking algorithm
  - Set up tracking configuration file (tracker_config.yml)
  - Implement tracking ID consistency through DeepStream's tracking callbacks
  - Add tracking failure handling using DeepStream's metadata system
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [x] 6. Implement behavior analysis using Python probe functions
  - Create Python probe function to analyze object metadata from DeepStream pipeline
  - Implement object appearance detection using tracking ID changes in Python
  - Add static object detection logic with configurable thresholds using Python classes
  - Create Python event generation system that processes DeepStream metadata
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 7. Add event delivery system in Python
  - Implement HTTP POST event delivery using Python requests library with retry logic
  - Add MQTT event publishing using Python paho-mqtt client
  - Create JSON event formatting using Python json library and structured event schema
  - Implement exponential backoff for failed deliveries using Python threading
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 8. Configure video output with annotations
  - Set up nvdsosd plugin for bounding box and label overlays
  - Configure RTSP output sink in DeepStream config
  - Add annotation customization through DeepStream OSD settings
  - Implement conditional annotation display based on detection presence
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

- [x] 9. Add Kinesis Video Streams output
  - Configure Kinesis Video Streams sink in DeepStream config
  - Set up AWS credentials and region configuration
  - Implement stream health monitoring through DeepStream sink callbacks
  - Add error handling for Kinesis connection failures
  - _Requirements: 7.3_

- [x] 10. Implement detection-only frame saving in Python
  - Create Python probe function to filter frames based on detection presence
  - Add frame saving logic using OpenCV or PIL that preserves timestamps
  - Implement configurable output path and file naming using Python pathlib
  - Add storage optimization through detection-based filtering with Python file operations
  - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [x] 11. Add configuration management in Python
  - Implement YAML configuration parser using PyYAML for custom settings
  - Add configuration validation using Python dataclasses or pydantic
  - Create configuration hot-reload capability using Python watchdog library
  - Integrate custom config with DeepStream's native configuration system using Python
  - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

- [x] 12. Create Docker container with GPU support
  - Create Dockerfile based on DeepStream container image
  - Configure CUDA and TensorRT runtime in container
  - Add GPU device access and driver compatibility
  - Test container deployment on both Jetson and desktop GPU platforms
  - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 13. Add comprehensive error handling and logging
  - Implement structured logging using DeepStream's logging framework
  - Add pipeline error recovery mechanisms
  - Create health monitoring and status reporting
  - Add graceful shutdown handling for all components
  - _Requirements: 1.3, 2.4, 3.4, 6.4_

- [x] 14. Create integration tests
  - Set up test RTSP stream sources for validation
  - Create automated tests for model loading and inference
  - Add event delivery testing with mock endpoints
  - Implement container deployment testing
  - _Requirements: All requirements validation_

- [x] 15. Add performance monitoring and optimization
  - Implement DeepStream performance measurement integration
  - Add GPU memory usage monitoring
  - Create throughput and latency metrics collection
  - Optimize pipeline configuration for target hardware
  - _Requirements: 1.2, 2.5, 9.2, 9.3_

- [x] 16. Replace placeholders like "This is a placeholder" or "placeholder implementation" with real implementations


- [x] 17. add a way to start the project with a rtsp flow created by something like everything within a docker compose and network and ffmpeg in docker too, also the deepstream.

    docker run --rm -e MTX_RTSPTRANSPORTS=tcp -e MTX_PATHSDEFAULTS_SOURCE=publisher -e MTX_WRITEQUEUESIZE=8388608 -e MTX_WEBRTCADDITIONALHOSTS=0.0.0.0 -p 8554:8554 -p 8888:8888 bluenviron/mediamtx:latest
    ffmpeg -re -stream_loop -1 -i ./input.mp4  \
        -c:v libx264 -filter:v fps=8  \
        -g 1 -keyint_min 1 -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/live.stream

- [x] 18. I see there is a index.html with fake metrics, make them real metrics. also, add iframes with the input and output stream hls outputs (no custom player).
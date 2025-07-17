# Technology Stack

## Core Technologies

- **NVIDIA DeepStream SDK**: GStreamer-based video analytics framework with GPU acceleration
- **Python 3**: Primary development language with DeepStream Python bindings (pyds)
- **GStreamer**: Multimedia framework for pipeline-based video processing
- **CUDA/TensorRT**: GPU acceleration and AI model optimization
- **Docker**: Containerization with GPU support

## Key Dependencies

- `pyds`: DeepStream Python bindings (installed with DeepStream SDK)
- `PyYAML`: Configuration file parsing
- `pydantic`: Configuration validation and data models
- `opencv-python`: Image processing for frame operations
- `requests`: HTTP event delivery
- `paho-mqtt`: MQTT event publishing
- `watchdog`: Configuration hot-reloading
- `structlog`: Structured logging

## Development & Testing

### Common Commands

```bash
# Install Python dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run specific test
python tests/test_config_manager.py

# Run main application
python src/main.py --config configs/deepstream_app_config.txt

# Run with custom RTSP URI
python src/main.py --config configs/deepstream_app_config.txt --rtsp-uri rtsp://example.com/stream

# Run with debug logging
python src/main.py --log-level DEBUG
```

### Prerequisites

- NVIDIA DeepStream SDK must be installed
- GPU drivers and CUDA toolkit
- Python 3.8+ with DeepStream Python bindings

## Architecture Patterns

- **Plugin-based pipeline**: Uses DeepStream's GStreamer plugin architecture
- **Probe-based processing**: Custom logic implemented via GStreamer pad probes
- **Event-driven**: Asynchronous event generation and delivery
- **Configuration-driven**: Behavior controlled via YAML files with hot-reloading
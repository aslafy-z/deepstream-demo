# RTSP Object Detection System

A production-ready, real-time RTSP object detection system built on NVIDIA DeepStream that processes video streams using AI models, analyzes object behaviors, and generates events based on configurable criteria. The system provides comprehensive monitoring, testing, and deployment capabilities.

## 🚀 Features

### Core Capabilities
- **Real-time RTSP Processing**: GPU-accelerated video stream processing with minimal latency
- **AI Object Detection**: Support for ONNX and TensorRT models via DeepStream with COCO dataset labels
- **Object Tracking**: Persistent object IDs across frames using DeepStream's nvtracker
- **Behavior Analysis**: Configurable detection of object appearance, static objects, and movement patterns
- **Multi-Protocol Events**: HTTP POST and MQTT event delivery with exponential backoff retry logic
- **Video Re-streaming**: Annotated RTSP output with customizable detection overlays

### Advanced Features
- **Performance Monitoring**: Real-time GPU/CPU usage, FPS, latency, and memory tracking
- **Health Monitoring**: Component status tracking with automatic error recovery
- **Structured Logging**: JSON logging with DeepStream integration and log rotation
- **Frame Saving**: Detection-only frame capture with timestamp preservation
- **AWS Integration**: Kinesis Video Streams support for cloud streaming
- **Hot-Reload Configuration**: YAML-based configuration with automatic reloading and validation

### Deployment & Operations
- **Multi-Platform Containers**: Docker support for desktop GPUs and Jetson devices
- **Complete Docker Stack**: Full ecosystem with RTSP server, FFmpeg, MQTT broker, and monitoring
- **Web Dashboard**: Real-time system monitoring and control interface
- **Integration Testing**: Comprehensive test suite for all components
- **Production Ready**: Health checks, logging, metrics, and graceful shutdown

## 🚀 One-Click Quick Start

**Complete RTSP Object Detection System with YOLO Model - Copy & Paste Ready:**

```bash
# Clone and start the complete system (includes RTSP server, FFmpeg, DeepStream, MQTT, monitoring)
git clone <repository-url> && cd rtsp-object-detection-system

# Download YOLOv5 model (or place your own ONNX/TensorRT model in models/)
mkdir -p models && wget -O models/yolov5s.onnx https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx

# Start complete system with one command
./start-system.sh start

# 🎉 System ready! Access points:
# 📺 Web Dashboard: http://localhost:8090
# 🎥 Input Stream: rtsp://localhost:8554/live.stream  
# 🔍 Detection Output: rtsp://localhost:8555/detection.stream
# 📊 Monitoring: http://localhost:3000 (Grafana)
# 🔌 MQTT Events: mqtt://localhost:1883/detection/events
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA support (desktop) or NVIDIA Jetson device
- Docker with NVIDIA Container Runtime
- Docker Compose

### Method 1: Complete System Startup (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd rtsp-object-detection-system

# Start complete system with comprehensive startup script
./start-system.sh start

# For development environment with additional monitoring
./start-system.sh start-dev
```

### Method 2: Manual Docker Compose

```bash
# Clone and setup
git clone <repository-url>
cd rtsp-object-detection-system

# Auto-detect platform and build
./build.sh build auto

# Start full stack
docker-compose up -d

# View logs
./build.sh logs
```

### Access Points

- **🌐 Web Dashboard**: `http://localhost:8090` - System monitoring and control
- **🎥 RTSP Input Stream**: `rtsp://localhost:8554/live.stream`
- **🔍 Detection Output**: `rtsp://localhost:8555/detection.stream`
- **📺 MediaMTX Web UI**: `http://localhost:8888` - Stream management
- **📊 Grafana Dashboard**: `http://localhost:3000` - Performance monitoring (admin/admin)
- **📈 Prometheus Metrics**: `http://localhost:9091` - System metrics
- **🔌 MQTT Broker**: `mqtt://localhost:1883` - Event streaming

### Quick Commands

```bash
# View system status
./start-system.sh status

# Show all access points and test commands
./start-system.sh info

# View logs for specific service
./start-system.sh logs deepstream-app

# Stop system
./start-system.sh stop

# Complete cleanup
./start-system.sh cleanup
```

## Configuration

### Application Configuration (`configs/app_config.yaml`)

```yaml
behavior:
  static_threshold_seconds: 30      # Time before object considered static
  position_tolerance_pixels: 10     # Movement tolerance for static detection
  debounce_seconds: 2              # Debounce time for events
  min_confidence: 0.5              # Minimum detection confidence

events:
  http:
    endpoint: "http://api.example.com/events"
    timeout_seconds: 5
    retry_attempts: 3
  mqtt:
    broker: "mqtt.example.com"
    topic: "detection/events"
    port: 1883
    qos: 1

frame_saving:
  detection_only_mode: false       # Save only frames with detections
  output_path: "/data/frames"
  max_frames_per_hour: 3600
  image_format: "jpg"
  image_quality: 95
```

### DeepStream Configuration (`configs/deepstream_app_config.txt`)

Standard DeepStream configuration file for pipeline setup, model configuration, and output settings.

## Development

### Project Structure

```
├── src/                    # Python source code
│   ├── pipeline_manager.py    # Main pipeline orchestration
│   ├── config_manager.py      # Configuration management
│   ├── behavior_analyzer.py   # Object behavior analysis
│   ├── event_dispatcher.py    # Event delivery system
│   └── ...
├── configs/               # Configuration files
├── tests/                # Test files
├── models/               # AI model files (ONNX/TensorRT)
├── data/                 # Output data and frames
└── logs/                 # Application logs
```

### Running Tests

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Run specific test
python tests/test_config_manager.py
```

### Development Commands

```bash
# Build development container
./build.sh build desktop

# Run with shell access
./build.sh shell

# View real-time logs
./build.sh logs

# Clean up containers
./build.sh clean
```

## Deployment

### Desktop GPU Deployment

```bash
# Build and run
./build.sh build desktop
./build.sh run desktop
```

### Jetson Device Deployment

```bash
# Build and run on Jetson
./build.sh build jetson
./build.sh run jetson
```

### Production Deployment

1. **Prepare Models**: Place ONNX/TensorRT model files in `models/` directory
2. **Configure System**: Update `configs/app_config.yaml` and `configs/deepstream_app_config.txt`
3. **Set RTSP Source**: Update RTSP URI in configuration or environment variables
4. **Deploy**: Use Docker Compose for production deployment

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d
```

## Monitoring

### Health Checks

The container includes built-in health checks:

```bash
# Check container health
docker ps --format "table {{.Names}}\t{{.Status}}"
```

### Logs

```bash
# Application logs
./build.sh logs

# Specific component logs
docker-compose logs deepstream-app
docker-compose logs mediamtx
```

### Performance Monitoring

- **Real-time Metrics**: Built-in performance monitoring with GPU/CPU usage, FPS, and latency
- **Web Dashboard**: `http://localhost:8090` - Real-time system status and metrics
- **Grafana Dashboard**: `http://localhost:3000` - Advanced performance visualization
- **Prometheus Metrics**: `http://localhost:9091` - Metrics collection and alerting
- **GPU utilization**: `nvidia-smi` - Direct GPU monitoring
- **Container stats**: `docker stats` - Container resource usage

## Troubleshooting

### Common Issues

1. **GPU Not Detected**
   ```bash
   # Check NVIDIA Docker runtime
   docker info | grep nvidia
   
   # Test GPU access
   docker run --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
   ```

2. **RTSP Stream Issues**
   ```bash
   # Test RTSP connectivity
   ffplay rtsp://localhost:8554/live.stream
   
   # Check stream status
   curl http://localhost:8888/v3/paths/list
   ```

3. **Model Loading Errors**
   - Ensure model files are in `models/` directory
   - Check model compatibility with DeepStream version
   - Verify GPU memory availability

4. **Configuration Errors**
   - Validate YAML syntax
   - Check file permissions
   - Review application logs for validation errors

### Debug Mode

```bash
# Run with debug logging
docker run -it --rm --gpus all \
  -e LOG_LEVEL=DEBUG \
  -v $(pwd)/configs:/app/configs \
  deepstream-detection:latest
```

## API Reference

### Event Schema

```json
{
  "event_id": "uuid",
  "event_type": "object_appeared|object_static|object_moving",
  "timestamp": "ISO8601",
  "object": {
    "tracking_id": "integer",
    "class_name": "string",
    "position": {"x": "float", "y": "float"}
  },
  "metadata": {
    "duration": "integer",
    "confidence": "float"
  }
}
```

### Configuration Hot-Reload

The system supports hot-reloading of configuration files. Simply modify `configs/app_config.yaml` and the system will automatically apply changes without restart.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes and add tests
4. Submit a pull request

## License

[License information]

## Support

For issues and questions:
- Check the troubleshooting section
- Review application logs
- Open an issue on GitHub
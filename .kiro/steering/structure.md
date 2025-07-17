# Project Structure

## Directory Organization

```
├── src/                    # Main source code
├── configs/               # Configuration files
├── tests/                 # Test files
├── .kiro/                # Kiro IDE settings and specs
├── requirements.txt       # Python dependencies
├── videos/               # Video files directory
│   └── input.mp4        # Sample input file
└── models/              # AI model files (gitignored)
```

## Source Code Structure (`src/`)

### Core Components

- `main.py`: Application entry point with CLI argument parsing
- `pipeline_manager.py`: DeepStream pipeline orchestration and management
- `config_manager.py`: YAML configuration loading, validation, and hot-reloading

### Processing Modules

- `stream_manager.py`: RTSP stream handling and input management
- `model_manager.py`: AI model loading and inference configuration
- `tracking_manager.py`: Object tracking with persistent IDs
- `behavior_analyzer.py`: Object behavior analysis and event generation

### Output & Integration

- `event_dispatcher.py`: Multi-protocol event delivery (HTTP, MQTT)
- `video_output_manager.py`: Annotated video re-streaming
- `frame_saver.py`: Frame capture and storage
- `kinesis_manager.py`: AWS Kinesis Video Streams integration

## Configuration Structure (`configs/`)

- `app_config.yaml`: Custom application behavior settings
- `deepstream_app_config.txt`: DeepStream pipeline configuration
- `detection_config.txt`: AI model detection parameters
- `tracker_config.yml`: Object tracking configuration
- `labels.txt`: Object class labels

## Coding Conventions

### Module Organization
- Each module handles a specific aspect of the pipeline
- Modules communicate via DeepStream metadata and probe functions
- Configuration is centralized and validated using Pydantic models

### Error Handling
- Use structured logging with `structlog`
- Graceful degradation when optional features fail
- DeepStream error handling via GStreamer bus messages

### Testing
- Unit tests in `tests/` directory
- Test configuration loading and validation
- Mock DeepStream components for isolated testing
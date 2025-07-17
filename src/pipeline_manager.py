#!/usr/bin/env python3

"""
DeepStream Pipeline Manager
Minimal wrapper for DeepStream GStreamer pipeline using Python bindings
"""

import sys
import gi
import logging
from pathlib import Path
import structlog

gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

try:
    import pyds
except ImportError:
    print("ERROR: DeepStream Python bindings (pyds) not found. Please install DeepStream SDK.")
    sys.exit(1)

from stream_manager import RTSPStreamManager, StreamProbeHandler
from model_manager import ModelManager, create_detection_model_config
from tracking_manager import TrackingManager, tracking_src_pad_buffer_probe
from behavior_analyzer import BehaviorAnalyzer, behavior_analysis_probe
from event_dispatcher import EventDispatcher
from video_output_manager import VideoOutputManager, video_output_probe
from kinesis_manager import KinesisVideoStreamsManager
from frame_saver import FrameSaver, frame_saving_probe
from config_manager import ConfigManager
from logging_config import setup_logging, health_monitor, error_recovery_manager
from performance_monitor import initialize_performance_monitoring, get_performance_manager

# Initialize GStreamer
Gst.init(None)

class DeepStreamPipelineManager:
    """Minimal DeepStream pipeline manager using Python bindings"""
    
    def __init__(self, config_file_path: str, app_config_path: str = "configs/app_config.yaml", rtsp_uri: str = None):
        """
        Initialize pipeline manager with DeepStream config file
        
        Args:
            config_file_path: Path to DeepStream application config file
            app_config_path: Path to custom application config YAML file
            rtsp_uri: Optional RTSP URI to override config file
        """
        self.config_file_path = Path(config_file_path)
        self.pipeline = None
        self.loop = None
        self.bus = None
        self.logger = structlog.get_logger(__name__)
        
        # Configuration management
        self.config_manager = ConfigManager(app_config_path)
        self.config_manager.add_change_listener(self._on_config_changed)
        
        # RTSP stream management
        self.rtsp_uri = rtsp_uri
        self.stream_manager = None
        self.stream_probe_handler = None
        
        # Model management
        self.model_manager = ModelManager()
        
        # Tracking management
        self.tracking_manager = None
        
        # Behavior analysis
        self.behavior_analyzer = None
        
        # Event dispatching
        self.event_dispatcher = None
        
        # Video output management
        self.video_output_manager = None
        
        # Kinesis Video Streams management
        self.kinesis_manager = None
        
        # Frame saving management
        self.frame_saver = None
        
        # Performance monitoring
        self.performance_manager = None
        
        # Validate config file exists
        if not self.config_file_path.exists():
            raise FileNotFoundError(f"DeepStream config file not found: {config_file_path}")
        
        # Initialize components
        self._init_models()
        self._init_tracking()
        self._init_event_dispatcher()
        self._init_video_output()
        self._init_kinesis()
        self._init_frame_saver()
        self._init_behavior_analysis()
        self._init_performance_monitoring()
        self._init_web_server()
        if self.rtsp_uri:
            self._init_stream_manager()
    
    def _init_models(self):
        """Initialize AI models for detection"""
        try:
            # Create detection model configuration
            detection_config = create_detection_model_config(
                onnx_file="/models/detection.onnx",
                engine_file="/models/detection.engine",
                labels_file="/configs/labels.txt",
                config_file="/configs/detection_config.txt",
                num_classes=80,
                batch_size=1,
                gpu_id=0
            )
            
            # Register the primary detection model
            success = self.model_manager.register_model("primary_detector", detection_config)
            
            if success:
                self.logger.info("Primary detection model registered successfully")
                
                # Validate GPU memory
                memory_info = self.model_manager.validate_gpu_memory(gpu_id=0)
                if not memory_info.get('sufficient_memory', False):
                    self.logger.warning("GPU memory may be insufficient for model inference")
                    
            else:
                self.logger.error("Failed to register primary detection model")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {e}")
            # Don't raise exception here - allow pipeline to continue without models for now
    
    def _init_tracking(self):
        """Initialize object tracking manager"""
        try:
            self.tracking_manager = TrackingManager("configs/tracker_config.yml")
            self.logger.info("Tracking manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize tracking manager: {e}")
            # Don't raise exception here - allow pipeline to continue without tracking for now
    
    def _init_event_dispatcher(self):
        """Initialize event dispatcher"""
        try:
            # Load event delivery config from configuration manager
            event_config = self.config_manager.get_events_config()
            
            self.event_dispatcher = EventDispatcher(event_config)
            self.logger.info("Event dispatcher initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize event dispatcher: {e}")
            # Don't raise exception here - allow pipeline to continue without event delivery for now
    
    def _init_video_output(self):
        """Initialize video output manager"""
        try:
            # Load video output config (could be from YAML file)
            video_config = {
                'rtsp': {
                    'enabled': True,
                    'port': 8554,
                    'formats': ['rtsp']
                },
                'annotations': {
                    'show_bounding_boxes': True,
                    'show_tracking_ids': True,
                    'show_class_labels': True,
                    'show_confidence': True,
                    'colors': {
                        'person': (1.0, 0.0, 0.0, 1.0),      # Red
                        'car': (0.0, 1.0, 0.0, 1.0),         # Green
                        'truck': (0.0, 0.0, 1.0, 1.0),       # Blue
                        'bus': (1.0, 1.0, 0.0, 1.0),         # Yellow
                        'bicycle': (1.0, 0.0, 1.0, 1.0),     # Magenta
                        'motorbike': (0.0, 1.0, 1.0, 1.0),   # Cyan
                        'default': (1.0, 1.0, 1.0, 1.0)      # White
                    },
                    'border_width': 3,
                    'text_size': 16
                },
                'conditional_display': {
                    'detection_only': False,
                    'min_confidence': 0.5
                }
            }
            
            self.video_output_manager = VideoOutputManager(video_config)
            self.logger.info("Video output manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize video output manager: {e}")
            # Don't raise exception here - allow pipeline to continue without video output management for now
    
    def _init_kinesis(self):
        """Initialize Kinesis Video Streams manager"""
        try:
            # Load Kinesis config (could be from YAML file)
            kinesis_config = {
                'stream_name': 'detection-stream',
                'aws_region': 'us-east-1',
                'retention_period_hours': 24
            }
            
            self.kinesis_manager = KinesisVideoStreamsManager(kinesis_config)
            self.logger.info("Kinesis Video Streams manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Kinesis manager: {e}")
            # Don't raise exception here - allow pipeline to continue without Kinesis for now
    
    def _init_frame_saver(self):
        """Initialize frame saver"""
        try:
            # Load frame saving config from configuration manager
            frame_config = self.config_manager.get_frame_saving_config()
            
            self.frame_saver = FrameSaver(frame_config)
            self.logger.info("Frame saver initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize frame saver: {e}")
            # Don't raise exception here - allow pipeline to continue without frame saving for now
    
    def _init_behavior_analysis(self):
        """Initialize behavior analysis"""
        try:
            # Load behavior config from configuration manager
            behavior_config = self.config_manager.get_behavior_config()
            
            self.behavior_analyzer = BehaviorAnalyzer(behavior_config)
            
            # Set up event callback for logging
            def log_event(event):
                self.logger.info(f"Behavior event: {event.event_type} - Track {event.tracking_id} - {event.class_name}")
            
            self.behavior_analyzer.add_event_callback(log_event)
            
            # Set up event callback for dispatching if event dispatcher is available
            if self.event_dispatcher:
                self.behavior_analyzer.add_event_callback(self.event_dispatcher.dispatch_event)
                self.logger.info("Behavior analyzer connected to event dispatcher")
            
            self.logger.info("Behavior analyzer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize behavior analyzer: {e}")
            # Don't raise exception here - allow pipeline to continue without behavior analysis for now
    
    def _init_performance_monitoring(self):
        """Initialize performance monitoring"""
        try:
            self.performance_manager = initialize_performance_monitoring(monitoring_interval=5.0)
            self.logger.info("Performance monitoring initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize performance monitoring: {e}")
            # Don't raise exception here - allow pipeline to continue without performance monitoring for now
    
    def _init_web_server(self):
        """Initialize web server for dashboard and metrics API"""
        try:
            from web_server import initialize_web_server
            
            self.web_server = initialize_web_server(
                host="0.0.0.0", 
                port=8080, 
                config_path=self.config_manager.config_path
            )
            
            # Start web server in background thread
            self.web_server.run_threaded(debug=False)
            self.logger.info("Web server initialized and started on port 8080")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize web server: {e}")
            # Don't raise exception here - allow pipeline to continue without web server for now
    
    def _init_stream_manager(self):
        """Initialize RTSP stream manager"""
        try:
            self.stream_manager = RTSPStreamManager(self.rtsp_uri)
            self.stream_probe_handler = StreamProbeHandler(self.stream_manager)
            
            # Set up stream event callbacks
            self.stream_manager.set_callbacks(
                on_connected=self._on_stream_connected,
                on_disconnected=self._on_stream_disconnected,
                on_reconnected=self._on_stream_reconnected
            )
            
            self.logger.info(f"RTSP stream manager initialized for: {self.rtsp_uri}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize stream manager: {e}")
            raise
    
    def _on_stream_connected(self):
        """Handle stream connection event"""
        self.logger.info("RTSP stream connected successfully")
    
    def _on_stream_disconnected(self):
        """Handle stream disconnection event"""
        self.logger.warning("RTSP stream disconnected - attempting reconnection")
    
    def _on_stream_reconnected(self):
        """Handle stream reconnection event"""
        self.logger.info("RTSP stream reconnected successfully")
    
    def create_pipeline(self):
        """Create GStreamer pipeline from DeepStream config"""
        try:
            # Create pipeline
            self.pipeline = Gst.Pipeline()
            
            if not self.pipeline:
                self.logger.error("Failed to create GStreamer pipeline")
                return False
            
            # Set up bus for message handling
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect("message", self._bus_call)
            
            self.logger.info(f"Pipeline created successfully with config: {self.config_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create pipeline: {e}")
            return False
    
    def _bus_call(self, bus, message):
        """Handle GStreamer bus messages"""
        t = message.type
        
        if t == Gst.MessageType.EOS:
            self.logger.info("End-of-stream received")
            if self.loop:
                self.loop.quit()
                
        elif t == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            self.logger.warning(f"Warning: {err}: {debug}")
            
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            self.logger.error(f"Error: {err}: {debug}")
            if self.loop:
                self.loop.quit()
                
        elif t == Gst.MessageType.ELEMENT:
            # Handle DeepStream-specific messages
            struct = message.get_structure()
            if struct:
                name = struct.get_name()
                if name == "stream-eos":
                    self.logger.info("Stream EOS received")
                elif name == "stream-removed":
                    self.logger.info("Stream removed")
        
        return True
    
    def start_pipeline(self):
        """Start the DeepStream pipeline"""
        if not self.pipeline:
            self.logger.error("Pipeline not created. Call create_pipeline() first.")
            return False
        
        try:
            # Set pipeline to PLAYING state
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            
            if ret == Gst.StateChangeReturn.FAILURE:
                self.logger.error("Failed to set pipeline to PLAYING state")
                return False
            
            self.logger.info("Pipeline started successfully")
            
            # Start RTSP stream monitoring if configured
            if self.stream_manager:
                self.stream_manager.start_monitoring()
            
            # Create main loop
            self.loop = GObject.MainLoop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start pipeline: {e}")
            return False
    
    def stop_pipeline(self):
        """Stop the DeepStream pipeline"""
        if self.pipeline:
            try:
                # Set pipeline to NULL state
                self.pipeline.set_state(Gst.State.NULL)
                self.logger.info("Pipeline stopped")
                
                # Stop RTSP stream monitoring if configured
                if self.stream_manager:
                    self.stream_manager.stop_monitoring_thread()
                
                if self.loop and self.loop.is_running():
                    self.loop.quit()
                    
            except Exception as e:
                self.logger.error(f"Error stopping pipeline: {e}")
    
    def run(self):
        """Run the pipeline (blocking call)"""
        if not self.create_pipeline():
            return False
        
        if not self.start_pipeline():
            return False
        
        try:
            # Run main loop
            self.logger.info("Running pipeline...")
            self.loop.run()
            
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
        finally:
            self.stop_pipeline()
        
        return True
    
    def get_pipeline_state(self):
        """Get current pipeline state"""
        if self.pipeline:
            ret, state, pending = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)
            return state
        return None
    
    def get_model_info(self, model_id: str = "primary_detector"):
        """Get information about registered models"""
        return self.model_manager.get_model_info(model_id)
    
    def list_models(self):
        """Get list of all registered models"""
        return self.model_manager.list_models()
    
    def get_tracking_statistics(self):
        """Get tracking statistics"""
        if self.tracking_manager:
            return self.tracking_manager.get_tracking_statistics()
        return None
    
    def get_active_tracks(self):
        """Get all currently active tracks"""
        if self.tracking_manager:
            return self.tracking_manager.get_active_tracks()
        return {}
    
    def get_track_info(self, tracking_id: int):
        """Get information about a specific track"""
        if self.tracking_manager:
            return self.tracking_manager.get_track_info(tracking_id)
        return None
    
    def get_behavior_statistics(self):
        """Get behavior analysis statistics"""
        if self.behavior_analyzer:
            return self.behavior_analyzer.get_statistics()
        return None
    
    def get_recent_events(self, limit: int = 100):
        """Get recent behavior events"""
        if self.behavior_analyzer:
            return self.behavior_analyzer.get_recent_events(limit)
        return []
    
    def add_behavior_event_callback(self, callback):
        """Add callback for behavior events"""
        if self.behavior_analyzer:
            self.behavior_analyzer.add_event_callback(callback)
    
    def get_event_dispatcher_statistics(self):
        """Get event dispatcher statistics"""
        if self.event_dispatcher:
            return self.event_dispatcher.get_statistics()
        return None
    
    def get_video_output_statistics(self):
        """Get video output statistics"""
        if self.video_output_manager:
            return self.video_output_manager.get_statistics()
        return None
    
    def update_video_config(self, new_config: dict):
        """Update video output configuration at runtime"""
        if self.video_output_manager:
            self.video_output_manager.update_config(new_config)
    
    def get_kinesis_statistics(self):
        """Get Kinesis Video Streams statistics"""
        if self.kinesis_manager:
            return self.kinesis_manager.get_statistics()
        return None
    
    def is_kinesis_healthy(self):
        """Check if Kinesis Video Stream is healthy"""
        if self.kinesis_manager:
            return self.kinesis_manager.is_healthy()
        return False
    
    def get_frame_saver_statistics(self):
        """Get frame saver statistics"""
        if self.frame_saver:
            return self.frame_saver.get_statistics()
        return None
    
    def update_frame_saver_config(self, new_config: dict):
        """Update frame saver configuration at runtime"""
        if self.frame_saver:
            self.frame_saver.update_config(new_config)
    
    def _on_config_changed(self, new_config):
        """Handle configuration changes (hot-reload)"""
        self.logger.info("Configuration changed - updating components...")
        
        try:
            # Update behavior analyzer configuration
            if self.behavior_analyzer:
                behavior_config = self.config_manager.get_behavior_config()
                self.behavior_analyzer.update_config(behavior_config)
                self.logger.info("Behavior analyzer configuration updated")
            
            # Update event dispatcher configuration
            if self.event_dispatcher:
                events_config = self.config_manager.get_events_config()
                self.event_dispatcher.update_config(events_config)
                self.logger.info("Event dispatcher configuration updated")
            
            # Update frame saver configuration
            if self.frame_saver:
                frame_config = self.config_manager.get_frame_saving_config()
                self.frame_saver.update_config(frame_config)
                self.logger.info("Frame saver configuration updated")
            
            self.logger.info("Configuration hot-reload completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error during configuration hot-reload: {e}")
    
    def get_current_config(self):
        """Get current application configuration"""
        return self.config_manager.get_full_config()
    
    def reload_config(self):
        """Manually reload configuration from file"""
        return self.config_manager._load_config()
    
    def get_performance_metrics(self):
        """Get current performance metrics"""
        if self.performance_manager:
            return self.performance_manager.get_current_metrics()
        return None
    
    def get_performance_analysis(self):
        """Get performance analysis and recommendations"""
        if self.performance_manager:
            return self.performance_manager.get_performance_analysis()
        return None
    
    def get_system_info(self):
        """Get system information and resources"""
        if self.performance_manager:
            return self.performance_manager.get_system_info()
        return None
    
    def update_pipeline_metrics(self, frames_processed: int, objects_detected: int, 
                              inference_time_ms: float, tracking_time_ms: float):
        """Update pipeline-specific performance metrics"""
        if self.performance_manager:
            self.performance_manager.update_pipeline_metrics(
                frames_processed, objects_detected, inference_time_ms, tracking_time_ms
            )
    
    def shutdown(self):
        """Shutdown all components gracefully"""
        try:
            self.logger.info("Shutting down pipeline manager...")
            
            # Stop pipeline
            self.stop_pipeline()
            
            # Shutdown event dispatcher
            if self.event_dispatcher:
                self.event_dispatcher.shutdown()
            
            # Shutdown configuration manager
            if self.config_manager:
                self.config_manager.shutdown()
            
            self.logger.info("Pipeline manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='RTSP Object Detection System')
    parser.add_argument('--config', default='configs/deepstream_app_config.txt',
                       help='DeepStream configuration file path')
    parser.add_argument('--app-config', default='configs/app_config.yaml',
                       help='Application configuration file path')
    parser.add_argument('--rtsp-uri', help='RTSP stream URI')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--log-file', default='logs/deepstream_app.log',
                       help='Log file path')
    parser.add_argument('--json-logs', action='store_true',
                       help='Enable JSON formatted logging')
    
    args = parser.parse_args()
    
    try:
        # Set up comprehensive logging
        from logging_config import setup_logging
        health_mon, error_recovery, deepstream_handler = setup_logging(
            log_level=args.log_level,
            log_file=args.log_file,
            enable_json=args.json_logs
        )
        
        logger = structlog.get_logger(__name__)
        logger.info(
            "Starting RTSP Object Detection System",
            config_file=args.config,
            app_config=args.app_config,
            rtsp_uri=args.rtsp_uri,
            log_level=args.log_level
        )
        
        # Create and run pipeline manager
        manager = DeepStreamPipelineManager(
            args.config, 
            args.app_config, 
            args.rtsp_uri
        )
        
        # Update health monitor
        if health_mon:
            health_mon.update_component_health("pipeline_manager", "healthy")
        
        success = manager.run()
        
        if not success:
            logger.error("Pipeline execution failed")
            if health_mon:
                health_mon.update_component_health("pipeline_manager", "unhealthy", 
                                                 {"reason": "Pipeline execution failed"})
            sys.exit(1)
        else:
            logger.info("Pipeline execution completed successfully")
            if health_mon:
                health_mon.update_component_health("pipeline_manager", "healthy")
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    except FileNotFoundError as e:
        logger.error("Configuration file not found", error=str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Application error", error=str(e), exc_info=True)
        if 'error_recovery' in locals() and error_recovery:
            error_recovery.handle_error("application_error", {"error": str(e)}, "main")
        sys.exit(1)
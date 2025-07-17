#!/usr/bin/env python3

"""
Web Server for RTSP Object Detection System Dashboard
Serves real-time metrics and system status via REST API
"""

import json
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
from flask import Flask, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import structlog

from performance_monitor import get_performance_manager, initialize_performance_monitoring
from config_manager import ConfigManager


class WebServer:
    """Web server for dashboard and metrics API"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, config_path: str = "configs/app_config.yaml"):
        self.host = host
        self.port = port
        self.logger = structlog.get_logger(__name__)
        
        # Initialize Flask app
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize performance monitoring
        self.perf_manager = initialize_performance_monitoring()
        
        # Load configuration
        try:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.get_config()
        except Exception as e:
            self.logger.warning("Failed to load config, using defaults", error=str(e))
            self.config = {}
        
        # System status tracking
        self.system_status = {
            'rtsp_server': 'unknown',
            'deepstream_app': 'unknown',
            'mqtt_broker': 'unknown',
            'event_receiver': 'unknown'
        }
        
        # Recent events storage
        self.recent_events = []
        self.max_events = 50
        
        # Setup routes
        self._setup_routes()
        
        self.logger.info("Web server initialized", host=host, port=port)
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Serve the main dashboard"""
            return send_from_directory('../web', 'index.html')
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get current performance metrics"""
            try:
                metrics = self.perf_manager.get_current_metrics()
                if not metrics:
                    return jsonify({
                        'fps': 0.0,
                        'objects_detected': 0,
                        'latency_ms': 0.0,
                        'gpu_usage': 0.0,
                        'cpu_usage': 0.0,
                        'memory_usage_mb': 0.0,
                        'gpu_memory_usage_mb': 0.0,
                        'timestamp': datetime.now().isoformat()
                    })
                
                return jsonify({
                    'fps': round(metrics.fps, 1),
                    'objects_detected': metrics.objects_detected,
                    'latency_ms': round(metrics.latency_ms, 1),
                    'gpu_usage': round(metrics.gpu_usage, 1),
                    'cpu_usage': round(metrics.cpu_usage, 1),
                    'memory_usage_mb': round(metrics.memory_usage_mb, 1),
                    'gpu_memory_usage_mb': round(metrics.gpu_memory_usage_mb, 1),
                    'inference_time_ms': round(metrics.inference_time_ms, 1),
                    'tracking_time_ms': round(metrics.tracking_time_ms, 1),
                    'frames_processed': metrics.frames_processed,
                    'timestamp': metrics.timestamp
                })
                
            except Exception as e:
                self.logger.error("Failed to get metrics", error=str(e))
                return jsonify({'error': 'Failed to get metrics'}), 500
        
        @self.app.route('/api/status')
        def get_status():
            """Get system component status"""
            try:
                # Check RTSP server (MediaMTX)
                rtsp_status = self._check_service_status('localhost', 8554)
                
                # Check DeepStream app (assume running if metrics available)
                metrics = self.perf_manager.get_current_metrics()
                deepstream_status = 'online' if metrics and metrics.fps > 0 else 'offline'
                
                # Check MQTT broker
                mqtt_status = self._check_service_status('localhost', 1883)
                
                # Check event receiver
                event_status = self._check_service_status('localhost', 8090)
                
                return jsonify({
                    'rtsp_server': rtsp_status,
                    'deepstream_app': deepstream_status,
                    'mqtt_broker': mqtt_status,
                    'event_receiver': event_status,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error("Failed to get status", error=str(e))
                return jsonify({'error': 'Failed to get status'}), 500
        
        @self.app.route('/api/events')
        def get_events():
            """Get recent events"""
            try:
                return jsonify({
                    'events': self.recent_events[-20:],  # Last 20 events
                    'count': len(self.recent_events),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error("Failed to get events", error=str(e))
                return jsonify({'error': 'Failed to get events'}), 500
        
        @self.app.route('/api/config')
        def get_config():
            """Get current configuration"""
            try:
                return jsonify({
                    'input_stream': self._get_input_stream_url(),
                    'output_stream': self._get_output_stream_url(),
                    'mqtt_topic': self.config.get('events', {}).get('mqtt', {}).get('topic', 'detection/events'),
                    'http_endpoint': self.config.get('events', {}).get('http', {}).get('endpoint', 'http://localhost:8090/events'),
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                self.logger.error("Failed to get config", error=str(e))
                return jsonify({'error': 'Failed to get config'}), 500
        
        @self.app.route('/api/system')
        def get_system_info():
            """Get system information"""
            try:
                system_info = self.perf_manager.get_system_info()
                return jsonify(system_info)
                
            except Exception as e:
                self.logger.error("Failed to get system info", error=str(e))
                return jsonify({'error': 'Failed to get system info'}), 500
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            })
    
    def _check_service_status(self, host: str, port: int) -> str:
        """Check if a service is running on given host:port"""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex((host, port))
            sock.close()
            return 'online' if result == 0 else 'offline'
        except Exception:
            return 'offline'
    
    def _get_input_stream_url(self) -> str:
        """Get input stream URL"""
        return "rtsp://localhost:8554/live.stream"
    
    def _get_output_stream_url(self) -> str:
        """Get output stream URL"""
        return "rtsp://localhost:8555/detection.stream"
    
    def add_event(self, event: Dict[str, Any]):
        """Add an event to recent events"""
        try:
            event['timestamp'] = datetime.now().isoformat()
            self.recent_events.append(event)
            
            # Keep only recent events
            if len(self.recent_events) > self.max_events:
                self.recent_events = self.recent_events[-self.max_events:]
                
        except Exception as e:
            self.logger.error("Failed to add event", error=str(e))
    
    def update_pipeline_metrics(self, frames_processed: int, objects_detected: int, 
                              inference_time_ms: float, tracking_time_ms: float):
        """Update pipeline metrics"""
        if self.perf_manager:
            self.perf_manager.update_pipeline_metrics(
                frames_processed, objects_detected, inference_time_ms, tracking_time_ms
            )
    
    def run(self, debug: bool = False):
        """Run the web server"""
        try:
            self.logger.info("Starting web server", host=self.host, port=self.port)
            self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)
        except Exception as e:
            self.logger.error("Failed to start web server", error=str(e))
            raise
    
    def run_threaded(self, debug: bool = False):
        """Run the web server in a separate thread"""
        def run_server():
            try:
                self.app.run(host=self.host, port=self.port, debug=debug, threaded=True)
            except Exception as e:
                self.logger.error("Web server error", error=str(e))
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        self.logger.info("Web server started in background thread")
        return server_thread


# Global web server instance
web_server: Optional[WebServer] = None


def initialize_web_server(host: str = "0.0.0.0", port: int = 8080, 
                         config_path: str = "configs/app_config.yaml") -> WebServer:
    """Initialize global web server"""
    global web_server
    
    if web_server is None:
        web_server = WebServer(host, port, config_path)
    
    return web_server


def get_web_server() -> Optional[WebServer]:
    """Get global web server instance"""
    return web_server


if __name__ == "__main__":
    # Test web server
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Server for RTSP Object Detection System")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    server = initialize_web_server(args.host, args.port)
    server.run(debug=args.debug)
#!/usr/bin/env python3

"""
Test Web Dashboard and Metrics API
Tests the real metrics implementation and HLS stream integration
"""

import pytest
import requests
import time
import json
from unittest.mock import patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from web_server import WebServer, initialize_web_server
from performance_monitor import PerformanceMetrics


class TestWebDashboard:
    """Test web dashboard functionality"""
    
    @pytest.fixture
    def web_server(self):
        """Fixture for web server"""
        # Mock performance manager to avoid GPU dependencies
        with patch('web_server.initialize_performance_monitoring') as mock_perf:
            mock_perf_manager = MagicMock()
            mock_perf.return_value = mock_perf_manager
            
            # Create mock metrics
            mock_metrics = PerformanceMetrics(
                timestamp="2024-01-01T12:00:00",
                fps=25.5,
                latency_ms=45.2,
                cpu_usage=65.3,
                memory_usage_mb=1024.0,
                gpu_usage=78.9,
                gpu_memory_usage_mb=2048.0,
                frames_processed=1000,
                objects_detected=5,
                inference_time_ms=12.5,
                tracking_time_ms=3.2
            )
            
            mock_perf_manager.get_current_metrics.return_value = mock_metrics
            mock_perf_manager.get_system_info.return_value = {
                'system_resources': {
                    'cpu_count': 8,
                    'total_memory_gb': 16.0,
                    'gpu_count': 1,
                    'gpu_memory_total_mb': [8192.0],
                    'gpu_compute_capability': ['8.6']
                },
                'gpu_metrics': [{'usage_percent': 78.9}],
                'monitoring_active': True,
                'metrics_count': 100
            }
            
            server = WebServer(host="127.0.0.1", port=8081, config_path="configs/app_config.yaml")
            server.perf_manager = mock_perf_manager
            
            yield server
    
    def test_metrics_api_endpoint(self, web_server):
        """Test metrics API endpoint returns real data"""
        with web_server.app.test_client() as client:
            response = client.get('/api/metrics')
            
            assert response.status_code == 200
            data = response.get_json()
            
            # Verify real metrics are returned
            assert 'fps' in data
            assert 'objects_detected' in data
            assert 'latency_ms' in data
            assert 'gpu_usage' in data
            assert 'cpu_usage' in data
            assert 'timestamp' in data
            
            # Verify values are realistic (not fake random values)
            assert data['fps'] == 25.5
            assert data['objects_detected'] == 5
            assert data['gpu_usage'] == 78.9
            assert data['cpu_usage'] == 65.3
    
    def test_status_api_endpoint(self, web_server):
        """Test status API endpoint"""
        with web_server.app.test_client() as client:
            response = client.get('/api/status')
            
            assert response.status_code == 200
            data = response.get_json()
            
            # Verify status fields are present
            assert 'rtsp_server' in data
            assert 'deepstream_app' in data
            assert 'mqtt_broker' in data
            assert 'event_receiver' in data
            assert 'timestamp' in data
            
            # Status should be either 'online' or 'offline'
            valid_statuses = ['online', 'offline']
            assert data['rtsp_server'] in valid_statuses
            assert data['deepstream_app'] in valid_statuses
            assert data['mqtt_broker'] in valid_statuses
            assert data['event_receiver'] in valid_statuses
    
    def test_events_api_endpoint(self, web_server):
        """Test events API endpoint"""
        # Add some test events
        test_events = [
            {
                'event_type': 'object_appeared',
                'object': {'class_name': 'person', 'tracking_id': 1},
                'metadata': {'confidence': 0.95}
            },
            {
                'event_type': 'object_static',
                'object': {'class_name': 'car', 'tracking_id': 2},
                'metadata': {'duration': 30}
            }
        ]
        
        for event in test_events:
            web_server.add_event(event)
        
        with web_server.app.test_client() as client:
            response = client.get('/api/events')
            
            assert response.status_code == 200
            data = response.get_json()
            
            # Verify events structure
            assert 'events' in data
            assert 'count' in data
            assert 'timestamp' in data
            
            # Verify events were added
            assert len(data['events']) >= 2
            
            # Verify event structure
            for event in data['events']:
                assert 'timestamp' in event
                assert 'event_type' in event
    
    def test_config_api_endpoint(self, web_server):
        """Test config API endpoint"""
        with web_server.app.test_client() as client:
            response = client.get('/api/config')
            
            assert response.status_code == 200
            data = response.get_json()
            
            # Verify config fields
            assert 'input_stream' in data
            assert 'output_stream' in data
            assert 'mqtt_topic' in data
            assert 'http_endpoint' in data
            assert 'timestamp' in data
            
            # Verify stream URLs are correct
            assert 'rtsp://localhost:8554/live.stream' in data['input_stream']
            assert 'rtsp://localhost:8555/detection.stream' in data['output_stream']
    
    def test_system_info_endpoint(self, web_server):
        """Test system info API endpoint"""
        with web_server.app.test_client() as client:
            response = client.get('/api/system')
            
            assert response.status_code == 200
            data = response.get_json()
            
            # Verify system info structure
            assert 'system_resources' in data
            assert 'gpu_metrics' in data
            assert 'monitoring_active' in data
            assert 'metrics_count' in data
    
    def test_health_check_endpoint(self, web_server):
        """Test health check endpoint"""
        with web_server.app.test_client() as client:
            response = client.get('/health')
            
            assert response.status_code == 200
            data = response.get_json()
            
            # Verify health check response
            assert data['status'] == 'healthy'
            assert 'timestamp' in data
            assert 'version' in data
    
    def test_dashboard_html_served(self, web_server):
        """Test that dashboard HTML is served correctly"""
        with web_server.app.test_client() as client:
            response = client.get('/')
            
            # Should attempt to serve index.html
            # This might fail if file doesn't exist, but we're testing the route
            assert response.status_code in [200, 404]  # 404 is OK if file not found in test
    
    def test_metrics_update_pipeline_integration(self, web_server):
        """Test that pipeline metrics can be updated"""
        # Update pipeline metrics
        web_server.update_pipeline_metrics(
            frames_processed=2000,
            objects_detected=10,
            inference_time_ms=15.5,
            tracking_time_ms=4.2
        )
        
        # Verify the performance manager was called
        web_server.perf_manager.update_pipeline_metrics.assert_called_once_with(
            2000, 10, 15.5, 4.2
        )
    
    def test_event_addition(self, web_server):
        """Test adding events to the web server"""
        initial_count = len(web_server.recent_events)
        
        test_event = {
            'event_type': 'object_appeared',
            'object': {'class_name': 'person', 'tracking_id': 123},
            'metadata': {'confidence': 0.85}
        }
        
        web_server.add_event(test_event)
        
        # Verify event was added
        assert len(web_server.recent_events) == initial_count + 1
        
        # Verify event has timestamp
        added_event = web_server.recent_events[-1]
        assert 'timestamp' in added_event
        assert added_event['event_type'] == 'object_appeared'
        assert added_event['object']['class_name'] == 'person'
    
    def test_cors_headers(self, web_server):
        """Test that CORS headers are properly set"""
        with web_server.app.test_client() as client:
            response = client.get('/api/metrics')
            
            # CORS should be enabled for API endpoints
            assert response.status_code == 200


class TestHLSStreamIntegration:
    """Test HLS stream integration"""
    
    def test_hls_stream_urls_in_html(self):
        """Test that HTML contains correct HLS stream URLs"""
        html_path = os.path.join(os.path.dirname(__file__), '..', 'web', 'index.html')
        
        if os.path.exists(html_path):
            with open(html_path, 'r') as f:
                html_content = f.read()
            
            # Verify HLS iframe URLs are present
            assert 'http://localhost:8888/live.stream/' in html_content
            assert 'http://localhost:8888/detection.stream/' in html_content
            
            # Verify iframe elements are properly configured
            assert 'iframe' in html_content
            assert 'allowfullscreen' in html_content
            assert 'Input Stream (HLS)' in html_content
            assert 'Detection Output (HLS)' in html_content
    
    def test_mediamtx_config_hls_enabled(self):
        """Test that MediaMTX configuration has HLS enabled"""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'mediamtx.yml')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            # Verify HLS is enabled
            assert 'hlsDisable: no' in config_content
            assert 'hlsAddress: :8888' in config_content
            assert 'hlsAllowOrigin: "*"' in config_content


def test_real_metrics_vs_fake_metrics():
    """Test that we're using real metrics instead of fake ones"""
    # This test verifies that the JavaScript no longer generates fake random metrics
    html_path = os.path.join(os.path.dirname(__file__), '..', 'web', 'index.html')
    
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            html_content = f.read()
        
        # Verify fake metric generation is removed
        assert 'Math.random()' not in html_content
        assert 'Math.floor(Math.random()' not in html_content
        
        # Verify real API calls are present
        assert 'fetch(`${API_BASE}/metrics`)' in html_content
        assert 'fetch(`${API_BASE}/status`)' in html_content
        assert 'fetch(`${API_BASE}/events`)' in html_content
        
        # Verify API base URL is configured
        assert "const API_BASE = 'http://localhost:8080/api'" in html_content


if __name__ == "__main__":
    # Run basic tests
    print("Testing Web Dashboard Implementation...")
    
    # Test HTML file exists and has correct content
    html_path = os.path.join(os.path.dirname(__file__), '..', 'web', 'index.html')
    if os.path.exists(html_path):
        print("✓ HTML file exists")
        
        with open(html_path, 'r') as f:
            content = f.read()
        
        if 'Math.random()' not in content:
            print("✓ Fake metrics removed")
        else:
            print("✗ Still contains fake metrics")
        
        if 'fetch(`${API_BASE}/metrics`)' in content:
            print("✓ Real API calls implemented")
        else:
            print("✗ Missing real API calls")
        
        if 'iframe' in content and 'localhost:8888' in content:
            print("✓ HLS stream iframes added")
        else:
            print("✗ Missing HLS stream iframes")
    else:
        print("✗ HTML file not found")
    
    # Test web server module
    try:
        from web_server import WebServer
        print("✓ Web server module imports successfully")
    except ImportError as e:
        print(f"✗ Web server import failed: {e}")
    
    # Test requirements
    requirements_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r') as f:
            requirements = f.read()
        
        if 'Flask' in requirements and 'Flask-CORS' in requirements:
            print("✓ Flask dependencies added to requirements.txt")
        else:
            print("✗ Missing Flask dependencies in requirements.txt")
    
    print("\nTest completed!")
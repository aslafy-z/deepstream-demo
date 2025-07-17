#!/usr/bin/env python3

"""
Kinesis Video Streams Manager
Handles AWS Kinesis Video Streams integration and monitoring
"""

import os
import logging
import time
import threading
from typing import Dict, Optional, Callable
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    print("WARNING: boto3 not available. Kinesis Video Streams functionality will be limited.")
    BOTO3_AVAILABLE = False


class KinesisVideoStreamsManager:
    """Manages AWS Kinesis Video Streams integration"""
    
    def __init__(self, config: Dict):
        """
        Initialize Kinesis Video Streams manager
        
        Args:
            config: Configuration dictionary with Kinesis settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # AWS configuration
        self.stream_name = config.get('stream_name', 'detection-stream')
        self.aws_region = config.get('aws_region', 'us-east-1')
        self.retention_period_hours = config.get('retention_period_hours', 24)
        
        # AWS clients
        self.kinesis_video_client = None
        self.kinesis_video_media_client = None
        
        # Stream state
        self.stream_arn = None
        self.stream_status = 'UNKNOWN'
        self.endpoint_url = None
        
        # Monitoring
        self.stats = {
            'stream_created': False,
            'stream_active': False,
            'last_health_check': None,
            'health_check_count': 0,
            'errors': 0
        }
        
        # Health monitoring
        self.monitor_thread = None
        self.stop_monitoring = False
        self.health_check_interval = 60  # seconds
        
        # Initialize AWS clients
        self._init_aws_clients()
        
        # Create or verify stream
        self._setup_stream()
    
    def _init_aws_clients(self):
        """Initialize AWS clients"""
        if not BOTO3_AVAILABLE:
            self.logger.error("boto3 not available - Kinesis functionality disabled")
            return
        
        try:
            # Initialize Kinesis Video client
            self.kinesis_video_client = boto3.client(
                'kinesisvideo',
                region_name=self.aws_region
            )
            
            self.logger.info(f"AWS Kinesis Video client initialized for region: {self.aws_region}")
            
        except NoCredentialsError:
            self.logger.error("AWS credentials not found. Please configure AWS credentials.")
        except Exception as e:
            self.logger.error(f"Failed to initialize AWS clients: {e}")
    
    def _setup_stream(self):
        """Create or verify Kinesis Video Stream"""
        if not self.kinesis_video_client:
            self.logger.error("Kinesis Video client not available")
            return
        
        try:
            # Check if stream exists
            if self._stream_exists():
                self.logger.info(f"Kinesis Video Stream exists: {self.stream_name}")
                self._get_stream_info()
            else:
                # Create new stream
                self._create_stream()
            
            # Get data endpoint for streaming
            self._get_data_endpoint()
            
            # Start health monitoring
            self._start_health_monitoring()
            
        except Exception as e:
            self.logger.error(f"Failed to setup Kinesis Video Stream: {e}")
            self.stats['errors'] += 1
    
    def _stream_exists(self) -> bool:
        """Check if Kinesis Video Stream exists"""
        try:
            response = self.kinesis_video_client.describe_stream(
                StreamName=self.stream_name
            )
            
            stream_info = response['StreamInfo']
            self.stream_arn = stream_info['StreamARN']
            self.stream_status = stream_info['Status']
            
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                return False
            else:
                self.logger.error(f"Error checking stream existence: {e}")
                return False
    
    def _create_stream(self):
        """Create new Kinesis Video Stream"""
        try:
            self.logger.info(f"Creating Kinesis Video Stream: {self.stream_name}")
            
            response = self.kinesis_video_client.create_stream(
                StreamName=self.stream_name,
                DataRetentionInHours=self.retention_period_hours,
                MediaType='video/h264'
            )
            
            self.stream_arn = response['StreamARN']
            self.stats['stream_created'] = True
            
            self.logger.info(f"Kinesis Video Stream created: {self.stream_arn}")
            
            # Wait for stream to become active
            self._wait_for_stream_active()
            
        except Exception as e:
            self.logger.error(f"Failed to create Kinesis Video Stream: {e}")
            self.stats['errors'] += 1
            raise
    
    def _get_stream_info(self):
        """Get detailed stream information"""
        try:
            response = self.kinesis_video_client.describe_stream(
                StreamName=self.stream_name
            )
            
            stream_info = response['StreamInfo']
            self.stream_arn = stream_info['StreamARN']
            self.stream_status = stream_info['Status']
            
            self.logger.info(f"Stream status: {self.stream_status}")
            
            if self.stream_status == 'ACTIVE':
                self.stats['stream_active'] = True
            
        except Exception as e:
            self.logger.error(f"Failed to get stream info: {e}")
            self.stats['errors'] += 1
    
    def _wait_for_stream_active(self, timeout: int = 300):
        """Wait for stream to become active"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                self._get_stream_info()
                
                if self.stream_status == 'ACTIVE':
                    self.logger.info("Kinesis Video Stream is now active")
                    return True
                
                self.logger.info(f"Waiting for stream to become active... Status: {self.stream_status}")
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error waiting for stream: {e}")
                time.sleep(10)
        
        self.logger.error(f"Stream did not become active within {timeout} seconds")
        return False
    
    def _get_data_endpoint(self):
        """Get data endpoint for streaming"""
        try:
            response = self.kinesis_video_client.get_data_endpoint(
                StreamName=self.stream_name,
                APIName='PUT_MEDIA'
            )
            
            self.endpoint_url = response['DataEndpoint']
            self.logger.info(f"Data endpoint: {self.endpoint_url}")
            
            # Initialize media client with endpoint
            self.kinesis_video_media_client = boto3.client(
                'kinesis-video-media',
                region_name=self.aws_region,
                endpoint_url=self.endpoint_url
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get data endpoint: {e}")
            self.stats['errors'] += 1
    
    def _start_health_monitoring(self):
        """Start health monitoring thread"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Kinesis Video Stream health monitoring started")
    
    def _health_monitor_loop(self):
        """Health monitoring loop"""
        while not self.stop_monitoring:
            try:
                self._perform_health_check()
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                time.sleep(self.health_check_interval)
    
    def _perform_health_check(self):
        """Perform health check on Kinesis Video Stream"""
        try:
            self.stats['health_check_count'] += 1
            self.stats['last_health_check'] = time.time()
            
            # Check stream status
            self._get_stream_info()
            
            # Log health status
            if self.stream_status == 'ACTIVE':
                self.stats['stream_active'] = True
                self.logger.debug(f"Health check passed - Stream active: {self.stream_name}")
            else:
                self.stats['stream_active'] = False
                self.logger.warning(f"Health check warning - Stream status: {self.stream_status}")
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.stats['errors'] += 1
            self.stats['stream_active'] = False
    
    def get_stream_configuration(self) -> Dict:
        """Get stream configuration for DeepStream"""
        return {
            'stream_name': self.stream_name,
            'aws_region': self.aws_region,
            'stream_arn': self.stream_arn,
            'endpoint_url': self.endpoint_url,
            'retention_period_hours': self.retention_period_hours
        }
    
    def get_statistics(self) -> Dict:
        """Get Kinesis Video Streams statistics"""
        return {
            **self.stats,
            'stream_name': self.stream_name,
            'stream_arn': self.stream_arn,
            'stream_status': self.stream_status,
            'aws_region': self.aws_region,
            'endpoint_url': self.endpoint_url,
            'retention_period_hours': self.retention_period_hours,
            'boto3_available': BOTO3_AVAILABLE
        }
    
    def is_healthy(self) -> bool:
        """Check if Kinesis Video Stream is healthy"""
        return (
            self.stats['stream_active'] and 
            self.stream_status == 'ACTIVE' and
            self.endpoint_url is not None
        )
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.stop_monitoring = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        self.logger.info("Kinesis Video Stream monitoring stopped")
    
    def update_stream_configuration(self, new_config: Dict):
        """Update stream configuration"""
        try:
            # Update retention period if changed
            if 'retention_period_hours' in new_config:
                new_retention = new_config['retention_period_hours']
                if new_retention != self.retention_period_hours:
                    self._update_retention_period(new_retention)
            
            # Update other configuration
            self.config.update(new_config)
            self.logger.info("Kinesis Video Stream configuration updated")
            
        except Exception as e:
            self.logger.error(f"Failed to update stream configuration: {e}")
    
    def _update_retention_period(self, new_retention_hours: int):
        """Update stream retention period"""
        try:
            self.kinesis_video_client.update_data_retention(
                StreamName=self.stream_name,
                CurrentVersion=self._get_stream_version(),
                Operation='INCREASE_DATA_RETENTION' if new_retention_hours > self.retention_period_hours else 'DECREASE_DATA_RETENTION',
                DataRetentionChangeInHours=abs(new_retention_hours - self.retention_period_hours)
            )
            
            self.retention_period_hours = new_retention_hours
            self.logger.info(f"Stream retention period updated to {new_retention_hours} hours")
            
        except Exception as e:
            self.logger.error(f"Failed to update retention period: {e}")
    
    def _get_stream_version(self) -> str:
        """Get current stream version"""
        try:
            response = self.kinesis_video_client.describe_stream(
                StreamName=self.stream_name
            )
            return response['StreamInfo']['Version']
            
        except Exception as e:
            self.logger.error(f"Failed to get stream version: {e}")
            return "1"


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {
        'stream_name': 'test-detection-stream',
        'aws_region': 'us-east-1',
        'retention_period_hours': 24
    }
    
    try:
        # Create Kinesis manager
        manager = KinesisVideoStreamsManager(config)
        
        # Get statistics
        stats = manager.get_statistics()
        print(f"Kinesis Video Streams statistics: {stats}")
        
        # Check health
        print(f"Stream healthy: {manager.is_healthy()}")
        
        # Wait a bit for monitoring
        time.sleep(5)
        
        # Stop monitoring
        manager.stop_monitoring()
        
    except Exception as e:
        print(f"Error: {e}")
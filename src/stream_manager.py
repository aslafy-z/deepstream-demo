#!/usr/bin/env python3

"""
RTSP Stream Manager
Handles RTSP input stream connection, monitoring, and reconnection logic
"""

import time
import threading
import logging
from typing import Optional, Callable
from urllib.parse import urlparse

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

try:
    import pyds
except ImportError:
    print("ERROR: DeepStream Python bindings (pyds) not found.")


class RTSPStreamManager:
    """Manages RTSP stream connection and health monitoring"""
    
    def __init__(self, rtsp_uri: str, reconnect_interval: int = 5, max_reconnect_attempts: int = -1):
        """
        Initialize RTSP stream manager
        
        Args:
            rtsp_uri: RTSP stream URL
            reconnect_interval: Seconds between reconnection attempts
            max_reconnect_attempts: Maximum reconnection attempts (-1 for infinite)
        """
        self.rtsp_uri = rtsp_uri
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.logger = logging.getLogger(__name__)
        
        # Stream state
        self.is_connected = False
        self.reconnect_count = 0
        self.last_frame_time = None
        
        # Threading
        self.monitor_thread = None
        self.stop_monitoring = False
        
        # Callbacks
        self.on_connected_callback: Optional[Callable] = None
        self.on_disconnected_callback: Optional[Callable] = None
        self.on_reconnected_callback: Optional[Callable] = None
        
        # Validate RTSP URI
        self._validate_rtsp_uri()
    
    def _validate_rtsp_uri(self):
        """Validate RTSP URI format"""
        try:
            parsed = urlparse(self.rtsp_uri)
            if parsed.scheme not in ['rtsp', 'rtsps']:
                raise ValueError(f"Invalid RTSP scheme: {parsed.scheme}")
            if not parsed.hostname:
                raise ValueError("RTSP URI must include hostname")
            self.logger.info(f"RTSP URI validated: {self.rtsp_uri}")
        except Exception as e:
            self.logger.error(f"Invalid RTSP URI: {e}")
            raise
    
    def set_callbacks(self, on_connected=None, on_disconnected=None, on_reconnected=None):
        """Set callback functions for stream events"""
        self.on_connected_callback = on_connected
        self.on_disconnected_callback = on_disconnected
        self.on_reconnected_callback = on_reconnected
    
    def start_monitoring(self):
        """Start stream health monitoring in background thread"""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.logger.warning("Stream monitoring already running")
            return
        
        self.stop_monitoring = False
        self.monitor_thread = threading.Thread(target=self._monitor_stream, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Stream monitoring started")
    
    def stop_monitoring_thread(self):
        """Stop stream health monitoring"""
        self.stop_monitoring = True
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        self.logger.info("Stream monitoring stopped")
    
    def _monitor_stream(self):
        """Monitor stream health and handle reconnections"""
        while not self.stop_monitoring:
            try:
                # Check if we need to attempt connection/reconnection
                if not self.is_connected:
                    if self._should_attempt_reconnect():
                        self._attempt_connection()
                
                # Check stream health if connected
                elif self.is_connected:
                    if not self._check_stream_health():
                        self._handle_disconnection()
                
                time.sleep(self.reconnect_interval)
                
            except Exception as e:
                self.logger.error(f"Stream monitoring error: {e}")
                time.sleep(self.reconnect_interval)
    
    def _should_attempt_reconnect(self) -> bool:
        """Check if we should attempt reconnection"""
        if self.max_reconnect_attempts == -1:
            return True  # Infinite attempts
        return self.reconnect_count < self.max_reconnect_attempts
    
    def _attempt_connection(self):
        """Attempt to connect to RTSP stream"""
        try:
            self.logger.info(f"Attempting to connect to RTSP stream: {self.rtsp_uri}")
            
            # In a real implementation, this would test the RTSP connection
            # For now, we'll simulate connection success
            # You would typically use GStreamer elements to test connectivity
            
            self.is_connected = True
            self.reconnect_count += 1
            self.last_frame_time = time.time()
            
            self.logger.info(f"Connected to RTSP stream (attempt {self.reconnect_count})")
            
            # Call appropriate callback
            if self.reconnect_count == 1 and self.on_connected_callback:
                self.on_connected_callback()
            elif self.reconnect_count > 1 and self.on_reconnected_callback:
                self.on_reconnected_callback()
                
        except Exception as e:
            self.logger.error(f"Failed to connect to RTSP stream: {e}")
            self.is_connected = False
    
    def _check_stream_health(self) -> bool:
        """Check if stream is healthy (receiving frames)"""
        if not self.last_frame_time:
            return True  # No frame time set yet
        
        # Check if we've received frames recently (within 30 seconds)
        time_since_last_frame = time.time() - self.last_frame_time
        if time_since_last_frame > 30:
            self.logger.warning(f"No frames received for {time_since_last_frame:.1f} seconds")
            return False
        
        return True
    
    def _handle_disconnection(self):
        """Handle stream disconnection"""
        self.logger.warning("RTSP stream disconnected")
        self.is_connected = False
        self.last_frame_time = None
        
        if self.on_disconnected_callback:
            self.on_disconnected_callback()
    
    def update_frame_time(self):
        """Update last frame received time (called by pipeline probes)"""
        self.last_frame_time = time.time()
    
    def get_connection_status(self) -> dict:
        """Get current connection status"""
        return {
            'connected': self.is_connected,
            'reconnect_count': self.reconnect_count,
            'last_frame_time': self.last_frame_time,
            'rtsp_uri': self.rtsp_uri
        }


class StreamProbeHandler:
    """Handles GStreamer probe functions for stream monitoring"""
    
    def __init__(self, stream_manager: RTSPStreamManager):
        self.stream_manager = stream_manager
        self.logger = logging.getLogger(__name__)
    
    def src_pad_buffer_probe(self, pad, info, user_data):
        """Probe function for source pad to monitor incoming frames"""
        try:
            # Update frame time in stream manager
            self.stream_manager.update_frame_time()
            
            # Get buffer info
            gst_buffer = info.get_buffer()
            if gst_buffer:
                # Log frame info periodically
                if hasattr(self, '_frame_count'):
                    self._frame_count += 1
                else:
                    self._frame_count = 1
                
                # Log every 100 frames
                if self._frame_count % 100 == 0:
                    self.logger.debug(f"Processed {self._frame_count} frames from RTSP stream")
            
            return Gst.PadProbeReturn.OK
            
        except Exception as e:
            self.logger.error(f"Error in source probe: {e}")
            return Gst.PadProbeReturn.OK
    
    def attach_to_source(self, source_element):
        """Attach probe to source element"""
        try:
            # Get source pad
            src_pad = source_element.get_static_pad("src")
            if src_pad:
                # Add buffer probe
                src_pad.add_probe(
                    Gst.PadProbeType.BUFFER,
                    self.src_pad_buffer_probe,
                    None
                )
                self.logger.info("Stream monitoring probe attached to source")
            else:
                self.logger.warning("Could not get source pad for monitoring")
                
        except Exception as e:
            self.logger.error(f"Failed to attach stream probe: {e}")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def on_connected():
        print("Stream connected!")
    
    def on_disconnected():
        print("Stream disconnected!")
    
    def on_reconnected():
        print("Stream reconnected!")
    
    # Test RTSP stream manager
    rtsp_uri = "rtsp://192.168.1.100:554/stream"
    manager = RTSPStreamManager(rtsp_uri)
    manager.set_callbacks(on_connected, on_disconnected, on_reconnected)
    
    try:
        manager.start_monitoring()
        time.sleep(30)  # Monitor for 30 seconds
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        manager.stop_monitoring_thread()
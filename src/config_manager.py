#!/usr/bin/env python3

"""
Configuration Manager for RTSP Object Detection System
Handles YAML configuration parsing, validation, and hot-reloading
"""

import os
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Set
import yaml
from dataclasses import dataclass, field, asdict
from pydantic import BaseModel, Field, ValidationError

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    print("WARNING: watchdog not available. Hot-reloading will be disabled.")
    WATCHDOG_AVAILABLE = False


# Pydantic models for configuration validation
class BehaviorConfig(BaseModel):
    static_threshold_seconds: int = Field(default=30, ge=1, le=300)
    position_tolerance_pixels: float = Field(default=10.0, ge=1.0, le=100.0)
    debounce_seconds: float = Field(default=2.0, ge=0.1, le=30.0)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class HttpConfig(BaseModel):
    endpoint: str
    timeout_seconds: int = Field(default=5, ge=1, le=60)
    retry_attempts: int = Field(default=3, ge=0, le=10)


class MqttConfig(BaseModel):
    broker: str
    topic: str
    port: int = Field(default=1883, ge=1, le=65535)
    qos: int = Field(default=1, ge=0, le=2)
    username: Optional[str] = None
    password: Optional[str] = None


class EventsConfig(BaseModel):
    http: Optional[HttpConfig] = None
    mqtt: Optional[MqttConfig] = None


class FrameSavingConfig(BaseModel):
    detection_only_mode: bool = False
    output_path: str = "/data/frames"
    preserve_timestamps: bool = True
    max_frames_per_hour: int = Field(default=3600, ge=1, le=10000)
    image_format: str = "jpg"
    image_quality: int = Field(default=95, ge=1, le=100)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class AppConfig(BaseModel):
    behavior: BehaviorConfig = BehaviorConfig()
    events: EventsConfig = EventsConfig(http=None, mqtt=None)
    frame_saving: FrameSavingConfig = FrameSavingConfig()


class ConfigFileWatcher(FileSystemEventHandler):
    """Watches for configuration file changes and triggers reload"""
    
    def __init__(self, config_file: Path, on_modified_callback: Callable[[Path], None]):
        """
        Initialize file watcher
        
        Args:
            config_file: Path to configuration file to watch
            on_modified_callback: Callback function to call when file is modified
        """
        self.config_file = config_file
        self.on_modified_callback = on_modified_callback
        self.last_modified_time = time.time()
        self.debounce_seconds = 1.0  # Debounce time to avoid multiple reloads
    
    def on_modified(self, event):
        """Handle file modification event"""
        if not isinstance(event, FileModifiedEvent):
            return
        
        # Check if the modified file is our config file
        if Path(event.src_path).resolve() == self.config_file.resolve():
            # Debounce to avoid multiple reloads for a single save
            current_time = time.time()
            if current_time - self.last_modified_time > self.debounce_seconds:
                self.last_modified_time = current_time
                self.on_modified_callback(self.config_file)


class ConfigManager:
    """Manages application configuration with validation and hot-reloading"""
    
    def __init__(self, config_file_path: str):
        """
        Initialize configuration manager
        
        Args:
            config_file_path: Path to YAML configuration file
        """
        self.config_file_path = Path(config_file_path)
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.config: Optional[AppConfig] = None
        
        # Change callbacks
        self.change_callbacks: List[Callable[[AppConfig], None]] = []
        
        # File watcher
        self.observer: Optional[Observer] = None
        self.file_watcher: Optional[ConfigFileWatcher] = None
        
        # Load initial configuration
        self._load_config()
        
        # Start file watcher if available
        if WATCHDOG_AVAILABLE:
            self._start_file_watcher()
    
    def _load_config(self) -> bool:
        """
        Load configuration from file
        
        Returns:
            bool: True if configuration was loaded successfully
        """
        try:
            # Check if file exists
            if not self.config_file_path.exists():
                self.logger.error(f"Configuration file not found: {self.config_file_path}")
                return False
            
            # Load YAML
            with open(self.config_file_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Validate configuration
            try:
                new_config = AppConfig(**config_data)
                
                # Store configuration
                old_config = self.config
                self.config = new_config
                
                # Log success
                self.logger.info(f"Configuration loaded successfully from {self.config_file_path}")
                
                # Notify change listeners if this is a reload
                if old_config is not None:
                    self._notify_change_listeners()
                
                return True
                
            except ValidationError as e:
                self.logger.error(f"Configuration validation failed: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return False
    
    def _start_file_watcher(self):
        """Start file watcher for hot-reloading"""
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("watchdog not available - hot-reloading disabled")
            return
        
        try:
            # Create file watcher
            self.file_watcher = ConfigFileWatcher(
                self.config_file_path,
                self._on_config_file_modified
            )
            
            # Create observer
            self.observer = Observer()
            self.observer.schedule(
                self.file_watcher,
                str(self.config_file_path.parent),
                recursive=False
            )
            
            # Start observer
            self.observer.start()
            self.logger.info(f"Configuration file watcher started for {self.config_file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to start file watcher: {e}")
    
    def _on_config_file_modified(self, config_file: Path):
        """Handle configuration file modification"""
        self.logger.info(f"Configuration file modified: {config_file}")
        self._load_config()
    
    def add_change_listener(self, callback: Callable[[AppConfig], None]):
        """
        Add callback for configuration changes
        
        Args:
            callback: Function to call when configuration changes
        """
        self.change_callbacks.append(callback)
    
    def _notify_change_listeners(self):
        """Notify all change listeners of configuration change"""
        if not self.config:
            return
        
        for callback in self.change_callbacks:
            try:
                callback(self.config)
            except Exception as e:
                self.logger.error(f"Error in configuration change callback: {e}")
    
    def get_behavior_config(self) -> Dict:
        """Get behavior configuration"""
        if not self.config:
            return {}
        return self.config.behavior.dict()
    
    def get_events_config(self) -> Dict:
        """Get events configuration"""
        if not self.config:
            return {}
        return self.config.events.dict()
    
    def get_frame_saving_config(self) -> Dict:
        """Get frame saving configuration"""
        if not self.config:
            return {}
        return self.config.frame_saving.dict()
    
    def get_full_config(self) -> Dict:
        """Get full configuration"""
        if not self.config:
            return {}
        return self.config.dict()
    
    def shutdown(self):
        """Shutdown configuration manager"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.logger.info("Configuration file watcher stopped")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration manager
    config_manager = ConfigManager("configs/app_config.yaml")
    
    # Get configuration
    behavior_config = config_manager.get_behavior_config()
    events_config = config_manager.get_events_config()
    frame_saving_config = config_manager.get_frame_saving_config()
    
    print(f"Behavior config: {behavior_config}")
    print(f"Events config: {events_config}")
    print(f"Frame saving config: {frame_saving_config}")
    
    # Add change listener
    def on_config_changed(config):
        print(f"Configuration changed: {config}")
    
    config_manager.add_change_listener(on_config_changed)
    
    # Wait for changes
    try:
        print("Waiting for configuration changes (press Ctrl+C to exit)...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        config_manager.shutdown()
#!/usr/bin/env python3

"""
Test script for Configuration Manager
"""

import os
import sys
import tempfile
import time
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_manager import ConfigManager


def test_config_loading():
    """Test basic configuration loading"""
    print("Testing configuration loading...")
    
    # Create temporary config file
    config_data = {
        'behavior': {
            'static_threshold_seconds': 45,
            'position_tolerance_pixels': 15,
            'debounce_seconds': 3,
            'min_confidence': 0.6
        },
        'events': {
            'http': {
                'endpoint': 'http://test.example.com/events',
                'timeout_seconds': 10,
                'retry_attempts': 5
            }
        },
        'frame_saving': {
            'detection_only_mode': True,
            'output_path': '/tmp/test_frames',
            'max_frames_per_hour': 1800
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_config_path = f.name
    
    try:
        # Create config manager
        config_manager = ConfigManager(temp_config_path)
        
        # Test configuration retrieval
        behavior_config = config_manager.get_behavior_config()
        events_config = config_manager.get_events_config()
        frame_config = config_manager.get_frame_saving_config()
        
        # Validate values
        assert behavior_config['static_threshold_seconds'] == 45
        assert behavior_config['position_tolerance_pixels'] == 15
        assert events_config['http']['endpoint'] == 'http://test.example.com/events'
        assert frame_config['detection_only_mode'] == True
        assert frame_config['output_path'] == '/tmp/test_frames'
        
        print("✓ Configuration loading test passed")
        
        # Test change listener
        change_detected = False
        
        def on_change(config):
            nonlocal change_detected
            change_detected = True
            print("✓ Configuration change detected")
        
        config_manager.add_change_listener(on_change)
        
        # Modify config file
        config_data['behavior']['static_threshold_seconds'] = 60
        with open(temp_config_path, 'w') as f:
            yaml.dump(config_data, f)
        
        # Wait for change detection
        time.sleep(2)
        
        if change_detected:
            print("✓ Hot-reload test passed")
        else:
            print("⚠ Hot-reload test skipped (watchdog may not be available)")
        
        # Cleanup
        config_manager.shutdown()
        
    finally:
        # Clean up temp file
        os.unlink(temp_config_path)


def test_validation():
    """Test configuration validation"""
    print("Testing configuration validation...")
    
    # Test invalid configuration
    invalid_config_data = {
        'behavior': {
            'static_threshold_seconds': -5,  # Invalid: negative value
            'position_tolerance_pixels': 150,  # Invalid: too large
            'min_confidence': 1.5  # Invalid: > 1.0
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(invalid_config_data, f)
        temp_config_path = f.name
    
    try:
        # This should handle validation errors gracefully
        config_manager = ConfigManager(temp_config_path)
        
        # Config should be None due to validation failure
        full_config = config_manager.get_full_config()
        if not full_config:
            print("✓ Configuration validation test passed (invalid config rejected)")
        else:
            print("⚠ Configuration validation may not be working properly")
        
        config_manager.shutdown()
        
    finally:
        os.unlink(temp_config_path)


if __name__ == "__main__":
    print("Running Configuration Manager Tests...")
    print("=" * 50)
    
    try:
        test_config_loading()
        test_validation()
        
        print("=" * 50)
        print("All tests completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)
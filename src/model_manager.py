#!/usr/bin/env python3

"""
Model Manager for ONNX/TensorRT models in DeepStream
Handles model loading, validation, and configuration
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List

try:
    import pyds
except ImportError:
    print("ERROR: DeepStream Python bindings (pyds) not found.")


class ModelManager:
    """Manages AI model loading and configuration for DeepStream"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models: Dict[str, Dict] = {}
        
    def validate_model_files(self, model_config: Dict) -> bool:
        """
        Validate that model files exist and are accessible
        
        Args:
            model_config: Dictionary containing model configuration
            
        Returns:
            bool: True if all model files are valid
        """
        try:
            required_files = []
            
            # Check ONNX file if specified
            if 'onnx_file' in model_config and model_config['onnx_file']:
                required_files.append(('ONNX model', model_config['onnx_file']))
            
            # Check TensorRT engine file if specified
            if 'engine_file' in model_config and model_config['engine_file']:
                required_files.append(('TensorRT engine', model_config['engine_file']))
            
            # Check labels file
            if 'labels_file' in model_config and model_config['labels_file']:
                required_files.append(('Labels file', model_config['labels_file']))
            
            # Check config file
            if 'config_file' in model_config and model_config['config_file']:
                required_files.append(('Config file', model_config['config_file']))
            
            # Validate each file exists
            for file_type, file_path in required_files:
                if not Path(file_path).exists():
                    self.logger.error(f"{file_type} not found: {file_path}")
                    return False
                else:
                    self.logger.info(f"{file_type} validated: {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating model files: {e}")
            return False
    
    def load_labels(self, labels_file: str) -> Optional[List[str]]:
        """
        Load class labels from file
        
        Args:
            labels_file: Path to labels file
            
        Returns:
            List of class labels or None if failed
        """
        try:
            labels_path = Path(labels_file)
            if not labels_path.exists():
                self.logger.error(f"Labels file not found: {labels_file}")
                return None
            
            with open(labels_path, 'r') as f:
                labels = [line.strip() for line in f.readlines() if line.strip()]
            
            self.logger.info(f"Loaded {len(labels)} class labels from {labels_file}")
            return labels
            
        except Exception as e:
            self.logger.error(f"Failed to load labels from {labels_file}: {e}")
            return None
    
    def validate_model_config(self, config_file: str) -> bool:
        """
        Validate DeepStream model configuration file
        
        Args:
            config_file: Path to DeepStream model config file
            
        Returns:
            bool: True if config is valid
        """
        try:
            config_path = Path(config_file)
            if not config_path.exists():
                self.logger.error(f"Model config file not found: {config_file}")
                return False
            
            # Read and validate basic config structure
            with open(config_path, 'r') as f:
                content = f.read()
            
            # Check for required sections
            required_sections = ['[property]']
            for section in required_sections:
                if section not in content:
                    self.logger.error(f"Missing required section {section} in {config_file}")
                    return False
            
            # Check for required properties
            required_properties = [
                'gpu-id',
                'batch-size',
                'network-mode',
                'num-detected-classes',
                'gie-unique-id'
            ]
            
            for prop in required_properties:
                if prop not in content:
                    self.logger.warning(f"Missing recommended property {prop} in {config_file}")
            
            self.logger.info(f"Model config validated: {config_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate model config {config_file}: {e}")
            return False
    
    def check_tensorrt_engine(self, onnx_file: str, engine_file: str) -> bool:
        """
        Check if TensorRT engine needs to be regenerated
        
        Args:
            onnx_file: Path to ONNX model file
            engine_file: Path to TensorRT engine file
            
        Returns:
            bool: True if engine is up to date, False if needs regeneration
        """
        try:
            onnx_path = Path(onnx_file)
            engine_path = Path(engine_file)
            
            # If engine doesn't exist, it needs to be generated
            if not engine_path.exists():
                self.logger.info(f"TensorRT engine will be generated: {engine_file}")
                return False
            
            # If ONNX is newer than engine, regenerate
            if onnx_path.exists():
                onnx_mtime = onnx_path.stat().st_mtime
                engine_mtime = engine_path.stat().st_mtime
                
                if onnx_mtime > engine_mtime:
                    self.logger.info(f"ONNX model is newer than engine, will regenerate: {engine_file}")
                    return False
            
            self.logger.info(f"TensorRT engine is up to date: {engine_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking TensorRT engine: {e}")
            return False
    
    def register_model(self, model_id: str, model_config: Dict) -> bool:
        """
        Register a model configuration
        
        Args:
            model_id: Unique identifier for the model
            model_config: Model configuration dictionary
            
        Returns:
            bool: True if registration successful
        """
        try:
            # Validate model configuration
            if not self.validate_model_files(model_config):
                return False
            
            if 'config_file' in model_config:
                if not self.validate_model_config(model_config['config_file']):
                    return False
            
            # Load labels if specified
            if 'labels_file' in model_config:
                labels = self.load_labels(model_config['labels_file'])
                if labels:
                    model_config['labels'] = labels
            
            # Check TensorRT engine status
            if 'onnx_file' in model_config and 'engine_file' in model_config:
                engine_status = self.check_tensorrt_engine(
                    model_config['onnx_file'],
                    model_config['engine_file']
                )
                model_config['engine_up_to_date'] = engine_status
            
            # Register the model
            self.models[model_id] = model_config
            self.logger.info(f"Model registered successfully: {model_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register model {model_id}: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get information about a registered model"""
        return self.models.get(model_id)
    
    def list_models(self) -> List[str]:
        """Get list of registered model IDs"""
        return list(self.models.keys())
    
    def validate_gpu_memory(self, gpu_id: int = 0) -> Dict:
        """
        Check GPU memory availability using NVIDIA Management Library
        
        Args:
            gpu_id: GPU device ID
            
        Returns:
            Dictionary with memory information
        """
        try:
            # Try to use pynvml for accurate GPU memory information
            try:
                import pynvml
                pynvml.nvmlInit()
                
                handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                total_memory_mb = mem_info.total / (1024 * 1024)
                free_memory_mb = mem_info.free / (1024 * 1024)
                used_memory_mb = mem_info.used / (1024 * 1024)
                memory_usage_percent = (used_memory_mb / total_memory_mb) * 100
                
                # Consider memory sufficient if less than 80% used and at least 2GB free
                sufficient_memory = (memory_usage_percent < 80.0) and (free_memory_mb > 2048)
                
                memory_info = {
                    'gpu_id': gpu_id,
                    'total_memory_mb': total_memory_mb,
                    'free_memory_mb': free_memory_mb,
                    'used_memory_mb': used_memory_mb,
                    'memory_usage_percent': memory_usage_percent,
                    'sufficient_memory': sufficient_memory
                }
                
                self.logger.info(f"GPU {gpu_id} memory: {free_memory_mb:.0f}MB free of {total_memory_mb:.0f}MB total")
                return memory_info
                
            except ImportError:
                # Fallback: Use nvidia-smi command if pynvml not available
                import subprocess
                
                cmd = f"nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,noheader,nounits -i {gpu_id}"
                result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    values = result.stdout.strip().split(', ')
                    total_memory_mb = float(values[0])
                    free_memory_mb = float(values[1])
                    used_memory_mb = float(values[2])
                    memory_usage_percent = (used_memory_mb / total_memory_mb) * 100
                    
                    sufficient_memory = (memory_usage_percent < 80.0) and (free_memory_mb > 2048)
                    
                    memory_info = {
                        'gpu_id': gpu_id,
                        'total_memory_mb': total_memory_mb,
                        'free_memory_mb': free_memory_mb,
                        'used_memory_mb': used_memory_mb,
                        'memory_usage_percent': memory_usage_percent,
                        'sufficient_memory': sufficient_memory
                    }
                    
                    self.logger.info(f"GPU {gpu_id} memory: {free_memory_mb:.0f}MB free of {total_memory_mb:.0f}MB total")
                    return memory_info
                else:
                    raise Exception(f"nvidia-smi command failed: {result.stderr}")
            
        except Exception as e:
            self.logger.warning(f"Could not check GPU memory accurately: {e}")
            # Return conservative estimates
            return {
                'gpu_id': gpu_id,
                'total_memory_mb': 8192,  # Conservative estimate
                'free_memory_mb': 4096,   # Conservative estimate
                'used_memory_mb': 4096,   # Conservative estimate
                'memory_usage_percent': 50.0,
                'sufficient_memory': True,  # Assume sufficient for now
                'error': str(e),
                'method': 'fallback_estimate'
            }


def create_detection_model_config(
    onnx_file: str,
    engine_file: str,
    labels_file: str,
    config_file: str,
    num_classes: int = 80,
    batch_size: int = 1,
    gpu_id: int = 0
) -> Dict:
    """
    Create a standard detection model configuration
    
    Args:
        onnx_file: Path to ONNX model file
        engine_file: Path to TensorRT engine file
        labels_file: Path to class labels file
        config_file: Path to DeepStream config file
        num_classes: Number of detection classes
        batch_size: Inference batch size
        gpu_id: GPU device ID
        
    Returns:
        Model configuration dictionary
    """
    return {
        'onnx_file': onnx_file,
        'engine_file': engine_file,
        'labels_file': labels_file,
        'config_file': config_file,
        'num_classes': num_classes,
        'batch_size': batch_size,
        'gpu_id': gpu_id,
        'model_type': 'detection'
    }


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create model manager
    manager = ModelManager()
    
    # Create detection model config
    detection_config = create_detection_model_config(
        onnx_file="/models/detection.onnx",
        engine_file="/models/detection.engine",
        labels_file="/configs/labels.txt",
        config_file="/configs/detection_config.txt"
    )
    
    # Register model
    success = manager.register_model("primary_detector", detection_config)
    print(f"Model registration: {'Success' if success else 'Failed'}")
    
    # Check GPU memory
    memory_info = manager.validate_gpu_memory()
    print(f"GPU Memory: {memory_info}")
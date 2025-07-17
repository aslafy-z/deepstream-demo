#!/usr/bin/env python3

"""
Performance Monitoring and Optimization for RTSP Object Detection System
Integrates with DeepStream performance measurement and system monitoring
"""

import time
import threading
import psutil
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from collections import deque
import structlog

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("WARNING: pynvml not available. GPU monitoring will be limited.")

try:
    import pyds
    DEEPSTREAM_AVAILABLE = True
except ImportError:
    DEEPSTREAM_AVAILABLE = False
    print("WARNING: DeepStream Python bindings not available. DeepStream metrics will be limited.")


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: str
    fps: float
    latency_ms: float
    cpu_usage: float
    memory_usage_mb: float
    gpu_usage: float
    gpu_memory_usage_mb: float
    frames_processed: int
    objects_detected: int
    inference_time_ms: float
    tracking_time_ms: float


@dataclass
class SystemResources:
    """System resource information"""
    cpu_count: int
    total_memory_gb: float
    gpu_count: int
    gpu_memory_total_mb: List[float]
    gpu_compute_capability: List[str]


class GPUMonitor:
    """GPU monitoring using NVIDIA Management Library"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.initialized = False
        self.device_count = 0
        
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.device_count = pynvml.nvmlDeviceGetCount()
                self.initialized = True
                self.logger.info("GPU monitoring initialized", device_count=self.device_count)
            except Exception as e:
                self.logger.warning("Failed to initialize GPU monitoring", error=str(e))
    
    def get_gpu_metrics(self, device_id: int = 0) -> Dict[str, Any]:
        """Get GPU metrics for specified device"""
        if not self.initialized or device_id >= self.device_count:
            return {
                'usage_percent': 0.0,
                'memory_used_mb': 0.0,
                'memory_total_mb': 0.0,
                'memory_free_mb': 0.0,
                'temperature_c': 0.0,
                'power_usage_w': 0.0
            }
        
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = util.gpu
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used_mb = mem_info.used / (1024 * 1024)
            memory_total_mb = mem_info.total / (1024 * 1024)
            memory_free_mb = mem_info.free / (1024 * 1024)
            
            # Temperature
            try:
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temperature = 0.0
            
            # Power usage
            try:
                power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
            except:
                power_usage = 0.0
            
            return {
                'usage_percent': float(gpu_usage),
                'memory_used_mb': float(memory_used_mb),
                'memory_total_mb': float(memory_total_mb),
                'memory_free_mb': float(memory_free_mb),
                'temperature_c': float(temperature),
                'power_usage_w': float(power_usage)
            }
            
        except Exception as e:
            self.logger.error("Failed to get GPU metrics", device_id=device_id, error=str(e))
            return {
                'usage_percent': 0.0,
                'memory_used_mb': 0.0,
                'memory_total_mb': 0.0,
                'memory_free_mb': 0.0,
                'temperature_c': 0.0,
                'power_usage_w': 0.0
            }
    
    def get_all_gpu_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics for all available GPUs"""
        return [self.get_gpu_metrics(i) for i in range(self.device_count)]
    
    def get_system_resources(self) -> SystemResources:
        """Get system resource information"""
        cpu_count = psutil.cpu_count()
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        gpu_memory_total = []
        gpu_compute_capability = []
        
        if self.initialized:
            for i in range(self.device_count):
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_total.append(mem_info.total / (1024 * 1024))
                    
                    # Get compute capability
                    try:
                        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                        gpu_compute_capability.append(f"{major}.{minor}")
                    except:
                        gpu_compute_capability.append("unknown")
                        
                except Exception:
                    gpu_memory_total.append(0.0)
                    gpu_compute_capability.append("unknown")
        
        return SystemResources(
            cpu_count=cpu_count,
            total_memory_gb=total_memory_gb,
            gpu_count=self.device_count,
            gpu_memory_total_mb=gpu_memory_total,
            gpu_compute_capability=gpu_compute_capability
        )


class DeepStreamPerformanceMonitor:
    """DeepStream-specific performance monitoring"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.metrics_history = deque(maxlen=1000)
        self.last_frame_count = 0
        self.last_timestamp = time.time()
        self.fps_history = deque(maxlen=60)  # Last 60 FPS measurements
    
    def process_deepstream_metadata(self, batch_meta) -> Dict[str, Any]:
        """Process DeepStream batch metadata for performance metrics"""
        if not DEEPSTREAM_AVAILABLE:
            return {}
        
        try:
            current_time = time.time()
            frame_count = 0
            object_count = 0
            inference_times = []
            
            # Process batch metadata
            l_frame = batch_meta.frame_meta_list
            while l_frame is not None:
                try:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                    frame_count += 1
                    
                    # Count objects in this frame
                    l_obj = frame_meta.obj_meta_list
                    while l_obj is not None:
                        object_count += 1
                        l_obj = l_obj.next
                    
                    # Get inference time if available
                    if hasattr(frame_meta, 'ntp_timestamp'):
                        inference_time = (current_time * 1000) - (frame_meta.ntp_timestamp / 1000000)
                        if inference_time > 0:
                            inference_times.append(inference_time)
                    
                except Exception as e:
                    self.logger.debug("Error processing frame metadata", error=str(e))
                
                try:
                    l_frame = l_frame.next
                except StopIteration:
                    break
            
            # Calculate FPS
            time_diff = current_time - self.last_timestamp
            if time_diff > 0:
                fps = (frame_count - self.last_frame_count) / time_diff
                self.fps_history.append(fps)
            else:
                fps = 0.0
            
            self.last_frame_count = frame_count
            self.last_timestamp = current_time
            
            # Calculate average inference time
            avg_inference_time = sum(inference_times) / len(inference_times) if inference_times else 0.0
            
            return {
                'frames_processed': frame_count,
                'objects_detected': object_count,
                'fps': fps,
                'avg_inference_time_ms': avg_inference_time,
                'timestamp': current_time
            }
            
        except Exception as e:
            self.logger.error("Failed to process DeepStream metadata", error=str(e))
            return {}
    
    def get_average_fps(self, window_seconds: int = 10) -> float:
        """Get average FPS over specified time window"""
        if not self.fps_history:
            return 0.0
        
        # Take last N measurements based on window
        window_size = min(window_seconds, len(self.fps_history))
        recent_fps = list(self.fps_history)[-window_size:]
        
        return sum(recent_fps) / len(recent_fps) if recent_fps else 0.0


class PerformanceOptimizer:
    """Performance optimization recommendations and automatic tuning"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.optimization_history = []
    
    def analyze_performance(self, metrics: PerformanceMetrics, system_resources: SystemResources) -> Dict[str, Any]:
        """Analyze performance metrics and provide optimization recommendations"""
        recommendations = []
        warnings = []
        
        # FPS analysis
        if metrics.fps < 15:
            recommendations.append({
                'type': 'fps_low',
                'message': 'Low FPS detected. Consider reducing input resolution or model complexity.',
                'priority': 'high'
            })
        
        # GPU utilization analysis
        if metrics.gpu_usage < 50:
            recommendations.append({
                'type': 'gpu_underutilized',
                'message': 'GPU underutilized. Consider increasing batch size or model complexity.',
                'priority': 'medium'
            })
        elif metrics.gpu_usage > 95:
            warnings.append({
                'type': 'gpu_overutilized',
                'message': 'GPU heavily utilized. Performance may be limited.',
                'priority': 'high'
            })
        
        # Memory analysis
        memory_usage_percent = (metrics.gpu_memory_usage_mb / system_resources.gpu_memory_total_mb[0]) * 100 if system_resources.gpu_memory_total_mb else 0
        
        if memory_usage_percent > 90:
            warnings.append({
                'type': 'gpu_memory_high',
                'message': 'GPU memory usage is very high. Risk of out-of-memory errors.',
                'priority': 'critical'
            })
        elif memory_usage_percent > 80:
            recommendations.append({
                'type': 'gpu_memory_optimize',
                'message': 'GPU memory usage is high. Consider reducing batch size or model size.',
                'priority': 'medium'
            })
        
        # Latency analysis
        if metrics.latency_ms > 100:
            recommendations.append({
                'type': 'latency_high',
                'message': 'High latency detected. Check network connectivity and processing pipeline.',
                'priority': 'high'
            })
        
        # CPU analysis
        if metrics.cpu_usage > 80:
            recommendations.append({
                'type': 'cpu_high',
                'message': 'High CPU usage. Consider optimizing preprocessing or using GPU acceleration.',
                'priority': 'medium'
            })
        
        return {
            'recommendations': recommendations,
            'warnings': warnings,
            'overall_health': self._calculate_health_score(metrics, system_resources),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_health_score(self, metrics: PerformanceMetrics, system_resources: SystemResources) -> Dict[str, Any]:
        """Calculate overall system health score"""
        score = 100
        factors = []
        
        # FPS factor
        if metrics.fps < 10:
            score -= 30
            factors.append("Very low FPS")
        elif metrics.fps < 20:
            score -= 15
            factors.append("Low FPS")
        
        # GPU utilization factor
        if metrics.gpu_usage > 95:
            score -= 20
            factors.append("GPU overutilized")
        elif metrics.gpu_usage < 30:
            score -= 10
            factors.append("GPU underutilized")
        
        # Memory factor
        if system_resources.gpu_memory_total_mb:
            memory_usage_percent = (metrics.gpu_memory_usage_mb / system_resources.gpu_memory_total_mb[0]) * 100
            if memory_usage_percent > 90:
                score -= 25
                factors.append("Critical GPU memory usage")
            elif memory_usage_percent > 80:
                score -= 10
                factors.append("High GPU memory usage")
        
        # Latency factor
        if metrics.latency_ms > 200:
            score -= 20
            factors.append("Very high latency")
        elif metrics.latency_ms > 100:
            score -= 10
            factors.append("High latency")
        
        # CPU factor
        if metrics.cpu_usage > 90:
            score -= 15
            factors.append("Very high CPU usage")
        elif metrics.cpu_usage > 80:
            score -= 5
            factors.append("High CPU usage")
        
        score = max(0, score)  # Ensure score doesn't go below 0
        
        if score >= 90:
            status = "excellent"
        elif score >= 75:
            status = "good"
        elif score >= 60:
            status = "fair"
        elif score >= 40:
            status = "poor"
        else:
            status = "critical"
        
        return {
            'score': score,
            'status': status,
            'factors': factors
        }


class PerformanceManager:
    """Main performance management class"""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.logger = structlog.get_logger(__name__)
        self.monitoring_interval = monitoring_interval
        self.running = False
        self.monitor_thread = None
        
        # Initialize components
        self.gpu_monitor = GPUMonitor()
        self.deepstream_monitor = DeepStreamPerformanceMonitor()
        self.optimizer = PerformanceOptimizer()
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = None
        self.system_resources = self.gpu_monitor.get_system_resources()
        
        self.logger.info("Performance manager initialized", 
                        monitoring_interval=monitoring_interval,
                        gpu_count=self.system_resources.gpu_count)
    
    def start_monitoring(self):
        """Start performance monitoring thread"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                metrics = self._collect_metrics()
                if metrics:
                    self.current_metrics = metrics
                    self.metrics_history.append(metrics)
                    
                    # Log performance summary periodically
                    if len(self.metrics_history) % 12 == 0:  # Every minute at 5s intervals
                        self._log_performance_summary()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error("Error in monitoring loop", error=str(e))
                time.sleep(self.monitoring_interval)
    
    def _collect_metrics(self) -> Optional[PerformanceMetrics]:
        """Collect current performance metrics"""
        try:
            # System metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / (1024 * 1024)
            
            # GPU metrics
            gpu_metrics = self.gpu_monitor.get_gpu_metrics(0)
            
            # DeepStream metrics (if available)
            fps = self.deepstream_monitor.get_average_fps()
            
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=datetime.now().isoformat(),
                fps=fps,
                latency_ms=0.0,  # Would be calculated from actual pipeline
                cpu_usage=cpu_usage,
                memory_usage_mb=memory_usage_mb,
                gpu_usage=gpu_metrics['usage_percent'],
                gpu_memory_usage_mb=gpu_metrics['memory_used_mb'],
                frames_processed=0,  # Would be updated by pipeline
                objects_detected=0,  # Would be updated by pipeline
                inference_time_ms=0.0,  # Would be calculated from DeepStream
                tracking_time_ms=0.0   # Would be calculated from DeepStream
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error("Failed to collect metrics", error=str(e))
            return None
    
    def _log_performance_summary(self):
        """Log performance summary"""
        if not self.current_metrics:
            return
        
        analysis = self.optimizer.analyze_performance(self.current_metrics, self.system_resources)
        
        self.logger.info(
            "Performance summary",
            fps=self.current_metrics.fps,
            cpu_usage=self.current_metrics.cpu_usage,
            gpu_usage=self.current_metrics.gpu_usage,
            gpu_memory_mb=self.current_metrics.gpu_memory_usage_mb,
            health_score=analysis['overall_health']['score'],
            health_status=analysis['overall_health']['status']
        )
        
        # Log warnings
        for warning in analysis['warnings']:
            self.logger.warning("Performance warning", 
                              type=warning['type'], 
                              message=warning['message'],
                              priority=warning['priority'])
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current performance metrics"""
        return self.current_metrics
    
    def get_metrics_history(self, limit: int = 100) -> List[PerformanceMetrics]:
        """Get performance metrics history"""
        return list(self.metrics_history)[-limit:]
    
    def get_performance_analysis(self) -> Dict[str, Any]:
        """Get performance analysis and recommendations"""
        if not self.current_metrics:
            return {'error': 'No metrics available'}
        
        return self.optimizer.analyze_performance(self.current_metrics, self.system_resources)
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            'system_resources': asdict(self.system_resources),
            'gpu_metrics': self.gpu_monitor.get_all_gpu_metrics(),
            'monitoring_active': self.running,
            'metrics_count': len(self.metrics_history)
        }
    
    def update_pipeline_metrics(self, frames_processed: int, objects_detected: int, 
                              inference_time_ms: float, tracking_time_ms: float):
        """Update pipeline-specific metrics"""
        if self.current_metrics:
            self.current_metrics.frames_processed = frames_processed
            self.current_metrics.objects_detected = objects_detected
            self.current_metrics.inference_time_ms = inference_time_ms
            self.current_metrics.tracking_time_ms = tracking_time_ms


# Global performance manager instance
performance_manager: Optional[PerformanceManager] = None


def initialize_performance_monitoring(monitoring_interval: float = 5.0) -> PerformanceManager:
    """Initialize global performance monitoring"""
    global performance_manager
    
    if performance_manager is None:
        performance_manager = PerformanceManager(monitoring_interval)
        performance_manager.start_monitoring()
    
    return performance_manager


def get_performance_manager() -> Optional[PerformanceManager]:
    """Get global performance manager instance"""
    return performance_manager


if __name__ == "__main__":
    # Test performance monitoring
    print("Testing Performance Monitoring System")
    print("=" * 50)
    
    # Initialize performance manager
    perf_manager = initialize_performance_monitoring(monitoring_interval=2.0)
    
    try:
        # Run for a short time
        time.sleep(10)
        
        # Get current metrics
        metrics = perf_manager.get_current_metrics()
        if metrics:
            print(f"Current FPS: {metrics.fps:.2f}")
            print(f"CPU Usage: {metrics.cpu_usage:.1f}%")
            print(f"GPU Usage: {metrics.gpu_usage:.1f}%")
            print(f"GPU Memory: {metrics.gpu_memory_usage_mb:.1f} MB")
        
        # Get performance analysis
        analysis = perf_manager.get_performance_analysis()
        print(f"Health Score: {analysis['overall_health']['score']}")
        print(f"Health Status: {analysis['overall_health']['status']}")
        
        # Get system info
        system_info = perf_manager.get_system_info()
        print(f"GPU Count: {system_info['system_resources']['gpu_count']}")
        print(f"Total Memory: {system_info['system_resources']['total_memory_gb']:.1f} GB")
        
    finally:
        perf_manager.stop_monitoring()
#!/usr/bin/env python3

"""
Structured Logging Configuration for RTSP Object Detection System
Provides comprehensive logging with DeepStream integration
"""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import structlog
from structlog.stdlib import LoggerFactory


class DeepStreamLogHandler(logging.Handler):
    """Custom log handler for DeepStream-specific logging"""
    
    def __init__(self):
        super().__init__()
        self.component_stats = {}
    
    def emit(self, record):
        """Process log record and extract DeepStream metrics"""
        try:
            # Extract component information
            if hasattr(record, 'component'):
                component = record.component
                if component not in self.component_stats:
                    self.component_stats[component] = {
                        'error_count': 0,
                        'warning_count': 0,
                        'last_error': None,
                        'last_warning': None
                    }
                
                if record.levelno >= logging.ERROR:
                    self.component_stats[component]['error_count'] += 1
                    self.component_stats[component]['last_error'] = {
                        'message': record.getMessage(),
                        'timestamp': datetime.now().isoformat()
                    }
                elif record.levelno >= logging.WARNING:
                    self.component_stats[component]['warning_count'] += 1
                    self.component_stats[component]['last_warning'] = {
                        'message': record.getMessage(),
                        'timestamp': datetime.now().isoformat()
                    }
        
        except Exception:
            # Don't let logging errors break the application
            pass
    
    def get_component_stats(self) -> Dict[str, Any]:
        """Get component error/warning statistics"""
        return self.component_stats.copy()


class HealthMonitor:
    """System health monitoring and status reporting"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.component_health = {}
        self.system_metrics = {
            'start_time': datetime.now(),
            'total_frames_processed': 0,
            'total_objects_detected': 0,
            'total_events_generated': 0,
            'error_count': 0,
            'warning_count': 0
        }
    
    def update_component_health(self, component: str, status: str, details: Optional[Dict] = None):
        """Update health status for a component"""
        self.component_health[component] = {
            'status': status,  # 'healthy', 'degraded', 'unhealthy'
            'last_update': datetime.now().isoformat(),
            'details': details or {}
        }
        
        self.logger.info(
            "Component health updated",
            component=component,
            status=status,
            details=details
        )
    
    def increment_metric(self, metric: str, value: int = 1):
        """Increment system metric"""
        if metric in self.system_metrics:
            self.system_metrics[metric] += value
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        uptime = datetime.now() - self.system_metrics['start_time']
        
        return {
            'system': {
                'status': self._get_overall_status(),
                'uptime_seconds': uptime.total_seconds(),
                'metrics': self.system_metrics
            },
            'components': self.component_health,
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_overall_status(self) -> str:
        """Determine overall system status"""
        if not self.component_health:
            return 'unknown'
        
        statuses = [comp['status'] for comp in self.component_health.values()]
        
        if any(status == 'unhealthy' for status in statuses):
            return 'unhealthy'
        elif any(status == 'degraded' for status in statuses):
            return 'degraded'
        else:
            return 'healthy'


class ErrorRecoveryManager:
    """Manages error recovery mechanisms"""
    
    def __init__(self):
        self.logger = structlog.get_logger(__name__)
        self.recovery_strategies = {}
        self.error_history = []
        self.max_history = 100
    
    def register_recovery_strategy(self, error_type: str, strategy_func):
        """Register recovery strategy for specific error type"""
        self.recovery_strategies[error_type] = strategy_func
        self.logger.info("Recovery strategy registered", error_type=error_type)
    
    def handle_error(self, error_type: str, error_details: Dict, component: str) -> bool:
        """Handle error with appropriate recovery strategy"""
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'component': component,
            'details': error_details
        }
        
        # Add to history
        self.error_history.append(error_record)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        self.logger.error(
            "Error occurred",
            error_type=error_type,
            component=component,
            details=error_details
        )
        
        # Attempt recovery
        if error_type in self.recovery_strategies:
            try:
                recovery_func = self.recovery_strategies[error_type]
                success = recovery_func(error_details, component)
                
                if success:
                    self.logger.info(
                        "Error recovery successful",
                        error_type=error_type,
                        component=component
                    )
                else:
                    self.logger.warning(
                        "Error recovery failed",
                        error_type=error_type,
                        component=component
                    )
                
                return success
                
            except Exception as e:
                self.logger.error(
                    "Recovery strategy failed",
                    error_type=error_type,
                    component=component,
                    recovery_error=str(e)
                )
        
        return False
    
    def get_error_history(self) -> list:
        """Get recent error history"""
        return self.error_history.copy()


def setup_structured_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json: bool = False,
    enable_deepstream_handler: bool = True
) -> tuple:
    """
    Set up structured logging configuration
    
    Returns:
        tuple: (health_monitor, error_recovery_manager, deepstream_handler)
    """
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if enable_json else structlog.dev.ConsoleRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        
        if enable_json:
            file_formatter = logging.Formatter('%(message)s')
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # DeepStream-specific handler
    deepstream_handler = None
    if enable_deepstream_handler:
        deepstream_handler = DeepStreamLogHandler()
        root_logger.addHandler(deepstream_handler)
    
    # Initialize health monitor and error recovery
    health_monitor = HealthMonitor()
    error_recovery_manager = ErrorRecoveryManager()
    
    # Log startup
    logger = structlog.get_logger(__name__)
    logger.info(
        "Logging system initialized",
        log_level=log_level,
        log_file=log_file,
        json_enabled=enable_json,
        deepstream_handler_enabled=enable_deepstream_handler
    )
    
    return health_monitor, error_recovery_manager, deepstream_handler


def setup_error_recovery_strategies(error_recovery_manager: ErrorRecoveryManager):
    """Set up default error recovery strategies"""
    
    def pipeline_error_recovery(error_details: Dict, component: str) -> bool:
        """Recovery strategy for pipeline errors"""
        logger = structlog.get_logger(__name__)
        logger.info("Attempting pipeline error recovery", component=component)
        
        # Implementation would depend on specific error type
        # For now, return False to indicate manual intervention needed
        return False
    
    def stream_connection_recovery(error_details: Dict, component: str) -> bool:
        """Recovery strategy for stream connection errors"""
        logger = structlog.get_logger(__name__)
        logger.info("Attempting stream connection recovery", component=component)
        
        # Could implement reconnection logic here
        return False
    
    def model_loading_recovery(error_details: Dict, component: str) -> bool:
        """Recovery strategy for model loading errors"""
        logger = structlog.get_logger(__name__)
        logger.info("Attempting model loading recovery", component=component)
        
        # Could implement fallback model loading here
        return False
    
    def event_delivery_recovery(error_details: Dict, component: str) -> bool:
        """Recovery strategy for event delivery errors"""
        logger = structlog.get_logger(__name__)
        logger.info("Attempting event delivery recovery", component=component)
        
        # Could implement retry with backoff here
        return False
    
    # Register recovery strategies
    error_recovery_manager.register_recovery_strategy("pipeline_error", pipeline_error_recovery)
    error_recovery_manager.register_recovery_strategy("stream_connection", stream_connection_recovery)
    error_recovery_manager.register_recovery_strategy("model_loading", model_loading_recovery)
    error_recovery_manager.register_recovery_strategy("event_delivery", event_delivery_recovery)


# Global instances (initialized by setup_logging)
health_monitor: Optional[HealthMonitor] = None
error_recovery_manager: Optional[ErrorRecoveryManager] = None
deepstream_handler: Optional[DeepStreamLogHandler] = None


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = "logs/deepstream_app.log",
    enable_json: bool = False
):
    """
    Main logging setup function
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file (None to disable file logging)
        enable_json: Enable JSON formatted logging
    """
    global health_monitor, error_recovery_manager, deepstream_handler
    
    # Set up structured logging
    health_monitor, error_recovery_manager, deepstream_handler = setup_structured_logging(
        log_level=log_level,
        log_file=log_file,
        enable_json=enable_json
    )
    
    # Set up error recovery strategies
    setup_error_recovery_strategies(error_recovery_manager)
    
    return health_monitor, error_recovery_manager, deepstream_handler


if __name__ == "__main__":
    # Test logging setup
    setup_logging(log_level="DEBUG", enable_json=True)
    
    logger = structlog.get_logger(__name__)
    
    # Test structured logging
    logger.info("Test info message", component="test", value=42)
    logger.warning("Test warning message", component="test", issue="minor")
    logger.error("Test error message", component="test", error_code=500)
    
    # Test health monitoring
    if health_monitor:
        health_monitor.update_component_health("test_component", "healthy", {"test": True})
        health_monitor.increment_metric("total_frames_processed", 100)
        
        print("Health Report:")
        print(json.dumps(health_monitor.get_health_report(), indent=2))
    
    # Test error recovery
    if error_recovery_manager:
        error_recovery_manager.handle_error(
            "test_error",
            {"message": "Test error for recovery"},
            "test_component"
        )
        
        print("Error History:")
        print(json.dumps(error_recovery_manager.get_error_history(), indent=2))
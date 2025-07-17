#!/usr/bin/env python3

"""
Event Dispatcher for RTSP Object Detection System
Handles event delivery via HTTP and MQTT with retry logic
"""

import json
import time
import logging
import threading
from typing import Dict, List, Optional, Any
from queue import Queue, Empty
from dataclasses import asdict
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("WARNING: paho-mqtt not available. MQTT functionality will be disabled.")
    mqtt = None

from behavior_analyzer import BehaviorEvent


class HTTPEventDelivery:
    """Handles HTTP POST event delivery with retry logic"""
    
    def __init__(self, endpoint: str, timeout: int = 5, retry_attempts: int = 3):
        """
        Initialize HTTP event delivery
        
        Args:
            endpoint: HTTP endpoint URL
            timeout: Request timeout in seconds
            retry_attempts: Number of retry attempts
        """
        self.endpoint = endpoint
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.logger = logging.getLogger(__name__)
        
        # Set up requests session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=retry_attempts,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.logger.info(f"HTTP event delivery initialized: {endpoint}")
    
    def send_event(self, event: BehaviorEvent) -> bool:
        """
        Send event via HTTP POST
        
        Args:
            event: BehaviorEvent to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert event to JSON
            event_data = self._event_to_json(event)
            
            # Send HTTP POST request
            response = self.session.post(
                self.endpoint,
                json=event_data,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            response.raise_for_status()
            
            self.logger.debug(f"HTTP event sent successfully: {event.event_id}")
            return True
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"HTTP event delivery failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error in HTTP delivery: {e}")
            return False
    
    def _event_to_json(self, event: BehaviorEvent) -> Dict:
        """Convert BehaviorEvent to JSON-serializable dictionary"""
        return {
            'event_id': event.event_id,
            'event_type': event.event_type,
            'timestamp': event.timestamp,
            'object': {
                'tracking_id': event.tracking_id,
                'class_name': event.class_name,
                'position': event.position
            },
            'metadata': event.metadata,
            'confidence': event.confidence
        }


class MQTTEventDelivery:
    """Handles MQTT event delivery with retry logic"""
    
    def __init__(self, broker: str, topic: str, port: int = 1883, qos: int = 1,
                 username: str = None, password: str = None):
        """
        Initialize MQTT event delivery
        
        Args:
            broker: MQTT broker hostname
            topic: MQTT topic to publish to
            port: MQTT broker port
            qos: Quality of Service level
            username: Optional MQTT username
            password: Optional MQTT password
        """
        if mqtt is None:
            raise ImportError("paho-mqtt is required for MQTT functionality")
        
        self.broker = broker
        self.topic = topic
        self.port = port
        self.qos = qos
        self.username = username
        self.password = password
        self.logger = logging.getLogger(__name__)
        
        # MQTT client setup
        self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
        if username and password:
            self.client.username_pw_set(username, password)
        
        self.connected = False
        self.connect_lock = threading.Lock()
        
        self.logger.info(f"MQTT event delivery initialized: {broker}:{port}/{topic}")
    
    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            self.connected = True
            self.logger.info(f"Connected to MQTT broker: {self.broker}")
        else:
            self.connected = False
            self.logger.error(f"Failed to connect to MQTT broker: {rc}")
    
    def _on_disconnect(self, client, userdata, rc):
        """MQTT disconnection callback"""
        self.connected = False
        self.logger.warning(f"Disconnected from MQTT broker: {rc}")
    
    def _on_publish(self, client, userdata, mid):
        """MQTT publish callback"""
        self.logger.debug(f"MQTT message published: {mid}")
    
    def connect(self) -> bool:
        """Connect to MQTT broker"""
        with self.connect_lock:
            if self.connected:
                return True
            
            try:
                self.client.connect(self.broker, self.port, 60)
                self.client.loop_start()
                
                # Wait for connection
                timeout = 10
                start_time = time.time()
                while not self.connected and (time.time() - start_time) < timeout:
                    time.sleep(0.1)
                
                return self.connected
                
            except Exception as e:
                self.logger.error(f"MQTT connection error: {e}")
                return False
    
    def disconnect(self):
        """Disconnect from MQTT broker"""
        try:
            self.client.loop_stop()
            self.client.disconnect()
            self.connected = False
        except Exception as e:
            self.logger.error(f"MQTT disconnection error: {e}")
    
    def send_event(self, event: BehaviorEvent) -> bool:
        """
        Send event via MQTT
        
        Args:
            event: BehaviorEvent to send
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Ensure connection
            if not self.connected:
                if not self.connect():
                    return False
            
            # Convert event to JSON
            event_data = self._event_to_json(event)
            payload = json.dumps(event_data)
            
            # Publish message
            result = self.client.publish(self.topic, payload, qos=self.qos)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                self.logger.debug(f"MQTT event sent successfully: {event.event_id}")
                return True
            else:
                self.logger.error(f"MQTT publish failed: {result.rc}")
                return False
                
        except Exception as e:
            self.logger.error(f"MQTT event delivery failed: {e}")
            return False
    
    def _event_to_json(self, event: BehaviorEvent) -> Dict:
        """Convert BehaviorEvent to JSON-serializable dictionary"""
        return {
            'event_id': event.event_id,
            'event_type': event.event_type,
            'timestamp': event.timestamp,
            'object': {
                'tracking_id': event.tracking_id,
                'class_name': event.class_name,
                'position': event.position
            },
            'metadata': event.metadata,
            'confidence': event.confidence
        }


class EventDispatcher:
    """Main event dispatcher that manages multiple delivery methods"""
    
    def __init__(self, config: Dict):
        """
        Initialize event dispatcher
        
        Args:
            config: Configuration dictionary with delivery settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Delivery methods
        self.http_delivery = None
        self.mqtt_delivery = None
        
        # Event queue for async processing
        self.event_queue = Queue()
        self.worker_thread = None
        self.stop_worker = False
        
        # Statistics
        self.stats = {
            'events_received': 0,
            'events_sent_http': 0,
            'events_sent_mqtt': 0,
            'events_failed_http': 0,
            'events_failed_mqtt': 0
        }
        
        # Initialize delivery methods
        self._init_delivery_methods()
        
        # Start worker thread
        self._start_worker()
    
    def _init_delivery_methods(self):
        """Initialize configured delivery methods"""
        try:
            # Initialize HTTP delivery
            if 'http' in self.config and self.config['http'].get('endpoint'):
                http_config = self.config['http']
                self.http_delivery = HTTPEventDelivery(
                    endpoint=http_config['endpoint'],
                    timeout=http_config.get('timeout_seconds', 5),
                    retry_attempts=http_config.get('retry_attempts', 3)
                )
                self.logger.info("HTTP event delivery initialized")
            
            # Initialize MQTT delivery
            if 'mqtt' in self.config and self.config['mqtt'].get('broker'):
                mqtt_config = self.config['mqtt']
                self.mqtt_delivery = MQTTEventDelivery(
                    broker=mqtt_config['broker'],
                    topic=mqtt_config['topic'],
                    port=mqtt_config.get('port', 1883),
                    qos=mqtt_config.get('qos', 1),
                    username=mqtt_config.get('username'),
                    password=mqtt_config.get('password')
                )
                self.logger.info("MQTT event delivery initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize delivery methods: {e}")
    
    def _start_worker(self):
        """Start background worker thread for event processing"""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info("Event dispatcher worker thread started")
    
    def _worker_loop(self):
        """Background worker loop for processing events"""
        while not self.stop_worker:
            try:
                # Get event from queue (with timeout)
                event = self.event_queue.get(timeout=1.0)
                
                # Process the event
                self._process_event(event)
                
                # Mark task as done
                self.event_queue.task_done()
                
            except Empty:
                # Timeout - continue loop
                continue
            except Exception as e:
                self.logger.error(f"Error in worker loop: {e}")
    
    def _process_event(self, event: BehaviorEvent):
        """Process a single event through all delivery methods"""
        try:
            # Send via HTTP if configured
            if self.http_delivery:
                if self.http_delivery.send_event(event):
                    self.stats['events_sent_http'] += 1
                else:
                    self.stats['events_failed_http'] += 1
            
            # Send via MQTT if configured
            if self.mqtt_delivery:
                if self.mqtt_delivery.send_event(event):
                    self.stats['events_sent_mqtt'] += 1
                else:
                    self.stats['events_failed_mqtt'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error processing event {event.event_id}: {e}")
    
    def dispatch_event(self, event: BehaviorEvent):
        """
        Dispatch an event for delivery
        
        Args:
            event: BehaviorEvent to dispatch
        """
        try:
            self.stats['events_received'] += 1
            self.event_queue.put(event)
            self.logger.debug(f"Event queued for dispatch: {event.event_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to queue event: {e}")
    
    def get_statistics(self) -> Dict:
        """Get event delivery statistics"""
        return {
            **self.stats,
            'queue_size': self.event_queue.qsize(),
            'http_enabled': self.http_delivery is not None,
            'mqtt_enabled': self.mqtt_delivery is not None
        }
    
    def shutdown(self):
        """Shutdown event dispatcher"""
        try:
            self.logger.info("Shutting down event dispatcher...")
            
            # Stop worker thread
            self.stop_worker = True
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=5)
            
            # Wait for queue to empty
            self.event_queue.join()
            
            # Disconnect MQTT if connected
            if self.mqtt_delivery:
                self.mqtt_delivery.disconnect()
            
            self.logger.info("Event dispatcher shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {
        'http': {
            'endpoint': 'http://localhost:8080/events',
            'timeout_seconds': 5,
            'retry_attempts': 3
        },
        'mqtt': {
            'broker': 'localhost',
            'topic': 'detection/events',
            'port': 1883,
            'qos': 1
        }
    }
    
    # Create event dispatcher
    dispatcher = EventDispatcher(config)
    
    # Create test event
    from behavior_analyzer import BehaviorEvent
    import uuid
    from datetime import datetime
    
    test_event = BehaviorEvent(
        event_id=str(uuid.uuid4()),
        event_type="object_appeared",
        timestamp=datetime.now().isoformat(),
        tracking_id=123,
        class_name="person",
        position={'x': 100.0, 'y': 200.0},
        metadata={'test': True},
        confidence=0.95
    )
    
    # Dispatch test event
    dispatcher.dispatch_event(test_event)
    
    # Wait a bit and check stats
    time.sleep(2)
    stats = dispatcher.get_statistics()
    print(f"Event dispatcher statistics: {stats}")
    
    # Shutdown
    dispatcher.shutdown()
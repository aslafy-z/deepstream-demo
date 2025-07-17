#!/usr/bin/env python3

"""
Behavior Analyzer for Object Detection System
Analyzes object behaviors and generates events based on tracking data
"""

import time
import logging
import uuid
from typing import Dict, List, Optional, Callable, Set
from dataclasses import dataclass
from datetime import datetime

try:
    import pyds
except ImportError:
    print("ERROR: DeepStream Python bindings (pyds) not found.")


@dataclass
class BehaviorEvent:
    """Represents a behavior event"""
    event_id: str
    event_type: str  # object_appeared, object_static, object_moving
    timestamp: str
    tracking_id: int
    class_name: str
    position: Dict[str, float]
    metadata: Dict
    confidence: float = 1.0


class BehaviorAnalyzer:
    """Analyzes object behaviors and generates events"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize behavior analyzer
        
        Args:
            config: Configuration dictionary with behavior parameters
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration with defaults
        default_config = {
            'static_threshold_seconds': 30,
            'position_tolerance_pixels': 10,
            'debounce_seconds': 2,
            'min_confidence': 0.5
        }
        
        self.config = {**default_config, **(config or {})}
        
        # State tracking
        self.tracked_objects: Dict[int, Dict] = {}  # tracking_id -> object_state
        self.recent_events: List[BehaviorEvent] = []
        self.event_callbacks: List[Callable] = []
        
        # Debouncing
        self.last_appearance_events: Dict[int, float] = {}  # tracking_id -> timestamp
        self.last_static_events: Dict[int, float] = {}
        self.last_moving_events: Dict[int, float] = {}
        
        self.logger.info(f"Behavior analyzer initialized with config: {self.config}")
    
    def add_event_callback(self, callback: Callable[[BehaviorEvent], None]):
        """Add callback function to be called when events are generated"""
        self.event_callbacks.append(callback)
    
    def analyze_frame(self, tracking_manager, frame_meta):
        """
        Analyze a frame for behavioral events
        
        Args:
            tracking_manager: TrackingManager instance
            frame_meta: DeepStream frame metadata
        """
        try:
            current_time = time.time()
            current_tracks = tracking_manager.get_active_tracks()
            
            # Check for new object appearances
            self._check_object_appearances(current_tracks, current_time)
            
            # Check for static objects
            self._check_static_objects(tracking_manager, current_time)
            
            # Check for moving objects (previously static)
            self._check_moving_objects(tracking_manager, current_time)
            
            # Update tracked objects state
            self._update_object_states(current_tracks, current_time)
            
            # Clean up old events
            self._cleanup_old_events()
            
        except Exception as e:
            self.logger.error(f"Error analyzing frame: {e}")
    
    def _check_object_appearances(self, current_tracks: Dict, current_time: float):
        """Check for newly appeared objects"""
        try:
            for tracking_id, track_info in current_tracks.items():
                # Skip if confidence is too low
                if track_info.get('confidence', 0) < self.config['min_confidence']:
                    continue
                
                # Check if this is a new object
                if tracking_id not in self.tracked_objects:
                    # Check debouncing
                    last_event_time = self.last_appearance_events.get(tracking_id, 0)
                    if current_time - last_event_time < self.config['debounce_seconds']:
                        continue
                    
                    # Generate appearance event
                    event = self._create_appearance_event(tracking_id, track_info, current_time)
                    self._emit_event(event)
                    
                    # Update debounce tracking
                    self.last_appearance_events[tracking_id] = current_time
                    
        except Exception as e:
            self.logger.error(f"Error checking object appearances: {e}")
    
    def _check_static_objects(self, tracking_manager, current_time: float):
        """Check for objects that have become static"""
        try:
            for tracking_id in tracking_manager.get_active_tracks().keys():
                track_info = tracking_manager.get_track_info(tracking_id)
                if not track_info or track_info.get('confidence', 0) < self.config['min_confidence']:
                    continue
                
                # Check if object is static
                is_static = tracking_manager.is_track_static(
                    tracking_id,
                    threshold_pixels=self.config['position_tolerance_pixels'],
                    min_frames=30  # Approximate frames for static_threshold_seconds
                )
                
                if is_static:
                    # Check debouncing
                    last_event_time = self.last_static_events.get(tracking_id, 0)
                    if current_time - last_event_time < self.config['static_threshold_seconds']:
                        continue
                    
                    # Get duration
                    duration = tracking_manager.get_track_duration(tracking_id)
                    if duration and duration >= self.config['static_threshold_seconds']:
                        # Generate static event
                        event = self._create_static_event(tracking_id, track_info, current_time, duration)
                        self._emit_event(event)
                        
                        # Update debounce tracking
                        self.last_static_events[tracking_id] = current_time
                        
        except Exception as e:
            self.logger.error(f"Error checking static objects: {e}")
    
    def _check_moving_objects(self, tracking_manager, current_time: float):
        """Check for objects that have started moving (were previously static)"""
        try:
            for tracking_id in tracking_manager.get_active_tracks().keys():
                track_info = tracking_manager.get_track_info(tracking_id)
                if not track_info or track_info.get('confidence', 0) < self.config['min_confidence']:
                    continue
                
                # Check if object was previously marked as static
                if tracking_id in self.last_static_events:
                    # Check if object is no longer static
                    is_static = tracking_manager.is_track_static(
                        tracking_id,
                        threshold_pixels=self.config['position_tolerance_pixels'],
                        min_frames=10  # Shorter check for movement detection
                    )
                    
                    if not is_static:
                        # Check debouncing
                        last_event_time = self.last_moving_events.get(tracking_id, 0)
                        if current_time - last_event_time < self.config['debounce_seconds']:
                            continue
                        
                        # Generate moving event
                        event = self._create_moving_event(tracking_id, track_info, current_time)
                        self._emit_event(event)
                        
                        # Update debounce tracking
                        self.last_moving_events[tracking_id] = current_time
                        
                        # Remove from static events (object is moving again)
                        del self.last_static_events[tracking_id]
                        
        except Exception as e:
            self.logger.error(f"Error checking moving objects: {e}")
    
    def _create_appearance_event(self, tracking_id: int, track_info: Dict, timestamp: float) -> BehaviorEvent:
        """Create an object appearance event"""
        return BehaviorEvent(
            event_id=str(uuid.uuid4()),
            event_type="object_appeared",
            timestamp=datetime.fromtimestamp(timestamp).isoformat(),
            tracking_id=tracking_id,
            class_name=self._get_class_name(track_info.get('class_id', 0)),
            position={
                'x': track_info['center']['x'],
                'y': track_info['center']['y']
            },
            metadata={
                'bbox': track_info['bbox'],
                'frame_number': track_info.get('frame_number', 0)
            },
            confidence=track_info.get('confidence', 1.0)
        )
    
    def _create_static_event(self, tracking_id: int, track_info: Dict, timestamp: float, duration: float) -> BehaviorEvent:
        """Create a static object event"""
        return BehaviorEvent(
            event_id=str(uuid.uuid4()),
            event_type="object_static",
            timestamp=datetime.fromtimestamp(timestamp).isoformat(),
            tracking_id=tracking_id,
            class_name=self._get_class_name(track_info.get('class_id', 0)),
            position={
                'x': track_info['center']['x'],
                'y': track_info['center']['y']
            },
            metadata={
                'duration': duration,
                'bbox': track_info['bbox'],
                'frame_number': track_info.get('frame_number', 0)
            },
            confidence=track_info.get('confidence', 1.0)
        )
    
    def _create_moving_event(self, tracking_id: int, track_info: Dict, timestamp: float) -> BehaviorEvent:
        """Create a moving object event"""
        return BehaviorEvent(
            event_id=str(uuid.uuid4()),
            event_type="object_moving",
            timestamp=datetime.fromtimestamp(timestamp).isoformat(),
            tracking_id=tracking_id,
            class_name=self._get_class_name(track_info.get('class_id', 0)),
            position={
                'x': track_info['center']['x'],
                'y': track_info['center']['y']
            },
            metadata={
                'bbox': track_info['bbox'],
                'frame_number': track_info.get('frame_number', 0)
            },
            confidence=track_info.get('confidence', 1.0)
        )
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID using COCO dataset labels"""
        # COCO dataset class names (commonly used with YOLO and other detection models)
        coco_class_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
            5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
            10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
            14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
            20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
            25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
            30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
            35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
            39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
            44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
            49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
            54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
            59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
            64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
            69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
            74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
            79: "toothbrush"
        }
        
        return coco_class_names.get(class_id, f"unknown_class_{class_id}")
    
    def _emit_event(self, event: BehaviorEvent):
        """Emit an event to all registered callbacks"""
        try:
            self.recent_events.append(event)
            self.logger.info(f"Generated event: {event.event_type} for track {event.tracking_id}")
            
            # Call all registered callbacks
            for callback in self.event_callbacks:
                try:
                    callback(event)
                except Exception as e:
                    self.logger.error(f"Error in event callback: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error emitting event: {e}")
    
    def _update_object_states(self, current_tracks: Dict, current_time: float):
        """Update internal object state tracking"""
        try:
            # Update existing objects
            for tracking_id, track_info in current_tracks.items():
                self.tracked_objects[tracking_id] = {
                    **track_info,
                    'last_seen': current_time
                }
            
            # Remove objects that are no longer active
            inactive_objects = []
            for tracking_id, obj_state in self.tracked_objects.items():
                if tracking_id not in current_tracks:
                    # Keep for a short time in case of temporary tracking loss
                    if current_time - obj_state['last_seen'] > 10:  # 10 seconds
                        inactive_objects.append(tracking_id)
            
            for tracking_id in inactive_objects:
                del self.tracked_objects[tracking_id]
                # Also clean up debounce tracking
                self.last_appearance_events.pop(tracking_id, None)
                self.last_static_events.pop(tracking_id, None)
                self.last_moving_events.pop(tracking_id, None)
                
        except Exception as e:
            self.logger.error(f"Error updating object states: {e}")
    
    def _cleanup_old_events(self, max_events: int = 1000):
        """Clean up old events to prevent memory growth"""
        if len(self.recent_events) > max_events:
            self.recent_events = self.recent_events[-max_events:]
    
    def get_recent_events(self, limit: int = 100) -> List[BehaviorEvent]:
        """Get recent events"""
        return self.recent_events[-limit:]
    
    def get_statistics(self) -> Dict:
        """Get behavior analysis statistics"""
        return {
            'tracked_objects': len(self.tracked_objects),
            'recent_events': len(self.recent_events),
            'appearance_events_tracked': len(self.last_appearance_events),
            'static_events_tracked': len(self.last_static_events),
            'moving_events_tracked': len(self.last_moving_events),
            'config': self.config
        }


# Probe function for behavior analysis integration
def behavior_analysis_probe(pad, info, user_data):
    """
    Probe function to perform behavior analysis on DeepStream pipeline
    
    Args:
        pad: GStreamer pad
        info: Probe info
        user_data: Tuple of (behavior_analyzer, tracking_manager)
    """
    try:
        behavior_analyzer, tracking_manager = user_data
        
        if not behavior_analyzer or not tracking_manager:
            return Gst.PadProbeReturn.OK
        
        # Get buffer
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK
        
        # Get batch metadata
        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK
        
        # Process each frame in the batch
        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                if frame_meta:
                    # Perform behavior analysis
                    behavior_analyzer.analyze_frame(tracking_manager, frame_meta)
                
            except StopIteration:
                break
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        return Gst.PadProbeReturn.OK
        
    except Exception as e:
        print(f"Error in behavior analysis probe: {e}")
        return Gst.PadProbeReturn.OK


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def event_handler(event: BehaviorEvent):
        print(f"Event: {event.event_type} - Track {event.tracking_id} - {event.class_name}")
    
    # Create behavior analyzer
    config = {
        'static_threshold_seconds': 30,
        'position_tolerance_pixels': 10,
        'debounce_seconds': 2
    }
    
    analyzer = BehaviorAnalyzer(config)
    analyzer.add_event_callback(event_handler)
    
    # Get statistics
    stats = analyzer.get_statistics()
    print(f"Behavior analyzer statistics: {stats}")
#!/usr/bin/env python3

"""
Tracking Manager for DeepStream nvtracker
Handles object tracking configuration and tracking ID management
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import time

try:
    import pyds
except ImportError:
    print("ERROR: DeepStream Python bindings (pyds) not found.")


class TrackingManager:
    """Manages object tracking using DeepStream nvtracker"""
    
    def __init__(self, tracker_config_file: str = "configs/tracker_config.yml"):
        """
        Initialize tracking manager
        
        Args:
            tracker_config_file: Path to tracker configuration file
        """
        self.tracker_config_file = Path(tracker_config_file)
        self.logger = logging.getLogger(__name__)
        
        # Tracking state
        self.active_tracks: Dict[int, Dict] = {}  # tracking_id -> track_info
        self.track_history: Dict[int, List] = {}  # tracking_id -> position_history
        self.lost_tracks: Set[int] = set()
        self.track_id_counter = 0
        
        # Configuration
        self.max_track_history = 100  # Maximum positions to keep in history
        self.track_timeout_seconds = 30  # Time before considering track lost
        
        # Validate tracker config
        self._validate_tracker_config()
    
    def _validate_tracker_config(self):
        """Validate tracker configuration file exists"""
        if not self.tracker_config_file.exists():
            self.logger.error(f"Tracker config file not found: {self.tracker_config_file}")
            raise FileNotFoundError(f"Tracker config not found: {self.tracker_config_file}")
        
        self.logger.info(f"Tracker config validated: {self.tracker_config_file}")
    
    def update_tracks(self, frame_meta):
        """
        Update tracking information from DeepStream frame metadata
        
        Args:
            frame_meta: DeepStream frame metadata containing object information
        """
        try:
            current_time = time.time()
            current_track_ids = set()
            
            # Process each object in the frame
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    
                    if obj_meta:
                        tracking_id = obj_meta.object_id
                        current_track_ids.add(tracking_id)
                        
                        # Extract object information
                        track_info = {
                            'tracking_id': tracking_id,
                            'class_id': obj_meta.class_id,
                            'confidence': obj_meta.confidence,
                            'bbox': {
                                'left': obj_meta.rect_params.left,
                                'top': obj_meta.rect_params.top,
                                'width': obj_meta.rect_params.width,
                                'height': obj_meta.rect_params.height
                            },
                            'center': {
                                'x': obj_meta.rect_params.left + obj_meta.rect_params.width / 2,
                                'y': obj_meta.rect_params.top + obj_meta.rect_params.height / 2
                            },
                            'timestamp': current_time,
                            'frame_number': frame_meta.frame_num
                        }
                        
                        # Update active tracks
                        self._update_track(tracking_id, track_info)
                        
                except Exception as e:
                    self.logger.error(f"Error processing object metadata: {e}")
                
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            # Check for lost tracks
            self._check_lost_tracks(current_track_ids, current_time)
            
        except Exception as e:
            self.logger.error(f"Error updating tracks: {e}")
    
    def _update_track(self, tracking_id: int, track_info: Dict):
        """Update information for a specific track"""
        try:
            # Check if this is a new track
            if tracking_id not in self.active_tracks:
                self.logger.info(f"New track detected: ID {tracking_id}, class {track_info['class_id']}")
                self.active_tracks[tracking_id] = track_info
                self.track_history[tracking_id] = []
                
                # Remove from lost tracks if it was there
                self.lost_tracks.discard(tracking_id)
            else:
                # Update existing track
                self.active_tracks[tracking_id].update(track_info)
            
            # Add position to history
            position = {
                'x': track_info['center']['x'],
                'y': track_info['center']['y'],
                'timestamp': track_info['timestamp'],
                'frame_number': track_info['frame_number']
            }
            
            self.track_history[tracking_id].append(position)
            
            # Limit history size
            if len(self.track_history[tracking_id]) > self.max_track_history:
                self.track_history[tracking_id] = self.track_history[tracking_id][-self.max_track_history:]
            
        except Exception as e:
            self.logger.error(f"Error updating track {tracking_id}: {e}")
    
    def _check_lost_tracks(self, current_track_ids: Set[int], current_time: float):
        """Check for tracks that are no longer present"""
        try:
            # Find tracks that are no longer active
            lost_track_ids = set(self.active_tracks.keys()) - current_track_ids
            
            for tracking_id in lost_track_ids:
                track_info = self.active_tracks[tracking_id]
                time_since_last_seen = current_time - track_info['timestamp']
                
                if time_since_last_seen > self.track_timeout_seconds:
                    self.logger.info(f"Track lost: ID {tracking_id} (timeout after {time_since_last_seen:.1f}s)")
                    
                    # Move to lost tracks
                    self.lost_tracks.add(tracking_id)
                    del self.active_tracks[tracking_id]
                    
                    # Keep history for a while longer
                    # Could implement cleanup of old history here
            
        except Exception as e:
            self.logger.error(f"Error checking lost tracks: {e}")
    
    def get_track_info(self, tracking_id: int) -> Optional[Dict]:
        """Get information about a specific track"""
        return self.active_tracks.get(tracking_id)
    
    def get_active_tracks(self) -> Dict[int, Dict]:
        """Get all currently active tracks"""
        return self.active_tracks.copy()
    
    def get_track_history(self, tracking_id: int) -> List[Dict]:
        """Get position history for a specific track"""
        return self.track_history.get(tracking_id, [])
    
    def is_track_new(self, tracking_id: int, max_age_frames: int = 5) -> bool:
        """Check if a track is newly appeared (within last N frames)"""
        if tracking_id not in self.track_history:
            return False
        
        history = self.track_history[tracking_id]
        return len(history) <= max_age_frames
    
    def is_track_static(self, tracking_id: int, threshold_pixels: float = 10.0, 
                       min_frames: int = 30) -> bool:
        """
        Check if a track has been static (not moving much)
        
        Args:
            tracking_id: Track ID to check
            threshold_pixels: Maximum movement to consider static
            min_frames: Minimum frames to consider for static detection
            
        Returns:
            bool: True if track is considered static
        """
        if tracking_id not in self.track_history:
            return False
        
        history = self.track_history[tracking_id]
        
        if len(history) < min_frames:
            return False
        
        # Check movement over the last min_frames positions
        recent_history = history[-min_frames:]
        
        if not recent_history:
            return False
        
        # Calculate maximum distance from first position in recent history
        start_pos = recent_history[0]
        max_distance = 0
        
        for pos in recent_history[1:]:
            distance = ((pos['x'] - start_pos['x']) ** 2 + 
                       (pos['y'] - start_pos['y']) ** 2) ** 0.5
            max_distance = max(max_distance, distance)
        
        return max_distance <= threshold_pixels
    
    def get_track_duration(self, tracking_id: int) -> Optional[float]:
        """Get how long a track has been active (in seconds)"""
        if tracking_id not in self.track_history:
            return None
        
        history = self.track_history[tracking_id]
        if len(history) < 2:
            return 0.0
        
        return history[-1]['timestamp'] - history[0]['timestamp']
    
    def get_tracking_statistics(self) -> Dict:
        """Get overall tracking statistics"""
        return {
            'active_tracks': len(self.active_tracks),
            'lost_tracks': len(self.lost_tracks),
            'total_tracks_seen': len(self.track_history),
            'config_file': str(self.tracker_config_file)
        }
    
    def cleanup_old_history(self, max_age_seconds: float = 300):
        """Clean up old tracking history to prevent memory growth"""
        try:
            current_time = time.time()
            tracks_to_remove = []
            
            for tracking_id, history in self.track_history.items():
                if not history:
                    continue
                
                # If track is not active and history is old, remove it
                if (tracking_id not in self.active_tracks and 
                    current_time - history[-1]['timestamp'] > max_age_seconds):
                    tracks_to_remove.append(tracking_id)
            
            for tracking_id in tracks_to_remove:
                del self.track_history[tracking_id]
                self.lost_tracks.discard(tracking_id)
                self.logger.debug(f"Cleaned up old history for track {tracking_id}")
            
            if tracks_to_remove:
                self.logger.info(f"Cleaned up history for {len(tracks_to_remove)} old tracks")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up tracking history: {e}")


# Probe function for tracking integration
def tracking_src_pad_buffer_probe(pad, info, user_data):
    """
    Probe function to extract tracking information from DeepStream pipeline
    
    Args:
        pad: GStreamer pad
        info: Probe info
        user_data: User data (should contain TrackingManager instance)
    """
    try:
        tracking_manager = user_data
        
        if not tracking_manager:
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
                    # Update tracking information
                    tracking_manager.update_tracks(frame_meta)
                
            except StopIteration:
                break
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        return Gst.PadProbeReturn.OK
        
    except Exception as e:
        print(f"Error in tracking probe: {e}")
        return Gst.PadProbeReturn.OK


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create tracking manager
    tracker = TrackingManager()
    
    # Get statistics
    stats = tracker.get_tracking_statistics()
    print(f"Tracking statistics: {stats}")
    
    # Test static detection (with dummy data)
    print("Tracking manager initialized successfully")
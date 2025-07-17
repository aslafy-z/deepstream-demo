#!/usr/bin/env python3

"""
Frame Saver for RTSP Object Detection System
Handles detection-only frame saving with timestamp preservation
"""

import os
import cv2
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
from queue import Queue, Empty
import numpy as np

try:
    import pyds
except ImportError:
    print("ERROR: DeepStream Python bindings (pyds) not found.")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    print("WARNING: PIL not available. Some frame saving features may be limited.")
    PIL_AVAILABLE = False


class FrameSaver:
    """Manages detection-only frame saving with timestamp preservation"""
    
    def __init__(self, config: Dict):
        """
        Initialize frame saver
        
        Args:
            config: Configuration dictionary with frame saving settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.detection_only_mode = config.get('detection_only_mode', False)
        self.output_path = Path(config.get('output_path', '/data/frames'))
        self.preserve_timestamps = config.get('preserve_timestamps', True)
        self.max_frames_per_hour = config.get('max_frames_per_hour', 3600)  # Rate limiting
        self.image_format = config.get('image_format', 'jpg')
        self.image_quality = config.get('image_quality', 95)
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Frame processing queue
        self.frame_queue = Queue(maxsize=100)
        self.worker_thread = None
        self.stop_worker = False
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'frames_saved': 0,
            'frames_skipped_no_detection': 0,
            'frames_skipped_rate_limit': 0,
            'total_detections_saved': 0,
            'disk_space_used_mb': 0,
            'last_save_time': None
        }
        
        # Rate limiting
        self.last_save_time = 0
        self.saves_this_hour = 0
        self.hour_start = time.time()
        
        # Start worker thread
        self._start_worker()
        
        self.logger.info(f"Frame saver initialized - Output: {self.output_path}, Detection-only: {self.detection_only_mode}")
    
    def _start_worker(self):
        """Start background worker thread for frame processing"""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info("Frame saver worker thread started")
    
    def _worker_loop(self):
        """Background worker loop for processing frames"""
        while not self.stop_worker:
            try:
                # Get frame from queue (with timeout)
                frame_data = self.frame_queue.get(timeout=1.0)
                
                # Process the frame
                self._process_frame(frame_data)
                
                # Mark task as done
                self.frame_queue.task_done()
                
            except Empty:
                # Timeout - continue loop
                continue
            except Exception as e:
                self.logger.error(f"Error in frame saver worker loop: {e}")
    
    def process_frame_metadata(self, frame_meta, frame_image: Optional[np.ndarray] = None):
        """
        Process frame metadata and queue for saving if needed
        
        Args:
            frame_meta: DeepStream frame metadata
            frame_image: Optional frame image data
        """
        try:
            self.stats['frames_processed'] += 1
            
            # Check if frame has detections
            has_detections, detection_count = self._check_frame_detections(frame_meta)
            
            # Decide whether to save frame
            should_save = self._should_save_frame(has_detections)
            
            if should_save:
                # Create frame data for processing
                frame_data = {
                    'frame_meta': frame_meta,
                    'frame_image': frame_image,
                    'timestamp': time.time(),
                    'detection_count': detection_count,
                    'frame_number': frame_meta.frame_num if frame_meta else 0
                }
                
                # Queue frame for processing
                try:
                    self.frame_queue.put_nowait(frame_data)
                except:
                    # Queue is full - skip this frame
                    self.logger.warning("Frame queue full - skipping frame")
            else:
                if not has_detections:
                    self.stats['frames_skipped_no_detection'] += 1
                else:
                    self.stats['frames_skipped_rate_limit'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error processing frame metadata: {e}")
    
    def _check_frame_detections(self, frame_meta) -> tuple[bool, int]:
        """
        Check if frame has detections above confidence threshold
        
        Args:
            frame_meta: DeepStream frame metadata
            
        Returns:
            Tuple of (has_detections, detection_count)
        """
        try:
            detection_count = 0
            min_confidence = self.config.get('min_confidence', 0.5)
            
            if not frame_meta:
                return False, 0
            
            # Process each object in the frame
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    
                    if obj_meta and obj_meta.confidence >= min_confidence:
                        detection_count += 1
                
                except Exception as e:
                    self.logger.error(f"Error processing object metadata: {e}")
                
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            return detection_count > 0, detection_count
            
        except Exception as e:
            self.logger.error(f"Error checking frame detections: {e}")
            return False, 0
    
    def _should_save_frame(self, has_detections: bool) -> bool:
        """
        Determine if frame should be saved
        
        Args:
            has_detections: Whether frame has detections
            
        Returns:
            bool: True if frame should be saved
        """
        # Check detection-only mode
        if self.detection_only_mode and not has_detections:
            return False
        
        # Check rate limiting
        current_time = time.time()
        
        # Reset hourly counter if needed
        if current_time - self.hour_start > 3600:
            self.saves_this_hour = 0
            self.hour_start = current_time
        
        # Check if we've exceeded hourly limit
        if self.saves_this_hour >= self.max_frames_per_hour:
            return False
        
        return True
    
    def _process_frame(self, frame_data: Dict):
        """
        Process and save a frame
        
        Args:
            frame_data: Dictionary containing frame information
        """
        try:
            timestamp = frame_data['timestamp']
            frame_number = frame_data['frame_number']
            detection_count = frame_data['detection_count']
            
            # Generate filename with timestamp
            if self.preserve_timestamps:
                dt = datetime.fromtimestamp(timestamp)
                filename = f"frame_{dt.strftime('%Y%m%d_%H%M%S_%f')[:-3]}_f{frame_number}_d{detection_count}.{self.image_format}"
            else:
                filename = f"frame_{frame_number}_d{detection_count}.{self.image_format}"
            
            filepath = self.output_path / filename
            
            # Save frame image
            if frame_data.get('frame_image') is not None:
                self._save_frame_image(frame_data['frame_image'], filepath)
            else:
                # Create metadata image with frame information
                self._create_metadata_image(frame_data, filepath)
            
            # Update statistics
            self.stats['frames_saved'] += 1
            self.stats['total_detections_saved'] += detection_count
            self.stats['last_save_time'] = timestamp
            self.saves_this_hour += 1
            
            # Update disk usage
            self._update_disk_usage()
            
            self.logger.debug(f"Frame saved: {filename} ({detection_count} detections)")
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
    
    def _save_frame_image(self, frame_image: np.ndarray, filepath: Path):
        """
        Save frame image to disk
        
        Args:
            frame_image: Frame image as numpy array
            filepath: Output file path
        """
        try:
            if self.image_format.lower() in ['jpg', 'jpeg']:
                cv2.imwrite(
                    str(filepath), 
                    frame_image, 
                    [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
                )
            elif self.image_format.lower() == 'png':
                cv2.imwrite(str(filepath), frame_image)
            else:
                # Default to JPEG
                cv2.imwrite(
                    str(filepath.with_suffix('.jpg')), 
                    frame_image, 
                    [cv2.IMWRITE_JPEG_QUALITY, self.image_quality]
                )
                
        except Exception as e:
            self.logger.error(f"Error saving frame image: {e}")
    
    def _create_metadata_image(self, frame_data: Dict, filepath: Path):
        """
        Create placeholder image with metadata when frame image is not available
        
        Args:
            frame_data: Frame data dictionary
            filepath: Output file path
        """
        try:
            if not PIL_AVAILABLE:
                self.logger.warning("PIL not available - cannot create metadata image")
                return
            
            # Create blank image
            img = Image.new('RGB', (640, 480), color='black')
            draw = ImageDraw.Draw(img)
            
            # Try to load font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
                small_font = ImageFont.truetype("arial.ttf", 16)
            except:
                font = ImageFont.load_default()
                small_font = font
            
            # Add metadata text
            timestamp = datetime.fromtimestamp(frame_data['timestamp'])
            text_lines = [
                f"Frame: {frame_data['frame_number']}",
                f"Time: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                f"Detections: {frame_data['detection_count']}",
                f"Detection-only mode: {self.detection_only_mode}"
            ]
            
            y_offset = 50
            for line in text_lines:
                draw.text((20, y_offset), line, fill='white', font=font)
                y_offset += 30
            
            # Save image
            img.save(filepath, quality=self.image_quality)
            
        except Exception as e:
            self.logger.error(f"Error creating metadata image: {e}")
    
    def _update_disk_usage(self):
        """Update disk usage statistics"""
        try:
            total_size = 0
            for file_path in self.output_path.glob(f"*.{self.image_format}"):
                total_size += file_path.stat().st_size
            
            self.stats['disk_space_used_mb'] = total_size / (1024 * 1024)
            
        except Exception as e:
            self.logger.error(f"Error calculating disk usage: {e}")
    
    def cleanup_old_frames(self, max_age_hours: int = 24):
        """
        Clean up old frame files
        
        Args:
            max_age_hours: Maximum age of frames to keep
        """
        try:
            current_time = time.time()
            cutoff_time = current_time - (max_age_hours * 3600)
            
            deleted_count = 0
            for file_path in self.output_path.glob(f"*.{self.image_format}"):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
            
            if deleted_count > 0:
                self.logger.info(f"Cleaned up {deleted_count} old frame files")
                self._update_disk_usage()
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old frames: {e}")
    
    def get_statistics(self) -> Dict:
        """Get frame saving statistics"""
        return {
            **self.stats,
            'detection_only_mode': self.detection_only_mode,
            'output_path': str(self.output_path),
            'queue_size': self.frame_queue.qsize(),
            'saves_this_hour': self.saves_this_hour,
            'max_frames_per_hour': self.max_frames_per_hour,
            'image_format': self.image_format,
            'preserve_timestamps': self.preserve_timestamps
        }
    
    def update_config(self, new_config: Dict):
        """Update configuration at runtime"""
        try:
            # Update detection-only mode
            if 'detection_only_mode' in new_config:
                self.detection_only_mode = new_config['detection_only_mode']
                self.logger.info(f"Detection-only mode updated: {self.detection_only_mode}")
            
            # Update other settings
            self.config.update(new_config)
            
        except Exception as e:
            self.logger.error(f"Error updating frame saver config: {e}")
    
    def shutdown(self):
        """Shutdown frame saver"""
        try:
            self.logger.info("Shutting down frame saver...")
            
            # Stop worker thread
            self.stop_worker = True
            if self.worker_thread and self.worker_thread.is_alive():
                self.worker_thread.join(timeout=5)
            
            # Wait for queue to empty
            self.frame_queue.join()
            
            self.logger.info("Frame saver shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during frame saver shutdown: {e}")


# Probe function for frame saving integration
def frame_saving_probe(pad, info, user_data):
    """
    Probe function to save frames based on detection presence
    
    Args:
        pad: GStreamer pad
        info: Probe info
        user_data: FrameSaver instance
    """
    try:
        frame_saver = user_data
        
        if not frame_saver:
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
                    # Process frame for saving
                    frame_saver.process_frame_metadata(frame_meta)
                
            except StopIteration:
                break
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        return Gst.PadProbeReturn.OK
        
    except Exception as e:
        print(f"Error in frame saving probe: {e}")
        return Gst.PadProbeReturn.OK


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {
        'detection_only_mode': True,
        'output_path': '/tmp/test_frames',
        'preserve_timestamps': True,
        'max_frames_per_hour': 100,
        'image_format': 'jpg',
        'image_quality': 95,
        'min_confidence': 0.5
    }
    
    # Create frame saver
    saver = FrameSaver(config)
    
    # Get statistics
    stats = saver.get_statistics()
    print(f"Frame saver statistics: {stats}")
    
    # Test cleanup
    saver.cleanup_old_frames(max_age_hours=1)
    
    # Shutdown
    saver.shutdown()
#!/usr/bin/env python3

"""
Video Output Manager for RTSP Object Detection System
Handles video output with annotations, RTSP streaming, and conditional display
"""

import logging
from typing import Dict, List, Optional, Tuple
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

try:
    import pyds
except ImportError:
    print("ERROR: DeepStream Python bindings (pyds) not found.")


class VideoOutputManager:
    """Manages video output with annotations and streaming"""
    
    def __init__(self, config: Dict = None):
        """
        Initialize video output manager
        
        Args:
            config: Configuration dictionary for video output settings
        """
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        default_config = {
            'rtsp': {
                'enabled': True,
                'port': 8554,
                'formats': ['rtsp']
            },
            'annotations': {
                'show_bounding_boxes': True,
                'show_tracking_ids': True,
                'show_class_labels': True,
                'show_confidence': True,
                'colors': {
                    'person': (1.0, 0.0, 0.0, 1.0),      # Red
                    'vehicle': (0.0, 1.0, 0.0, 1.0),     # Green
                    'car': (0.0, 1.0, 0.0, 1.0),         # Green
                    'truck': (0.0, 0.0, 1.0, 1.0),       # Blue
                    'bus': (1.0, 1.0, 0.0, 1.0),         # Yellow
                    'bicycle': (1.0, 0.0, 1.0, 1.0),     # Magenta
                    'motorbike': (0.0, 1.0, 1.0, 1.0),   # Cyan
                    'default': (1.0, 1.0, 1.0, 1.0)      # White
                },
                'border_width': 3,
                'text_size': 16,
                'text_color': (1.0, 1.0, 1.0, 1.0),     # White
                'text_bg_color': (0.3, 0.3, 0.3, 1.0)   # Dark gray
            },
            'conditional_display': {
                'detection_only': False,  # Only show frames with detections
                'min_confidence': 0.5     # Minimum confidence to show annotations
            }
        }
        
        self.config = {**default_config, **(config or {})}
        
        # State
        self.has_detections = False
        self.frame_count = 0
        self.annotation_stats = {
            'frames_processed': 0,
            'frames_with_detections': 0,
            'objects_annotated': 0
        }
        
        self.logger.info("Video output manager initialized")
    
    def configure_osd_colors(self, class_labels: List[str]) -> Dict:
        """
        Configure OSD colors based on class labels
        
        Args:
            class_labels: List of class label names
            
        Returns:
            Dictionary mapping class IDs to colors
        """
        color_mapping = {}
        
        for class_id, class_name in enumerate(class_labels):
            # Get color for this class or use default
            if class_name in self.config['annotations']['colors']:
                color = self.config['annotations']['colors'][class_name]
            else:
                color = self.config['annotations']['colors']['default']
            
            color_mapping[class_id] = color
            
        self.logger.info(f"Configured colors for {len(color_mapping)} classes")
        return color_mapping
    
    def should_display_frame(self, has_detections: bool) -> bool:
        """
        Determine if frame should be displayed based on detection presence
        
        Args:
            has_detections: Whether the frame contains detections
            
        Returns:
            bool: True if frame should be displayed
        """
        if not self.config['conditional_display']['detection_only']:
            return True  # Always display if not in detection-only mode
        
        return has_detections
    
    def process_frame_annotations(self, frame_meta) -> bool:
        """
        Process frame annotations and determine if frame has valid detections
        
        Args:
            frame_meta: DeepStream frame metadata
            
        Returns:
            bool: True if frame has detections above confidence threshold
        """
        try:
            self.frame_count += 1
            self.annotation_stats['frames_processed'] += 1
            
            has_detections = False
            objects_in_frame = 0
            
            # Process each object in the frame
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                    
                    if obj_meta:
                        confidence = obj_meta.confidence
                        
                        # Check if confidence meets threshold
                        if confidence >= self.config['conditional_display']['min_confidence']:
                            has_detections = True
                            objects_in_frame += 1
                            
                            # Customize annotation display
                            self._customize_object_annotation(obj_meta)
                        else:
                            # Hide low-confidence detections
                            obj_meta.rect_params.border_width = 0
                            obj_meta.text_params.display_text = 0
                
                except Exception as e:
                    self.logger.error(f"Error processing object metadata: {e}")
                
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break
            
            # Update statistics
            if has_detections:
                self.annotation_stats['frames_with_detections'] += 1
                self.annotation_stats['objects_annotated'] += objects_in_frame
            
            self.has_detections = has_detections
            return has_detections
            
        except Exception as e:
            self.logger.error(f"Error processing frame annotations: {e}")
            return False
    
    def _customize_object_annotation(self, obj_meta):
        """
        Customize annotation display for an object
        
        Args:
            obj_meta: DeepStream object metadata
        """
        try:
            class_id = obj_meta.class_id
            confidence = obj_meta.confidence
            tracking_id = obj_meta.object_id
            
            # Get class name using COCO dataset labels
            class_name = self._get_class_name(class_id)
            
            # Configure bounding box
            if self.config['annotations']['show_bounding_boxes']:
                # Get color for this class
                color = self._get_class_color(class_name)
                
                obj_meta.rect_params.border_color.red = color[0]
                obj_meta.rect_params.border_color.green = color[1]
                obj_meta.rect_params.border_color.blue = color[2]
                obj_meta.rect_params.border_color.alpha = color[3]
                obj_meta.rect_params.border_width = self.config['annotations']['border_width']
            else:
                obj_meta.rect_params.border_width = 0
            
            # Configure text display
            if (self.config['annotations']['show_class_labels'] or 
                self.config['annotations']['show_tracking_ids'] or
                self.config['annotations']['show_confidence']):
                
                # Build display text
                text_parts = []
                
                if self.config['annotations']['show_class_labels']:
                    text_parts.append(class_name)
                
                if self.config['annotations']['show_tracking_ids']:
                    text_parts.append(f"ID:{tracking_id}")
                
                if self.config['annotations']['show_confidence']:
                    text_parts.append(f"{confidence:.2f}")
                
                display_text = " | ".join(text_parts)
                
                # Set text properties
                obj_meta.text_params.display_text = 1
                obj_meta.text_params.x_offset = int(obj_meta.rect_params.left)
                obj_meta.text_params.y_offset = int(obj_meta.rect_params.top - 25)
                obj_meta.text_params.font_params.font_name = "Arial"
                obj_meta.text_params.font_params.font_size = self.config['annotations']['text_size']
                
                # Set text colors
                text_color = self.config['annotations']['text_color']
                obj_meta.text_params.font_params.font_color.red = text_color[0]
                obj_meta.text_params.font_params.font_color.green = text_color[1]
                obj_meta.text_params.font_params.font_color.blue = text_color[2]
                obj_meta.text_params.font_params.font_color.alpha = text_color[3]
                
                # Set background color
                bg_color = self.config['annotations']['text_bg_color']
                obj_meta.text_params.set_bg_clr = 1
                obj_meta.text_params.text_bg_clr.red = bg_color[0]
                obj_meta.text_params.text_bg_clr.green = bg_color[1]
                obj_meta.text_params.text_bg_clr.blue = bg_color[2]
                obj_meta.text_params.text_bg_clr.alpha = bg_color[3]
                
                # Note: DeepStream handles text rendering automatically based on the configured parameters
                # The display_text flag and other text parameters control the rendering
            else:
                obj_meta.text_params.display_text = 0
                
        except Exception as e:
            self.logger.error(f"Error customizing object annotation: {e}")
    
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
    
    def _get_class_color(self, class_name: str) -> Tuple[float, float, float, float]:
        """Get color for a class name"""
        colors = self.config['annotations']['colors']
        return colors.get(class_name, colors['default'])
    
    def get_statistics(self) -> Dict:
        """Get video output statistics"""
        return {
            **self.annotation_stats,
            'current_frame_has_detections': self.has_detections,
            'total_frames': self.frame_count,
            'detection_rate': (
                self.annotation_stats['frames_with_detections'] / 
                max(self.annotation_stats['frames_processed'], 1)
            ),
            'config': self.config
        }
    
    def update_config(self, new_config: Dict):
        """Update configuration at runtime"""
        try:
            # Merge new config with existing
            def deep_merge(dict1, dict2):
                result = dict1.copy()
                for key, value in dict2.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result
            
            self.config = deep_merge(self.config, new_config)
            self.logger.info("Video output configuration updated")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")


# Probe function for video output processing
def video_output_probe(pad, info, user_data):
    """
    Probe function to process video output and annotations
    
    Args:
        pad: GStreamer pad
        info: Probe info
        user_data: VideoOutputManager instance
    """
    try:
        video_manager = user_data
        
        if not video_manager:
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
                    # Process frame annotations
                    has_detections = video_manager.process_frame_annotations(frame_meta)
                    
                    # Check if frame should be displayed
                    should_display = video_manager.should_display_frame(has_detections)
                    
                    if not should_display:
                        # Skip this frame (in detection-only mode)
                        # Note: Actual frame dropping would require more complex pipeline manipulation
                        pass
                
            except StopIteration:
                break
            
            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        
        return Gst.PadProbeReturn.OK
        
    except Exception as e:
        print(f"Error in video output probe: {e}")
        return Gst.PadProbeReturn.OK


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example configuration
    config = {
        'annotations': {
            'show_bounding_boxes': True,
            'show_tracking_ids': True,
            'show_class_labels': True,
            'colors': {
                'person': (1.0, 0.0, 0.0, 1.0),  # Red
                'car': (0.0, 1.0, 0.0, 1.0)      # Green
            }
        },
        'conditional_display': {
            'detection_only': False,
            'min_confidence': 0.5
        }
    }
    
    # Create video output manager
    manager = VideoOutputManager(config)
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"Video output manager statistics: {stats}")
    
    print("Video output manager initialized successfully")
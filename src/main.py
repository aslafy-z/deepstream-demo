#!/usr/bin/env python3

"""
RTSP Object Detection System - Main Application
Built with NVIDIA DeepStream
"""

import sys
import argparse
import logging
from pathlib import Path

from pipeline_manager import DeepStreamPipelineManager, setup_logging


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RTSP Object Detection System using DeepStream"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/deepstream_app_config.txt",
        help="Path to DeepStream application config file"
    )
    
    parser.add_argument(
        "--rtsp-uri",
        type=str,
        help="RTSP stream URI (overrides config file)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    return parser.parse_args()


def main():
    """Main application entry point"""
    args = parse_arguments()
    
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, args.log_level))
    
    logger.info("Starting RTSP Object Detection System")
    logger.info(f"Using config file: {args.config}")
    
    try:
        # Validate config file exists
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Config file not found: {args.config}")
            return 1
        
        # Create and run pipeline manager
        app_config_path = "configs/app_config.yaml"
        manager = DeepStreamPipelineManager(args.config, app_config_path, args.rtsp_uri)
        success = manager.run()
        
        if success:
            logger.info("Application completed successfully")
            return 0
        else:
            logger.error("Application failed")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
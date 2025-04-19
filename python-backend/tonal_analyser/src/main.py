#!/usr/bin/env python3
"""
Real-Time Tonal Analysis Tool - Main Entry Point
"""

import sys
import logging
from PyQt5.QtWidgets import QApplication

from gui.main_window import MainWindow
from audio.capture import AudioCapture
from features.extractor import FeatureExtractor
from analysis.analyzer import TonalAnalyzer
from visualization.visualizer import Visualizer
from utils.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to initialize and run the application"""
    try:
        logger.info("Starting Real-Time Tonal Analysis Tool")
        
        # Load configuration
        config = Config()
        
        # Initialize components
        audio_capture = AudioCapture(config)
        feature_extractor = FeatureExtractor(config)
        analyzer = TonalAnalyzer(config)
        visualizer = Visualizer(config)
        
        # Initialize GUI
        app = QApplication(sys.argv)
        window = MainWindow(
            config=config,
            audio_capture=audio_capture,
            feature_extractor=feature_extractor,
            analyzer=analyzer,
            visualizer=visualizer
        )
        window.show()
        
        # Start application
        return app.exec_()
        
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
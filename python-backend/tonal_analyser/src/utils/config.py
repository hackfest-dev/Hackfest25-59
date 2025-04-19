"""
Configuration module for the Real-Time Tonal Analysis Tool
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class Config:
    # Audio settings
    sample_rate: int = 44100
    chunk_size: int = 1024
    channels: int = 1
    format_bytes: int = 2  # 16-bit audio
    
    # Feature extraction settings
    use_gpu: bool = True
    mfcc_count: int = 13
    audio_window_size: float = 0.025  # in seconds - for audio processing
    hop_size: float = 0.01  # in seconds
    
    # Analysis settings
    pitch_algorithm: str = "yin"  # options: yin, pyin, crepe
    emotion_model_path: str = "models/emotion_model.pt"
    emotion_classes: List[str] = field(default_factory=lambda: [
        "neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"
    ])
    
    # Visualization settings
    plot_update_interval: int = 30  # in milliseconds
    max_history: int = 100  # number of frames to keep in history
    
    # GUI settings
    window_title: str = "Real-Time Tonal Analysis Tool"
    window_width: int = 1024
    window_height: int = 768
    
    @property
    def window_size(self) -> Tuple[int, int]:
        """Return window size as a tuple for use with QWidget.resize()"""
        return (self.window_width, self.window_height)
    
    @window_size.setter
    def window_size(self, size: Tuple[int, int]) -> None:
        """Set window width and height from a tuple"""
        if isinstance(size, tuple) and len(size) == 2:
            self.window_width, self.window_height = size
    
    def __post_init__(self):
        """Load config from file if available, and set up GPU settings"""
        # Try to load config from file
        config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    
                    # Handle window_size for backward compatibility
                    if 'window_size' in config_data and isinstance(config_data['window_size'], float):
                        # This is the old audio window size, move it to audio_window_size
                        config_data['audio_window_size'] = config_data.pop('window_size')
                    
                    for key, value in config_data.items():
                        if hasattr(self, key):
                            setattr(self, key, value)
                            
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        # Check GPU availability
        if self.use_gpu:
            try:
                import torch
                self.use_gpu = torch.cuda.is_available()
                if self.use_gpu:
                    logger.info(f"GPU acceleration enabled. Device: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning("GPU requested but not available. Falling back to CPU.")
            except ImportError:
                logger.warning("PyTorch not installed. GPU acceleration disabled.")
                self.use_gpu = False
    
    def save(self, path: Optional[str] = None) -> bool:
        """Save the current configuration to a file"""
        if path is None:
            path = os.path.join(os.path.dirname(__file__), '..', '..', 'config.json')
        
        try:
            # Convert dataclass to dict, excluding methods and properties
            config_dict = {
                k: v for k, v in self.__dict__.items() 
                if not callable(v) and not k.startswith('__')
            }
            
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=4)
            
            logger.info(f"Configuration saved to {path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            return False 
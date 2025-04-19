"""
Real-time audio capture module for the Tonal Analysis Tool
"""

import logging
import numpy as np
import pyaudio
import threading
import time
from typing import Callable, List, Optional, Dict, Any
from collections import deque
from queue import Queue

from utils.config import Config

logger = logging.getLogger(__name__)

class AudioCapture:
    """Handles real-time audio capture from microphone input"""
    
    def __init__(self, config: Config):
        """Initialize audio capture with given configuration
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.sample_rate = config.sample_rate
        self.chunk_size = config.chunk_size
        self.channels = config.channels
        self.format = pyaudio.paInt16  # 16-bit format
        
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.thread = None
        self.callbacks = []
        self.data_queue = Queue()
        self.buffer = deque(maxlen=config.max_history)
        
        # Query available input devices
        self._available_devices = self._get_input_devices()
        
    def _get_input_devices(self) -> Dict[int, str]:
        """Get available input devices
        
        Returns:
            Dict mapping device indices to their names
        """
        devices = {}
        for i in range(self.audio.get_device_count()):
            device_info = self.audio.get_device_info_by_index(i)
            if device_info['maxInputChannels'] > 0:
                devices[i] = device_info['name']
        
        logger.info(f"Found {len(devices)} input devices")
        return devices
    
    def get_input_devices(self) -> Dict[int, str]:
        """Get available input devices
        
        Returns:
            Dict mapping device indices to their names
        """
        return self._available_devices
    
    def start(self, device_index: Optional[int] = None) -> bool:
        """Start audio capture
        
        Args:
            device_index: Optional index of input device to use, 
                          None for default device
        
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_recording:
            logger.warning("Audio capture already running")
            return False
        
        try:
            # Create and start audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            # Start thread for processing audio data
            self.is_recording = True
            self.thread = threading.Thread(target=self._process_audio)
            self.thread.daemon = True
            self.thread.start()
            
            device_name = "default"
            if device_index is not None:
                device_name = self._available_devices.get(
                    device_index, f"Device {device_index}"
                )
            
            logger.info(f"Started audio capture from {device_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting audio capture: {e}")
            self.is_recording = False
            return False
    
    def stop(self) -> None:
        """Stop audio capture"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        
        logger.info("Stopped audio capture")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback function for PyAudio stream
        
        This is called by PyAudio when new audio data is available
        """
        if status:
            logger.warning(f"Audio stream status: {status}")
        
        # Put data in queue for processing
        self.data_queue.put(in_data)
        
        return (None, pyaudio.paContinue)
    
    def _process_audio(self) -> None:
        """Process audio data from queue"""
        while self.is_recording:
            try:
                # Get data from queue
                if self.data_queue.empty():
                    time.sleep(0.001)  # Short sleep to prevent CPU hogging
                    continue
                
                # Get data from queue
                data = self.data_queue.get()
                
                # Convert bytes to numpy array
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Normalize to float in range [-1.0, 1.0]
                audio_data = audio_data.astype(np.float32) / 32768.0
                
                # Store in buffer
                self.buffer.append(audio_data)
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(audio_data)
                    except Exception as e:
                        logger.error(f"Error in audio callback: {e}")
                
            except Exception as e:
                logger.error(f"Error processing audio: {e}")
    
    def register_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Register a callback function to be called with each audio chunk
        
        Args:
            callback: Function that takes a numpy array of audio data
        """
        self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Unregister a previously registered callback
        
        Args:
            callback: The callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_buffer(self) -> List[np.ndarray]:
        """Get the current audio buffer
        
        Returns:
            List of audio chunks in buffer
        """
        return list(self.buffer)
    
    def __del__(self):
        """Clean up resources"""
        self.stop()
        if self.audio is not None:
            self.audio.terminate() 
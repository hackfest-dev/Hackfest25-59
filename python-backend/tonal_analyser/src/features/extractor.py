"""
Feature extraction module for the Real-Time Tonal Analysis Tool
"""

import logging
import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from queue import Queue
from dataclasses import dataclass, field
import librosa

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from utils.config import Config

logger = logging.getLogger(__name__)

@dataclass
class AudioFeatures:
    """Container for extracted audio features"""
    # Raw audio data
    audio: np.ndarray
    sample_rate: int
    
    # Time domain features
    energy: np.ndarray = None
    rms: np.ndarray = None
    zero_crossing_rate: np.ndarray = None
    
    # Frequency domain features
    pitch: np.ndarray = None
    pitch_confidence: np.ndarray = None
    spectral_centroid: np.ndarray = None
    spectral_bandwidth: np.ndarray = None
    spectral_rolloff: np.ndarray = None
    
    # Cepstral features
    mfccs: np.ndarray = None
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    chunk_index: int = 0
    processing_time: float = 0.0


class FeatureExtractor:
    """Extracts audio features from audio chunks"""
    
    def __init__(self, config: Config):
        """Initialize feature extractor
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.sample_rate = config.sample_rate
        self.window_length = int(config.audio_window_size * config.sample_rate)  # Audio processing window size in samples
        self.hop_size = int(config.hop_size * config.sample_rate)
        self.chunk_size = config.chunk_size
        self.mfcc_count = config.mfcc_count
        self.use_gpu = config.use_gpu and TORCH_AVAILABLE
        
        # Processing queue and thread
        self.queue = Queue()
        self.is_processing = False
        self.thread = None
        self.callbacks = []
        
        # GPU setup if available
        self.device = None
        if self.use_gpu:
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info(f"Feature extraction using device: {self.device}")
            except Exception as e:
                logger.error(f"Error setting up GPU: {e}")
                self.use_gpu = False
        
        # Create pitch extractor based on algorithm
        self._setup_pitch_extractor(config.pitch_algorithm)
        
        # Start processing thread
        self._start_thread()
    
    def _setup_pitch_extractor(self, algorithm: str) -> None:
        """Set up pitch extraction algorithm
        
        Args:
            algorithm: Name of pitch algorithm to use (yin, pyin, crepe)
        """
        self.pitch_algorithm = algorithm
        
        if algorithm == "crepe" and self.use_gpu and TORCH_AVAILABLE:
            try:
                # Try to import crepe
                import crepe
                self.pitch_extractor = "crepe"
                logger.info("Using CREPE pitch extraction with GPU acceleration")
            except ImportError:
                logger.warning("CREPE not installed, falling back to YIN algorithm")
                self.pitch_algorithm = "yin"
                self.pitch_extractor = "yin"
        else:
            self.pitch_extractor = algorithm
            logger.info(f"Using {algorithm} pitch extraction")
    
    def _start_thread(self) -> None:
        """Start the processing thread"""
        self.is_processing = True
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Feature extraction thread started")
    
    def process(self, audio_data: np.ndarray, chunk_index: int = 0) -> None:
        """Queue audio data for feature extraction
        
        Args:
            audio_data: Audio data as numpy array
            chunk_index: Index of the audio chunk
        """
        self.queue.put((audio_data, chunk_index))
    
    def _process_queue(self) -> None:
        """Process audio data from queue"""
        while self.is_processing:
            try:
                # Get data from queue
                if self.queue.empty():
                    time.sleep(0.001)  # Short sleep to prevent CPU hogging
                    continue
                
                # Get data from queue
                audio_data, chunk_index = self.queue.get()
                
                # Extract features
                start_time = time.time()
                features = self._extract_features(audio_data, chunk_index)
                features.processing_time = time.time() - start_time
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(features)
                    except Exception as e:
                        logger.error(f"Error in feature callback: {e}")
                
            except Exception as e:
                logger.error(f"Error extracting features: {e}")
    
    def _extract_features(self, audio_data: np.ndarray, chunk_index: int) -> AudioFeatures:
        """Extract features from audio data
        
        Args:
            audio_data: Audio data as numpy array
            chunk_index: Index of the audio chunk
            
        Returns:
            AudioFeatures object containing extracted features
        """
        # Create feature container
        features = AudioFeatures(
            audio=audio_data,
            sample_rate=self.sample_rate,
            chunk_index=chunk_index
        )
        
        # Extract time domain features
        features.energy = np.sum(audio_data**2)
        features.rms = librosa.feature.rms(y=audio_data)[0]
        features.zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
        
        # Extract pitch
        if self.pitch_algorithm == "yin":
            try:
                # Use a try/except block to handle different return formats
                try:
                    # New librosa versions return a tuple of (pitch, confidence)
                    pitch, pitch_confidence = librosa.yin(
                        audio_data, 
                        fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'),
                        sr=self.sample_rate
                    )
                except ValueError as unpacking_error:
                    if "too many values to unpack" in str(unpacking_error):
                        # Handle case where more than 2 values are returned
                        result = librosa.yin(
                            audio_data, 
                            fmin=librosa.note_to_hz('C2'),
                            fmax=librosa.note_to_hz('C7'),
                            sr=self.sample_rate
                        )
                        # Take the first two values from the result
                        if isinstance(result, tuple) and len(result) > 1:
                            pitch, pitch_confidence = result[0], result[1]
                        else:
                            # If result is not a tuple or doesn't have enough elements
                            logger.warning("Unexpected YIN result format: %s", type(result))
                            pitch = np.zeros(1)
                            pitch_confidence = np.zeros(1)
                    else:
                        # Re-raise if it's a different ValueError
                        raise
                
                features.pitch = pitch
                features.pitch_confidence = pitch_confidence
                logger.debug("Extracted pitch with YIN algorithm: %d values", len(pitch))
            except Exception as e:
                logger.error(f"Error extracting pitch with YIN: {e}")
                # Initialize empty arrays for pitch data
                features.pitch = np.array([])
                features.pitch_confidence = np.array([])
        elif self.pitch_algorithm == "pyin":
            # Use probabilistic YIN algorithm
            try:
                # pyin returns (pitch, voiced_flag, voiced_probs)
                pitch, voiced_flag, voiced_probs = librosa.pyin(
                    audio_data,
                    fmin=librosa.note_to_hz('C2'),
                    fmax=librosa.note_to_hz('C7'),
                    sr=self.sample_rate
                )
                features.pitch = pitch
                # Use voiced_probs as confidence
                features.pitch_confidence = voiced_probs
            except Exception as e:
                logger.error(f"Error extracting pitch with PYIN: {e}")
        elif self.pitch_algorithm == "crepe" and self.pitch_extractor == "crepe":
            # Use CREPE for pitch extraction if available
            try:
                import crepe
                time_step = 1000 * self.config.hop_size  # in milliseconds
                
                if self.use_gpu:
                    with torch.no_grad():
                        pitch, confidence, _ = crepe.predict(
                            audio_data, 
                            self.sample_rate,
                            step_size=time_step,
                            model_capacity="medium",
                            device=self.device
                        )
                else:
                    pitch, confidence, _ = crepe.predict(
                        audio_data, 
                        self.sample_rate,
                        step_size=time_step,
                        model_capacity="medium"
                    )
                    
                features.pitch = pitch
                features.pitch_confidence = confidence
            except Exception as e:
                logger.error(f"Error extracting pitch with CREPE: {e}")
                # Fall back to YIN
                try:
                    pitch, pitch_confidence = librosa.yin(
                        audio_data, 
                        fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'),
                        sr=self.sample_rate
                    )
                    features.pitch = pitch
                    features.pitch_confidence = pitch_confidence
                except Exception as fallback_error:
                    logger.error(f"Error in YIN fallback: {fallback_error}")
        
        # Extract frequency domain features
        if len(audio_data) > 0:
            try:
                features.spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
                features.spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
                features.spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
                
                # Extract MFCCs
                features.mfccs = librosa.feature.mfcc(
                    y=audio_data, 
                    sr=self.sample_rate, 
                    n_mfcc=self.mfcc_count
                )
            except Exception as e:
                logger.error(f"Error extracting frequency domain features: {e}")
        
        return features
    
    def register_callback(self, callback: Callable[[AudioFeatures], None]) -> None:
        """Register a callback for processed features
        
        Args:
            callback: Function that takes AudioFeatures object
        """
        self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[AudioFeatures], None]) -> None:
        """Unregister a callback
        
        Args:
            callback: The callback to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def stop(self) -> None:
        """Stop the feature extractor"""
        self.is_processing = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        logger.info("Feature extractor stopped") 
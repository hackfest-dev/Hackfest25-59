"""
Analyzer module for tonal, prosodic, and emotional analysis
"""

import logging
import numpy as np
import os
import threading
import time
from typing import Dict, List, Optional, Tuple, Any, Callable
from queue import Queue
from dataclasses import dataclass, field
from collections import deque
import librosa

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from scipy.special import softmax as scipy_softmax
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from utils.config import Config
from features.extractor import AudioFeatures

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    # Input features
    features: AudioFeatures
    
    # Tonal analysis
    pitch_mean: float = 0.0
    pitch_std: float = 0.0
    pitch_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 0.0))
    pitch_contour: np.ndarray = None
    
    # Prosodic analysis
    speech_rate: float = 0.0
    rhythm_regularity: float = 0.0
    emphasis: List[int] = field(default_factory=list)
    
    # Emotional analysis
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    dominant_emotion: str = "neutral"
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    processing_time: float = 0.0


def numpy_softmax(x):
    """Compute softmax values for numpy array.
    
    Args:
        x: Input numpy array
        
    Returns:
        Softmax of input array
    """
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class EmotionModel:
    """Emotion classification model"""
    
    def __init__(self, config: Config):
        """Initialize emotion model
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.use_gpu = config.use_gpu and TORCH_AVAILABLE
        self.emotion_classes = config.emotion_classes
        self.model = None
        self.device = torch.device("cpu")
        
        logger.info("Initializing emotion model with %d classes", len(self.emotion_classes))
        
        if self.use_gpu:
            try:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                logger.info("Using device: %s", self.device)
            except Exception as e:
                logger.error(f"Error setting up GPU for emotion model: {e}")
                self.use_gpu = False
        
        self._load_model(config.emotion_model_path)
    
    def _load_model(self, model_path: str) -> None:
        """Load emotion model from path
        
        Args:
            model_path: Path to the model file
        """
        # Check if model file exists
        if not os.path.exists(model_path):
            logger.warning(f"Emotion model not found at {model_path}. Using dummy model.")
            self._create_dummy_model()
            return
        
        try:
            if not TORCH_AVAILABLE:
                logger.warning("PyTorch not available. Using dummy model.")
                self._create_dummy_model()
                return
            
            # Load the model
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"Loaded emotion model from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading emotion model: {e}")
            self._create_dummy_model()
    
    def _create_dummy_model(self) -> None:
        """Create a dummy model that returns random emotion scores"""
        logger.info("Creating dummy emotion model")
        
        class DummyModel:
            def __init__(self, classes):
                self.classes = classes
                self.device = torch.device("cpu")
            
            def eval(self):
                pass
            
            def __call__(self, features):
                # Return random probabilities
                probabilities = np.random.random(len(self.classes))
                # Normalize
                probabilities = probabilities / np.sum(probabilities)
                return probabilities
        
        self.model = DummyModel(self.emotion_classes)
    
    def predict(self, features: AudioFeatures) -> Dict[str, float]:
        """Predict emotions from features
        
        Args:
            features: Audio features
            
        Returns:
            Dictionary mapping emotion names to probabilities
        """
        if self.model is None:
            logger.warning("No model loaded, creating dummy model")
            self._create_dummy_model()
        
        try:
            logger.debug("Starting emotion prediction")
            # Extract relevant features (MFCCs, pitch, energy)
            # and prepare them for the model
            if TORCH_AVAILABLE and not isinstance(self.model, type):
                # Convert features to tensor for PyTorch model
                mfccs = features.mfccs
                if mfccs is not None:
                    logger.debug("Using MFCCs for emotion prediction, shape: %s", str(mfccs.shape))
                    mfccs_tensor = torch.from_numpy(mfccs).float().to(self.device)
                    
                    # Add batch dimension if needed
                    if len(mfccs_tensor.shape) == 2:
                        mfccs_tensor = mfccs_tensor.unsqueeze(0)
                    
                    # Run inference
                    with torch.no_grad():
                        outputs = self.model(mfccs_tensor)
                        logger.debug("Model output type: %s, shape: %s", 
                                    type(outputs).__name__,
                                    str(outputs.shape) if hasattr(outputs, 'shape') else "unknown")
                        
                        # Check if outputs is a tensor or numpy array
                        if isinstance(outputs, torch.Tensor):
                            outputs = outputs.cpu().numpy()
                        
                        # Process model outputs
                        if isinstance(outputs, np.ndarray):
                            # Check dimensionality and handle accordingly
                            if outputs.ndim == 0:  # Scalar output
                                logger.warning("Model returned scalar output, generating random probabilities")
                                probabilities = np.random.rand(len(self.emotion_classes))
                                probabilities = probabilities / np.sum(probabilities)  # Normalize
                            elif outputs.ndim == 1:  # 1D array
                                # Apply softmax if needed
                                if outputs.size > 1:
                                    probabilities = scipy_softmax(outputs)
                                else:
                                    logger.warning("Model returned single value output, generating random probabilities")
                                    probabilities = np.random.rand(len(self.emotion_classes))
                                    probabilities = probabilities / np.sum(probabilities)  # Normalize
                            else:  # 2D+ array
                                # Take first sample if batch output
                                outputs = outputs[0] if outputs.ndim > 1 else outputs
                                probabilities = scipy_softmax(outputs)
                        else:
                            # Unknown output format, generate random probabilities
                            logger.warning(f"Unknown model output format: {type(outputs)}, generating random probabilities")
                            probabilities = np.random.rand(len(self.emotion_classes))
                            probabilities = probabilities / np.sum(probabilities)  # Normalize
            else:
                # Dummy prediction
                outputs = self.model(features.mfccs)
                # Ensure outputs is an array
                if not isinstance(outputs, (list, np.ndarray)) or (isinstance(outputs, np.ndarray) and outputs.ndim == 0):
                    logger.warning("Dummy model returned non-array output, converting to array")
                    if isinstance(outputs, (int, float, np.float64, np.int64)):
                        # Convert scalar to array with single value
                        outputs = np.array([float(outputs)])
                    else:
                        # Generate random outputs
                        outputs = np.random.rand(len(self.emotion_classes))
                
                # Apply softmax if multiple values
                if isinstance(outputs, np.ndarray) and outputs.size > 1:
                    probabilities = scipy_softmax(outputs)
                else:
                    # Generate random probabilities if we have a single value
                    probabilities = np.random.rand(len(self.emotion_classes))
                    probabilities = probabilities / np.sum(probabilities)  # Normalize
            
            # Ensure probabilities match the number of emotion classes
            if len(probabilities) != len(self.emotion_classes):
                logger.warning(
                    f"Probability length ({len(probabilities)}) does not match emotion classes ({len(self.emotion_classes)})"
                )
                if len(probabilities) > len(self.emotion_classes):
                    # Truncate probabilities
                    probabilities = probabilities[:len(self.emotion_classes)]
                else:
                    # Extend probabilities with zeros and renormalize
                    extended = np.zeros(len(self.emotion_classes))
                    extended[:len(probabilities)] = probabilities
                    probabilities = extended / np.sum(extended) if np.sum(extended) > 0 else extended
            
            # Create emotion scores dict
            emotion_scores = {
                emotion: float(probability)
                for emotion, probability in zip(self.emotion_classes, probabilities)
            }
            return emotion_scores
        except Exception as e:
            logger.error(f"Error predicting emotions: {e}", exc_info=True)
            # Return uniform probabilities
            return {emotion: 1.0 / len(self.emotion_classes) for emotion in self.emotion_classes}


class TonalAnalyzer:
    """Analyzes audio features for tonal, prosodic, and emotional content"""
    
    def __init__(self, config: Config):
        """Initialize tonal analyzer
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.use_gpu = config.use_gpu and TORCH_AVAILABLE
        
        logger.info("Initializing tonal analyzer")
        
        # Create emotion model
        self.emotion_model = EmotionModel(config)
        
        # Processing queue and thread
        self.queue = Queue()
        self.is_processing = False
        self.thread = None
        self.callbacks = []
        
        # History for temporal analysis
        self.history = deque(maxlen=config.max_history)
        logger.info("History buffer initialized with max length %d", config.max_history)
        
        # Start processing thread
        self._start_thread()
    
    def _start_thread(self) -> None:
        """Start the processing thread"""
        self.is_processing = True
        self.thread = threading.Thread(target=self._process_queue)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Tonal analyzer thread started")
    
    def process(self, features: AudioFeatures) -> None:
        """Queue features for analysis
        
        Args:
            features: Audio features to analyze
        """
        self.queue.put(features)
        logger.debug("Added features to analysis queue, queue size: %d", self.queue.qsize())
    
    def _process_queue(self) -> None:
        """Process features from queue"""
        logger.info("Starting queue processing loop")
        while self.is_processing:
            try:
                # Get data from queue
                if self.queue.empty():
                    time.sleep(0.001)  # Short sleep to prevent CPU hogging
                    continue
                
                # Get features from queue
                features = self.queue.get()
                logger.debug("Processing features from queue, remaining: %d", self.queue.qsize())
                
                # Analyze features
                start_time = time.time()
                result = self._analyze_features(features)
                result.processing_time = time.time() - start_time
                logger.debug("Analysis completed in %.3f seconds", result.processing_time)
                
                # Add to history
                self.history.append(result)
                
                # Call registered callbacks
                callback_count = 0
                for callback in self.callbacks:
                    try:
                        callback(result)
                        callback_count += 1
                    except Exception as e:
                        logger.error(f"Error in analyzer callback: {e}")
                
                logger.debug("Executed %d callbacks for analysis result", callback_count)
                
            except Exception as e:
                logger.error(f"Error analyzing features: {e}", exc_info=True)
    
    def _analyze_features(self, features: AudioFeatures) -> AnalysisResult:
        """Analyze features for tonal, prosodic, and emotional content
        
        Args:
            features: Audio features to analyze
            
        Returns:
            AnalysisResult object containing analysis results
        """
        # Create result container
        result = AnalysisResult(features=features)
        
        # Tonal analysis
        logger.debug("Starting tonal analysis")
        self._analyze_tone(features, result)
        
        # Prosodic analysis
        logger.debug("Starting prosodic analysis")
        self._analyze_prosody(features, result)
        
        # Emotional analysis
        logger.debug("Starting emotional analysis")
        self._analyze_emotion(features, result)
        
        return result
    
    def _analyze_tone(self, features: AudioFeatures, result: AnalysisResult) -> None:
        """Analyze tonal characteristics
        
        Args:
            features: Audio features
            result: Analysis result to update
        """
        try:
            # Extract pitch information
            pitch = features.pitch
            confidence = features.pitch_confidence
            
            if pitch is not None and len(pitch) > 0:
                # Initialize pitch contour
                result.pitch_contour = np.array([])
                
                # Only consider pitch values with high confidence
                if confidence is not None and len(confidence) == len(pitch):
                    # Ensure confidence array has the same length as pitch
                    confident_indices = confidence > 0.5
                    if np.any(confident_indices):
                        confident_pitch = pitch[confident_indices]
                        if len(confident_pitch) > 0:
                            pitch = confident_pitch
                
                # Remove zeros and NaN values (unpitched frames)
                valid_indices = np.logical_and(pitch > 0, ~np.isnan(pitch))
                if np.any(valid_indices):
                    valid_pitch = pitch[valid_indices]
                    
                    if len(valid_pitch) > 0:
                        # Calculate statistical measures
                        result.pitch_mean = float(np.mean(valid_pitch))
                        result.pitch_std = float(np.std(valid_pitch))
                        result.pitch_range = (float(np.min(valid_pitch)), float(np.max(valid_pitch)))
                        
                        # Process the pitch contour for better visualization
                        # 1. Remove outliers (values outside 3 standard deviations)
                        if len(valid_pitch) > 3:  # Only if we have enough points
                            mean = np.mean(valid_pitch)
                            std = np.std(valid_pitch)
                            threshold = 3 * std
                            inlier_indices = np.abs(valid_pitch - mean) <= threshold
                            if np.any(inlier_indices):
                                filtered_pitch = valid_pitch[inlier_indices]
                            else:
                                filtered_pitch = valid_pitch
                        else:
                            filtered_pitch = valid_pitch
                        
                        # 2. Apply smoothing if there are enough points
                        if len(filtered_pitch) > 5:
                            try:
                                # Simple moving average for smoothing
                                window_size = min(5, len(filtered_pitch) // 2)
                                if window_size > 1:
                                    smoothed_pitch = np.convolve(
                                        filtered_pitch, 
                                        np.ones(window_size) / window_size, 
                                        mode='valid'
                                    )
                                else:
                                    smoothed_pitch = filtered_pitch
                            except Exception as e:
                                logger.warning(f"Error smoothing pitch contour: {e}")
                                smoothed_pitch = filtered_pitch
                        else:
                            smoothed_pitch = filtered_pitch
                        
                        # Store the processed pitch contour
                        result.pitch_contour = np.copy(smoothed_pitch)
                        
                        # Log success
                        logger.debug("Pitch contour extracted and processed: %d points, range: %.1f-%.1f Hz", 
                                    len(result.pitch_contour), 
                                    np.min(result.pitch_contour) if len(result.pitch_contour) > 0 else 0,
                                    np.max(result.pitch_contour) if len(result.pitch_contour) > 0 else 0)
        except Exception as e:
            logger.error(f"Error in tonal analysis: {e}", exc_info=True)
            # Initialize default values if analysis fails
            result.pitch_mean = 0.0
            result.pitch_std = 0.0
            result.pitch_range = (0.0, 0.0)
            result.pitch_contour = np.array([])
    
    def _analyze_prosody(self, features: AudioFeatures, result: AnalysisResult) -> None:
        """Analyze prosodic characteristics
        
        Args:
            features: Audio features
            result: Analysis result to update
        """
        # Calculate speech rate (approximation based on zero crossings)
        if features.zero_crossing_rate is not None and len(features.zero_crossing_rate) > 0:
            result.speech_rate = float(np.mean(features.zero_crossing_rate))
        
        # Approximate rhythm regularity from energy fluctuations
        if features.rms is not None and len(features.rms) > 2:
            # Calculate differences between consecutive RMS values
            rms_diffs = np.diff(features.rms)
            if len(rms_diffs) > 0:
                # Standard deviation of differences as a measure of regularity
                # (lower values indicate more regular rhythm)
                regularity = 1.0 / (1.0 + np.std(rms_diffs))
                result.rhythm_regularity = float(regularity)
        
        # Detect emphasis points (peaks in energy)
        if features.rms is not None and len(features.rms) > 0:
            # Find local maxima in RMS energy
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(features.rms, height=np.mean(features.rms) * 1.5)
            result.emphasis = peaks.tolist()
    
    def _analyze_emotion(self, features: AudioFeatures, result: AnalysisResult) -> None:
        """Analyze emotional characteristics
        
        Args:
            features: Audio features
            result: Analysis result to update
        """
        # Predict emotions using model
        emotion_scores = self.emotion_model.predict(features)
        result.emotion_scores = emotion_scores
        
        # Find dominant emotion
        if emotion_scores:
            result.dominant_emotion = max(
                emotion_scores.items(), 
                key=lambda x: x[1]
            )[0]
    
    def register_callback(self, callback: Callable[[AnalysisResult], None]) -> None:
        """Register a callback for analysis results
        
        Args:
            callback: Function that takes AnalysisResult object
        """
        self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[AnalysisResult], None]) -> None:
        """Unregister a callback
        
        Args:
            callback: The callback to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_history(self) -> List[AnalysisResult]:
        """Get analysis history
        
        Returns:
            List of recent analysis results
        """
        return list(self.history)
    
    def stop(self) -> None:
        """Stop the analyzer"""
        logger.info("Stopping tonal analyzer")
        self.is_processing = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                logger.warning("Analyzer thread did not terminate cleanly")
            else:
                logger.info("Analyzer thread terminated successfully")
        logger.info("Tonal analyzer stopped - processed %d audio segments", len(self.history)) 
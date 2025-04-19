"""
Main GUI window for the Real-Time Tonal Analysis Tool
"""

import logging
import sys
import numpy as np
from typing import Dict, List, Optional
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QComboBox, QLabel, QGroupBox,
    QSplitter, QTextEdit, QScrollArea, QFrame,
    QSlider, QSpinBox, QCheckBox, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer, pyqtSlot

from utils.config import Config
from audio.capture import AudioCapture
from features.extractor import FeatureExtractor, AudioFeatures
from analysis.analyzer import TonalAnalyzer, AnalysisResult
from visualization.visualizer import Visualizer

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(
        self, 
        config: Config,
        audio_capture: AudioCapture,
        feature_extractor: FeatureExtractor,
        analyzer: TonalAnalyzer,
        visualizer: Visualizer
    ):
        """Initialize main window
        
        Args:
            config: Application configuration
            audio_capture: Audio capture component
            feature_extractor: Feature extraction component
            analyzer: Analysis component
            visualizer: Visualization component
        """
        super().__init__()
        
        # Store components
        self.config = config
        self.audio_capture = audio_capture
        self.feature_extractor = feature_extractor
        self.analyzer = analyzer
        self.visualizer = visualizer
        
        # Setup UI
        self.setWindowTitle(config.window_title)
        self.resize(config.window_width, config.window_height)
        
        # Create the main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        
        # Create control panel
        self.create_control_panel()
        
        # Create visualization panels
        self.create_visualization_panels()
        
        # Connect components
        self.connect_components()
        
        # Set up status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Setup update timer for status
        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(1000)  # Update every second
        
        logger.info("Main window initialized")
    
    def create_control_panel(self) -> None:
        """Create the control panel with buttons and settings"""
        control_group = QGroupBox("Controls")
        control_layout = QHBoxLayout()
        
        # Input device selection
        device_layout = QVBoxLayout()
        device_label = QLabel("Audio Input:")
        self.device_combo = QComboBox()
        self.populate_device_list()
        device_layout.addWidget(device_label)
        device_layout.addWidget(self.device_combo)
        control_layout.addLayout(device_layout)
        
        # Start/stop button
        self.start_stop_button = QPushButton("Start")
        self.start_stop_button.clicked.connect(self.toggle_capture)
        control_layout.addWidget(self.start_stop_button)
        
        # Algorithm selection
        algo_layout = QVBoxLayout()
        algo_label = QLabel("Pitch Algorithm:")
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["yin", "pyin", "crepe"])
        self.algo_combo.setCurrentText(self.config.pitch_algorithm)
        self.algo_combo.currentTextChanged.connect(self.change_algorithm)
        algo_layout.addWidget(algo_label)
        algo_layout.addWidget(self.algo_combo)
        control_layout.addLayout(algo_layout)
        
        # GPU checkbox
        self.gpu_checkbox = QCheckBox("Use GPU")
        self.gpu_checkbox.setChecked(self.config.use_gpu)
        self.gpu_checkbox.toggled.connect(self.toggle_gpu)
        control_layout.addWidget(self.gpu_checkbox)
        
        # Sample rate selection
        rate_layout = QVBoxLayout()
        rate_label = QLabel("Sample Rate:")
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["16000", "22050", "44100", "48000"])
        self.rate_combo.setCurrentText(str(self.config.sample_rate))
        rate_layout.addWidget(rate_label)
        rate_layout.addWidget(self.rate_combo)
        control_layout.addLayout(rate_layout)
        
        # Chunk size adjustment
        chunk_layout = QVBoxLayout()
        chunk_label = QLabel("Chunk Size:")
        self.chunk_spin = QSpinBox()
        self.chunk_spin.setRange(256, 8192)
        self.chunk_spin.setSingleStep(256)
        self.chunk_spin.setValue(self.config.chunk_size)
        chunk_layout.addWidget(chunk_label)
        chunk_layout.addWidget(self.chunk_spin)
        control_layout.addLayout(chunk_layout)
        
        control_group.setLayout(control_layout)
        self.main_layout.addWidget(control_group)
    
    def create_visualization_panels(self) -> None:
        """Create visualization panels for different data views"""
        # Create a horizontal splitter for the top row
        top_splitter = QSplitter(Qt.Horizontal)
        
        # Waveform and spectrogram on top row
        waveform_group = QGroupBox("Audio Waveform")
        waveform_layout = QVBoxLayout()
        self.waveform_canvas = self.visualizer.create_canvas("waveform")
        waveform_layout.addWidget(self.waveform_canvas)
        waveform_group.setLayout(waveform_layout)
        top_splitter.addWidget(waveform_group)
        
        spectrogram_group = QGroupBox("Spectrogram")
        spectrogram_layout = QVBoxLayout()
        self.spectrogram_canvas = self.visualizer.create_canvas("spectrogram")
        spectrogram_layout.addWidget(self.spectrogram_canvas)
        spectrogram_group.setLayout(spectrogram_layout)
        top_splitter.addWidget(spectrogram_group)
        
        # Add top row to main layout
        self.main_layout.addWidget(top_splitter)
        
        # Create a horizontal splitter for the bottom row
        bottom_splitter = QSplitter(Qt.Horizontal)
        
        # Pitch contour and emotion bars on bottom row
        pitch_group = QGroupBox("Pitch Contour")
        pitch_layout = QVBoxLayout()
        self.pitch_canvas = self.visualizer.create_canvas("pitch")
        pitch_layout.addWidget(self.pitch_canvas)
        pitch_group.setLayout(pitch_layout)
        bottom_splitter.addWidget(pitch_group)
        
        emotion_group = QGroupBox("Emotion Analysis")
        emotion_layout = QVBoxLayout()
        self.emotion_canvas = self.visualizer.create_canvas("emotion")
        emotion_layout.addWidget(self.emotion_canvas)
        emotion_group.setLayout(emotion_layout)
        bottom_splitter.addWidget(emotion_group)
        
        # Add bottom row to main layout
        self.main_layout.addWidget(bottom_splitter)
        
        # Set relative sizes
        top_splitter.setSizes([500, 500])
        bottom_splitter.setSizes([500, 500])
        
        # Start visualizer updates
        self.visualizer.start_updates(self)
    
    def connect_components(self) -> None:
        """Connect signal flows between components"""
        # Audio capture -> Feature extraction
        self.audio_capture.register_callback(self.on_audio_data)
        
        # Feature extraction -> Analysis
        self.feature_extractor.register_callback(self.on_features)
        
        # Analysis -> Visualization
        self.analyzer.register_callback(self.on_analysis)
    
    def populate_device_list(self) -> None:
        """Populate the device list combobox"""
        devices = self.audio_capture.get_input_devices()
        self.device_combo.clear()
        
        for idx, name in devices.items():
            self.device_combo.addItem(name, idx)
    
    @pyqtSlot()
    def toggle_capture(self) -> None:
        """Toggle audio capture on/off"""
        if self.audio_capture.is_recording:
            # Stop recording
            self.audio_capture.stop()
            self.start_stop_button.setText("Start")
            self.status_bar.showMessage("Audio capture stopped")
        else:
            # Get selected device
            device_idx = self.device_combo.currentData()
            
            # Apply settings if changed
            new_sample_rate = int(self.rate_combo.currentText())
            new_chunk_size = self.chunk_spin.value()
            
            if (new_sample_rate != self.config.sample_rate or
                new_chunk_size != self.config.chunk_size):
                # Would need to restart components to apply these changes
                logger.warning("Sample rate and chunk size changes require restart")
            
            # Start recording
            success = self.audio_capture.start(device_idx)
            
            if success:
                self.start_stop_button.setText("Stop")
                self.status_bar.showMessage("Audio capture started")
            else:
                self.status_bar.showMessage("Failed to start audio capture")
    
    @pyqtSlot(str)
    def change_algorithm(self, algorithm: str) -> None:
        """Change the pitch extraction algorithm
        
        Args:
            algorithm: New algorithm to use
        """
        if algorithm != self.config.pitch_algorithm:
            self.config.pitch_algorithm = algorithm
            # Would need to restart feature extractor to apply this change
            logger.info(f"Pitch algorithm changed to {algorithm} (restart needed)")
            self.status_bar.showMessage(f"Pitch algorithm changed to {algorithm} (restart to apply)")
    
    @pyqtSlot(bool)
    def toggle_gpu(self, use_gpu: bool) -> None:
        """Toggle GPU acceleration
        
        Args:
            use_gpu: Whether to use GPU
        """
        if use_gpu != self.config.use_gpu:
            self.config.use_gpu = use_gpu
            # Would need to restart components to apply this change
            logger.info(f"GPU acceleration {'enabled' if use_gpu else 'disabled'} (restart needed)")
            self.status_bar.showMessage(f"GPU acceleration {'enabled' if use_gpu else 'disabled'} (restart to apply)")
    
    def on_audio_data(self, audio_data: np.ndarray) -> None:
        """Handle new audio data
        
        Args:
            audio_data: Audio data from capture
        """
        # Update visualizations
        self.visualizer.update_audio_waveform(audio_data)
        self.visualizer.update_spectrogram(audio_data)
        
        # Queue for feature extraction
        chunk_index = getattr(self, '_chunk_index', 0)
        self.feature_extractor.process(audio_data, chunk_index)
        self._chunk_index = chunk_index + 1
    
    def on_features(self, features: AudioFeatures) -> None:
        """Handle extracted features
        
        Args:
            features: Extracted audio features
        """
        # Queue for analysis
        self.analyzer.process(features)
    
    def on_analysis(self, result: AnalysisResult) -> None:
        """Handle analysis results
        
        Args:
            result: Analysis results
        """
        # Update visualizations
        self.visualizer.update_pitch_contour(result)
        self.visualizer.update_emotion_bars(result)
        
        # Store the latest result for status updates
        self._latest_result = result
    
    def update_status(self) -> None:
        """Update status bar with performance metrics"""
        if hasattr(self, '_latest_result') and self._latest_result is not None:
            result = self._latest_result
            
            # Calculate total processing time
            feature_time = result.features.processing_time * 1000  # ms
            analysis_time = result.processing_time * 1000  # ms
            total_time = feature_time + analysis_time
            
            # Get audio chunk duration
            chunk_duration = len(result.features.audio) / result.features.sample_rate * 1000  # ms
            
            # Build status message
            status = (
                f"Processing: {total_time:.1f}ms (features: {feature_time:.1f}ms, "
                f"analysis: {analysis_time:.1f}ms) | "
                f"Audio chunk: {chunk_duration:.1f}ms | "
                f"Real-time factor: {total_time/chunk_duration:.2f}x | "
                f"Dominant emotion: {result.dominant_emotion} | "
                f"Pitch: {result.pitch_mean:.1f}Hz"
            )
            
            self.status_bar.showMessage(status)
    
    def closeEvent(self, event) -> None:
        """Handle window close event"""
        logger.info("Closing application")
        
        # Stop audio capture
        if self.audio_capture.is_recording:
            self.audio_capture.stop()
        
        # Stop visualizer updates
        self.visualizer.stop_updates()
        
        # Stop feature extractor
        self.feature_extractor.stop()
        
        # Stop analyzer
        self.analyzer.stop()
        
        # Save configuration
        self.config.save()
        
        # Accept the close event
        event.accept() 
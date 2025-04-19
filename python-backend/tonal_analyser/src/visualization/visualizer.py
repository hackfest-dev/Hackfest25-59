"""
Visualization module for real-time tonal analysis display
"""

import logging
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import deque
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5 backend for matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QTimer

from utils.config import Config
from features.extractor import AudioFeatures
from analysis.analyzer import AnalysisResult

logger = logging.getLogger(__name__)


class AudioWaveformCanvas(FigureCanvas):
    """Canvas for displaying audio waveform"""
    
    def __init__(self, config: Config, parent=None, width=5, height=2, dpi=100):
        """Initialize waveform canvas
        
        Args:
            config: Application configuration
            parent: Parent widget
            width: Width in inches
            height: Height in inches
            dpi: Dots per inch
        """
        self.config = config
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        # Initialize data storage
        self.max_history = config.max_history
        self.audio_data = np.zeros(config.chunk_size)
        self.time_data = np.arange(config.chunk_size) / config.sample_rate
        
        # Initialize plot
        self.waveform_line, = self.axes.plot(self.time_data, self.audio_data, 'b-')
        self.axes.set_ylim(-1.0, 1.0)
        self.axes.set_xlim(0, self.time_data[-1])
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Amplitude')
        self.axes.set_title('Audio Waveform')
        self.axes.grid(True)
    
    def update_plot(self, audio_data: np.ndarray) -> None:
        """Update the waveform plot with new audio data
        
        Args:
            audio_data: New audio data to display
        """
        if len(audio_data) != len(self.audio_data):
            # Resize if needed
            self.audio_data = np.zeros(len(audio_data))
            self.time_data = np.arange(len(audio_data)) / self.config.sample_rate
            self.axes.set_xlim(0, self.time_data[-1])
        
        # Update data
        self.audio_data = audio_data
        self.waveform_line.set_ydata(self.audio_data)
        
        # Redraw the canvas
        self.fig.canvas.draw_idle()


class PitchContourCanvas(FigureCanvas):
    """Canvas for displaying pitch contour"""
    
    def __init__(self, config: Config, parent=None, width=5, height=2, dpi=100):
        """Initialize pitch contour canvas
        
        Args:
            config: Application configuration
            parent: Parent widget
            width: Width in inches
            height: Height in inches
            dpi: Dots per inch
        """
        self.config = config
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        # Initialize data storage
        self.max_history = config.max_history
        self.pitch_data = deque(maxlen=self.max_history)
        self.time_data = deque(maxlen=self.max_history)
        self.current_time = 0
        
        # Initialize plot
        self.pitch_line, = self.axes.plot([], [], 'g-')
        self.axes.set_ylim(50, 500)  # Pitch range in Hz
        self.axes.set_xlim(0, 10)     # 10 seconds of history
        self.axes.set_xlabel('Time (s)')
        self.axes.set_ylabel('Frequency (Hz)')
        self.axes.set_title('Pitch Contour')
        self.axes.set_yscale('log')   # Log scale for pitch
        self.axes.grid(True)
    
    def update_plot(self, result: AnalysisResult) -> None:
        """Update the pitch contour plot with new analysis result
        
        Args:
            result: Analysis result containing pitch data
        """
        if result.pitch_contour is not None and len(result.pitch_contour) > 0:
            # Add mean pitch to history
            self.pitch_data.append(result.pitch_mean)
            self.current_time += result.features.audio.size / result.features.sample_rate
            self.time_data.append(self.current_time)
            
            # Update plot
            x_data = np.array(self.time_data)
            y_data = np.array(self.pitch_data)
            
            # Update line data
            self.pitch_line.set_data(x_data, y_data)
            
            # Adjust x-axis limits to show the most recent data
            if len(x_data) > 0:
                x_min = max(0, x_data[-1] - 10)  # Show last 10 seconds
                x_max = x_data[-1]
                self.axes.set_xlim(x_min, x_max)
            
            # Redraw the canvas
            self.fig.canvas.draw_idle()


class EmotionBarCanvas(FigureCanvas):
    """Canvas for displaying emotion probabilities"""
    
    def __init__(self, config: Config, parent=None, width=5, height=3, dpi=100):
        """Initialize emotion probability canvas
        
        Args:
            config: Application configuration
            parent: Parent widget
            width: Width in inches
            height: Height in inches
            dpi: Dots per inch
        """
        self.config = config
        self.emotion_classes = config.emotion_classes
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        # Initialize data
        self.probabilities = np.zeros(len(self.emotion_classes))
        
        # Create horizontal bar chart
        self.bars = self.axes.barh(
            self.emotion_classes, 
            self.probabilities,
            color='skyblue'
        )
        
        # Set up plot
        self.axes.set_xlim(0, 1)
        self.axes.set_xlabel('Probability')
        self.axes.set_title('Emotion Probabilities')
        self.axes.grid(True, axis='x')
    
    def update_plot(self, result: AnalysisResult) -> None:
        """Update the emotion probability bars
        
        Args:
            result: Analysis result containing emotion scores
        """
        if result.emotion_scores:
            # Extract probabilities for each emotion class
            probabilities = [
                result.emotion_scores.get(emotion, 0) 
                for emotion in self.emotion_classes
            ]
            
            # Update bar heights
            for bar, prob in zip(self.bars, probabilities):
                bar.set_width(prob)
            
            # Highlight the dominant emotion
            for i, emotion in enumerate(self.emotion_classes):
                if emotion == result.dominant_emotion:
                    self.bars[i].set_color('crimson')
                else:
                    self.bars[i].set_color('skyblue')
            
            # Redraw the canvas
            self.fig.canvas.draw_idle()


class SpectrogramCanvas(FigureCanvas):
    """Canvas for displaying audio spectrogram"""
    
    def __init__(self, config: Config, parent=None, width=5, height=3, dpi=100):
        """Initialize spectrogram canvas
        
        Args:
            config: Application configuration
            parent: Parent widget
            width: Width in inches
            height: Height in inches
            dpi: Dots per inch
        """
        self.config = config
        self.fig = Figure(figsize=(width, height), dpi=dpi, tight_layout=True)
        self.axes = self.fig.add_subplot(111)
        super().__init__(self.fig)
        
        # Set up spectrogram parameters
        self.n_fft = 2048
        self.hop_length = config.chunk_size // 4
        
        # Initialize data
        chunk_size = config.chunk_size
        self.audio_buffer = np.zeros(chunk_size * 4)  # Store multiple chunks
        
        # Create initial spectrogram
        self.spec_data = np.zeros((self.n_fft // 2 + 1, 1))
        self.img = self.axes.imshow(
            self.spec_data,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            extent=[0, 1, 0, config.sample_rate / 2 / 1000]  # in kHz
        )
        
        # Set up plot
        self.axes.set_ylabel('Frequency (kHz)')
        self.axes.set_xlabel('Time (s)')
        self.axes.set_title('Spectrogram')
        self.fig.colorbar(self.img, ax=self.axes, label='Power (dB)')
    
    def update_plot(self, audio_data: np.ndarray) -> None:
        """Update the spectrogram with new audio data
        
        Args:
            audio_data: New audio data
        """
        # Update audio buffer with new data (shift old data left)
        buffer_size = self.audio_buffer.size
        data_size = min(audio_data.size, buffer_size)
        self.audio_buffer = np.roll(self.audio_buffer, -data_size)
        self.audio_buffer[-data_size:] = audio_data[-data_size:]
        
        try:
            # Compute spectrogram
            import librosa
            
            # Get spectrogram using short-time Fourier transform
            S = librosa.stft(
                self.audio_buffer, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length
            )
            
            # Convert to power spectrogram
            D = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            
            # Update image data
            self.img.set_array(D)
            self.img.set_extent([0, self.audio_buffer.size / self.config.sample_rate, 0, self.config.sample_rate / 2 / 1000])
            
            # Update color scale if needed
            vmin, vmax = self.img.get_clim()
            new_vmin = max(-80, D.min())  # Cap at -80 dB
            new_vmax = D.max()
            if abs(vmin - new_vmin) > 5 or abs(vmax - new_vmax) > 5:
                self.img.set_clim(new_vmin, new_vmax)
            
            # Redraw the canvas
            self.fig.canvas.draw_idle()
        
        except Exception as e:
            logger.error(f"Error updating spectrogram: {e}")


class Visualizer:
    """Manages visualization components for tonal analysis"""
    
    def __init__(self, config: Config):
        """Initialize visualizer
        
        Args:
            config: Application configuration
        """
        self.config = config
        
        # Store canvases created by the GUI
        self.canvases = {}
        
        # Update timer
        self.update_timer = None
        self.update_interval = config.plot_update_interval
    
    def create_canvas(self, canvas_type: str, parent=None, **kwargs) -> FigureCanvas:
        """Create and return a visualization canvas
        
        Args:
            canvas_type: Type of canvas to create 
                         (waveform, pitch, emotion, spectrogram)
            parent: Parent widget
            **kwargs: Additional arguments for canvas constructor
            
        Returns:
            Visualization canvas
        """
        if canvas_type == "waveform":
            canvas = AudioWaveformCanvas(self.config, parent, **kwargs)
        elif canvas_type == "pitch":
            canvas = PitchContourCanvas(self.config, parent, **kwargs)
        elif canvas_type == "emotion":
            canvas = EmotionBarCanvas(self.config, parent, **kwargs)
        elif canvas_type == "spectrogram":
            canvas = SpectrogramCanvas(self.config, parent, **kwargs)
        else:
            logger.error(f"Unknown canvas type: {canvas_type}")
            return None
        
        # Store canvas
        self.canvases[canvas_type] = canvas
        
        return canvas
    
    def start_updates(self, parent) -> None:
        """Start periodic updates of visualizations
        
        Args:
            parent: Parent QObject for the timer
        """
        if self.update_timer is None:
            self.update_timer = QTimer(parent)
            self.update_timer.timeout.connect(self._update_timer_callback)
            self.update_timer.start(self.update_interval)
            logger.info(f"Started visualization updates every {self.update_interval} ms")
    
    def stop_updates(self) -> None:
        """Stop periodic updates"""
        if self.update_timer is not None:
            self.update_timer.stop()
            self.update_timer = None
            logger.info("Stopped visualization updates")
    
    def _update_timer_callback(self) -> None:
        """Called by the update timer to refresh visualizations"""
        # This is empty because updates are handled by callbacks
        # from audio capture, feature extraction, and analysis
        pass
    
    def update_audio_waveform(self, audio_data: np.ndarray) -> None:
        """Update audio waveform visualization
        
        Args:
            audio_data: Audio data to visualize
        """
        canvas = self.canvases.get("waveform")
        if canvas is not None:
            canvas.update_plot(audio_data)
    
    def update_pitch_contour(self, result: AnalysisResult) -> None:
        """Update pitch contour visualization
        
        Args:
            result: Analysis result to visualize
        """
        canvas = self.canvases.get("pitch")
        if canvas is not None:
            canvas.update_plot(result)
    
    def update_emotion_bars(self, result: AnalysisResult) -> None:
        """Update emotion probability bars
        
        Args:
            result: Analysis result to visualize
        """
        canvas = self.canvases.get("emotion")
        if canvas is not None:
            canvas.update_plot(result)
    
    def update_spectrogram(self, audio_data: np.ndarray) -> None:
        """Update spectrogram visualization
        
        Args:
            audio_data: Audio data to visualize
        """
        canvas = self.canvases.get("spectrogram")
        if canvas is not None:
            canvas.update_plot(audio_data) 
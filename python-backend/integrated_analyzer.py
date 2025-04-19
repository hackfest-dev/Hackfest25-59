import cv2
import numpy as np
import mediapipe as mp
from deepface import DeepFace
import time
import datetime
import csv
import os
import threading
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pandas as pd
import math
from scipy.signal import savgol_filter
import warnings
import json
import platform
import sounddevice as sd
import queue
import librosa
import parselmouth
warnings.filterwarnings("ignore")

# Constants for readability and easy tuning
EYE_AR_CONSEC_FRAMES_CLOSED = 2
EYE_AR_CONSEC_FRAMES_OPEN = 1
BLINK_COOLDOWN = 0.1  # Seconds to wait before detecting another blink
CALIBRATION_FRAMES = 30
PLOT_WIDTH = 200
PLOT_HEIGHT = 120

class IntegratedFeatureAnalyzer:
    def __init__(self, log_data=False, log_interval=5, camera_id=0, resolution=(640, 480), enable_emotion=True):
        # Initialize face mesh for facial landmarks with refined landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_landmarks=True  # Enable refined landmarks for better eye detection
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection for behavior analysis
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=1,  # Increased from 0 to 1 for better accuracy 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Configuration options
        self.camera_id = camera_id
        self.resolution = resolution
        self.enable_emotion = enable_emotion
        
        # Emotion analysis parameters
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.last_emotion_time = time.time()
        self.emotion_cooldown = 0.5  # Reduce cooldown for more frequent updates
        self.current_emotion = "Unknown"
        self.emotion_history = {emotion: 0 for emotion in self.emotion_labels}
        self.emotion_scores = {emotion: 0.0 for emotion in self.emotion_labels}
        self.dominant_emotion_history = deque(maxlen=10)  # Track recent emotions
        self.emotion_detected = False  # Flag to track if emotion was detected at least once
        
        # Initialize session statistics
        self.session_stats = {
            'emotion_counts': {emotion: 0 for emotion in self.emotion_labels},
            'attention_states': {},
            'posture_states': {},
            'movement_levels': {},
            'posture_issues': 0,
            'fatigue_incidents': 0
        }
        
        # Behavior analysis parameters
        self.posture_state = "Unknown"
        self.posture_history = deque(maxlen=30)  # Track posture over time
        self.movement_level = "Unknown"
        self.attention_state = "Unknown"
        self.attention_history = deque(maxlen=60)  # Track attention over time
        self.fatigue_level = "Normal"  # New fatigue tracking
        
        # Movement tracking
        self.movement_history = deque(maxlen=30)  # Store last 30 frames of movement data
        self.movement_magnitude = deque(maxlen=60)  # Store magnitudes for trend analysis
        
        # Robust blink detection parameters
        self.blink_count = 0
        self.total_blink_count = 0
        self.last_reset_time = time.time()
        self.last_blink_time = time.time()
        self.eye_closed = False
        self.blink_rate_history = deque(maxlen=10)  # Store recent blink rates
        self.blink_timestamps = deque(maxlen=100)  # Store timestamps of blinks for interval analysis
        
        # Frame counters for state confirmation
        self.eye_closed_frames = 0
        self.eye_closed_frames_threshold = EYE_AR_CONSEC_FRAMES_CLOSED  # Number of frames eye must be closed
        self.eye_open_frames = 0
        self.eye_open_frames_threshold = EYE_AR_CONSEC_FRAMES_OPEN  # Number of frames eye must be open
        
        # Define the eye landmark indices - these are key points for eye aspect ratio
        # Left eye
        self.left_eye_landmarks = {
            'top': 159,  # Upper eyelid
            'bottom': 145,  # Lower eyelid
            'left': 33,  # Left corner
            'right': 133,  # Right corner
            # Additional points for better detection
            'top_inner': 158,
            'top_outer': 160,
            'bottom_inner': 144,
            'bottom_outer': 153
        }
        
        # Right eye
        self.right_eye_landmarks = {
            'top': 386,  # Upper eyelid
            'bottom': 374,  # Lower eyelid
            'left': 362,  # Left corner
            'right': 263,  # Right corner
            # Additional points for better detection
            'top_inner': 385,
            'top_outer': 387,
            'bottom_inner': 373,
            'bottom_outer': 380
        }
        
        # Tracking and adaptive calibration for EAR
        self.ear_values = deque(maxlen=300)  # Store more EAR values for better statistics
        self.ear_baseline = deque(maxlen=50)  # Baseline for open eyes
        self.ear_closed = deque(maxlen=20)   # Values when eyes are detected as closed
        self.min_ear = 1.0
        self.max_ear = 0.0
        self.adaptive_threshold = 0.2  # Starting value, will be adjusted
        
        # Left/right eye values separately for asymmetry detection
        self.left_ear_values = deque(maxlen=100)
        self.right_ear_values = deque(maxlen=100)
        self.eye_asymmetry = 0.0  # Track eye asymmetry
        
        # Face distance normalization
        self.face_size_history = deque(maxlen=30)  # To track face size changes
        self.normalized_ear_values = deque(maxlen=100)  # EAR values normalized by face size
        
        # Continuous calibration
        self.calibration_counter = 0
        self.recalibration_interval = 100  # Frames between recalibrations
        self.calibration_samples = []
        self.is_calibrated = False
        
        # Moving average for ear
        self.ear_moving_avg = deque(maxlen=5)
        
        # Session analysis data
        self.session_start_time = time.time()
        
        # Advanced metrics
        self.blink_duration = deque(maxlen=30)  # Track duration of blinks
        self.blink_intervals = deque(maxlen=30)  # Time between blinks
        self.max_blink_rate = 0  # Maximum recorded blink rate
        self.min_blink_rate = float('inf')  # Minimum recorded blink rate
        
        # Fatigue detection parameters
        self.fatigue_threshold = 30  # Blinks per minute above which is considered fatigue
        self.drowsiness_eye_ratio = 0.24  # Low threshold for drowsiness detection
        self.drowsiness_frames = 0
        self.drowsiness_alerts = 0
        
        # Visualization parameters
        self.show_landmarks = False  # Toggle for showing all landmarks
        self.show_graphs = True  # Toggle for showing analytics graphs
        self.graph_data = {}  # Store data for graphs
        
        # Video recording capability
        self.recording = False
        self.video_writer = None
        self.recording_start_time = None
        
        # Advanced visualization
        self.active_tab = 0  # For UI tabs (0: main, 1: blinks, 2: emotion, 3: posture)
        self.plot_surfaces = {}  # Store plot surfaces for efficiency
        self.last_plot_update = {}  # Track when each plot was last updated
        
        # Performance metrics
        self.fps_history = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.current_fps = 0
        
        # Logging
        self.log_data = log_data
        self.log_interval = log_interval
        self.last_log_time = time.time()
        if log_data:
            self._setup_logging()
        
        # Start background emotion processing thread if enabled
        self.emotion_queue = deque(maxlen=5)
        self.emotion_result = None
        self.emotion_thread_active = True and enable_emotion
        
        if enable_emotion:
            self.emotion_thread = threading.Thread(target=self._process_emotions)
            self.emotion_thread.daemon = True
            self.emotion_thread.start()
            
        # Start background audio processing thread
        self.audio_queue = queue.Queue()
        self.voice_metrics = {}
        self.audio_sampling_rate = 16000
        self.audio_stream = sd.InputStream(callback=self._audio_callback, channels=1, samplerate=self.audio_sampling_rate)
        self.audio_stream.start()
        self.audio_thread_active = True
        self.audio_thread = threading.Thread(target=self._process_audio)
        self.audio_thread.daemon = True
        self.audio_thread.start()
        
        # Debug flag
        self.debug = False  # Set to True to enable debug visualizations

    def _setup_logging(self):
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        # Create subdirectories for different types of logs
        for subdir in ['csv', 'summary', 'raw_data', 'sessions']:
            if not os.path.exists(f'logs/{subdir}'):
                os.makedirs(f'logs/{subdir}')
        
        # Create timestamp for file naming
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create standard log files
        self.log_file = f"logs/csv/integrated_analysis_{timestamp}.csv"
        self.summary_file = f"logs/summary/session_summary_{timestamp}.txt"
        
        # Create raw data log file for complete frame-by-frame data
        self.raw_data_file = f"logs/raw_data/raw_data_{timestamp}.csv"
        
        # Create session info file
        self.session_info_file = f"logs/sessions/session_info_{timestamp}.json"
        
        # Initialize CSV file with headers
        with open(self.log_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Timestamp', 'Emotion', 'Attention', 'Posture', 'Movement',
                'Blink_Rate', 'Total_Blinks', 'EAR', 'EAR_Threshold',
                'Eye_Asymmetry', 'Blink_Duration', 'Blink_Interval',
                'Fatigue_Level', 'Drowsiness_Score',
                'Angry_Score', 'Disgust_Score', 'Fear_Score', 
                'Happy_Score', 'Sad_Score', 'Surprise_Score', 'Neutral_Score',
                'F0_Mean', 'Intensity', 'Jitter', 'Shimmer', 'Spectral_Centroid', 'Spectral_Rolloff', 'Zero_Crossing_Rate', 'Spectral_Bandwidth',
                'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13'
            ])
        
        # Initialize raw data log file with more detailed headers
        with open(self.raw_data_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'Frame_Number', 'Timestamp', 'FPS', 'Emotion', 'Attention', 'Posture', 'Movement',
                'EAR', 'Left_EAR', 'Right_EAR', 'EAR_Threshold', 'Eye_State', 'Blink_Count',
                'Horizontal_Angle', 'Vertical_Angle', 'Distance_From_Center',
                'Head_Movement', 'Torso_Movement', 'Left_Arm_Movement', 'Right_Arm_Movement',
                'Fatigue_Level', 'Drowsiness_Score',
                'Angry_Score', 'Disgust_Score', 'Fear_Score', 
                'Happy_Score', 'Sad_Score', 'Surprise_Score', 'Neutral_Score',
                'F0_Mean', 'Intensity', 'Jitter', 'Shimmer', 'Spectral_Centroid', 'Spectral_Rolloff', 'Zero_Crossing_Rate', 'Spectral_Bandwidth',
                'MFCC_1', 'MFCC_2', 'MFCC_3', 'MFCC_4', 'MFCC_5', 'MFCC_6', 'MFCC_7', 'MFCC_8', 'MFCC_9', 'MFCC_10', 'MFCC_11', 'MFCC_12', 'MFCC_13',
                'Processing_Time'
            ])
        
        # Initialize session stats file
        with open(self.summary_file, 'w') as file:
            file.write(f"Session started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            file.write(f"Logging interval: {self.log_interval} seconds\n")
            file.write(f"Emotion detection: {'Enabled' if self.enable_emotion else 'Disabled'}\n\n")
            
        # Initialize frame counter for raw data logging
        self.frame_counter = 0
        
        # Set up backup mechanism to save logs on crash
        import atexit
        import signal
        
        # Register function to call on normal exit
        atexit.register(self._save_backup_logs)
        
        # Register signals for abnormal termination
        signal.signal(signal.SIGINT, self._handle_signal)   # Ctrl+C
        signal.signal(signal.SIGTERM, self._handle_signal)  # Termination request
        
        # Try to register SIGBREAK on Windows for Ctrl+Break
        try:
            signal.signal(signal.SIGBREAK, self._handle_signal)
        except AttributeError:
            # SIGBREAK not available on this platform
            pass
    
    def _handle_signal(self, sig, frame):
        """Handle termination signals by saving backup logs before exit"""
        print(f"\nReceived termination signal {sig}. Saving logs before exit...")
        self._save_backup_logs()
        import sys
        sys.exit(0)
    
    def _save_backup_logs(self):
        """Save backup of all log data in case of unexpected termination"""
        try:
            if self.log_data:
                print("\nSaving backup logs before exit...")
                
                # Only proceed if we have some data
                if self.frame_counter > 0:
                    # Create backup tag
                    backup_tag = "_BACKUP"
                    
                    # Calculate session statistics for the backup
                    session_duration_seconds = time.time() - self.session_start_time
                    
                    # Create backup summary text file
                    backup_summary_file = self.summary_file.replace('.txt', f'{backup_tag}.txt')
                    
                    with open(backup_summary_file, 'w') as file:
                        file.write(f"BACKUP LOG - Session interrupted at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        file.write(f"Partial session duration: {session_duration_seconds/60:.1f} minutes\n")
                        file.write(f"Frames processed: {self.frame_counter}\n")
                        file.write(f"Total blinks detected: {self.total_blink_count}\n")
                        file.write(f"Average blink rate: {self.total_blink_count / max(1, session_duration_seconds / 60):.1f} bpm\n\n")
                        
                        # Add basic analytics
                        file.write("--- PARTIAL SESSION STATS ---\n")
                        
                        # Attention stats
                        file.write("\nAttention States:\n")
                        for state, count in self.session_stats['attention_states'].items():
                            file.write(f"  {state}: {count} frames\n")
                            
                        # Posture stats  
                        file.write("\nPosture States:\n")
                        for state, count in self.session_stats['posture_states'].items():
                            file.write(f"  {state}: {count} frames\n")
                            
                        # Emotion stats
                        if self.enable_emotion:
                            file.write("\nEmotion Distribution:\n")
                            for emotion, count in self.session_stats['emotion_counts'].items():
                                if count > 0:
                                    file.write(f"  {emotion}: {count} detections\n")
                    
                    # Create backup JSON with all available data
                    backup_json_file = self.session_info_file.replace('.json', f'{backup_tag}.json')
                    
                    backup_data = {
                        "backup_info": {
                            "created_at": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            "reason": "Session interrupted or terminated unexpectedly",
                            "frames_processed": self.frame_counter
                        },
                        "session_data": {
                            "duration_seconds": session_duration_seconds,
                            "blinks": self.total_blink_count,
                            "blink_rate": self.total_blink_count / max(1, session_duration_seconds / 60),
                        },
                        "attention_states": self.session_stats['attention_states'],
                        "posture_states": self.session_stats['posture_states'],
                        "emotion_counts": self.session_stats['emotion_counts'],
                        "performance": {
                            "average_fps": sum(self.fps_history) / max(1, len(self.fps_history)),
                            "average_processing_time": sum(self.processing_times) / max(1, len(self.processing_times))
                        }
                    }
                    
                    with open(backup_json_file, 'w') as file:
                        json.dump(backup_data, file, indent=4)
                    
                    print(f"Backup logs saved successfully:")
                    print(f"  Backup summary: {backup_summary_file}")
                    print(f"  Backup data: {backup_json_file}")
                else:
                    print("No data to save in backup logs.")
        except Exception as e:
            print(f"Error saving backup logs: {e}")
            import traceback
            traceback.print_exc()
    
    def _process_emotions(self):
        """Background thread to process emotions without blocking the main loop"""
        print("Emotion detection thread started successfully")
        while self.emotion_thread_active:
            if len(self.emotion_queue) > 0:
                try:
                    frame = self.emotion_queue.popleft()
                    if frame is None or frame.size == 0:
                        print("Empty frame received for emotion analysis, skipping...")
                        continue
                        
                    # Add additional check for frame dimensions
                    if len(frame.shape) != 3 or frame.shape[2] != 3:
                        print(f"Invalid frame format for emotion analysis: {frame.shape}, skipping...")
                        continue
                        
                    print("Processing emotion for frame with shape:", frame.shape)
                    
                    # Use ssd as primary backend - faster and more reliable than retinaface
                    emotion_analysis = DeepFace.analyze(
                        frame, 
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='ssd',  # Faster and more reliable than retinaface
                        silent=True
                    )
                    
                    if isinstance(emotion_analysis, list) and len(emotion_analysis) > 0:
                        self.emotion_result = emotion_analysis[0]
                        self.emotion_detected = True
                        
                        # Normalize emotion scores to ensure they're between 0-1
                        total_score = sum(self.emotion_result['emotion'].values())
                        if total_score > 0:
                            # Ensure all scores are properly normalized
                            for emotion in self.emotion_labels:
                                raw_score = self.emotion_result['emotion'].get(emotion, 0)
                                # Normalize the score and ensure it's between 0-1
                                self.emotion_scores[emotion] = min(1.0, raw_score / max(1.0, total_score))
                        
                        print(f"Emotion detected: {self.emotion_result['dominant_emotion']}")
                except Exception as e:
                    print(f"Primary emotion analysis error: {e}")
                    # Try alternate backend if first fails
                    try:
                        frame = self.emotion_queue[-1] if self.emotion_queue else None
                        if frame is not None:
                            print("Trying fallback emotion detection with OpenCV backend")
                            emotion_analysis = DeepFace.analyze(
                                frame, 
                                actions=['emotion'],
                                enforce_detection=False,
                                detector_backend='opencv',  # Fallback to faster method
                                silent=True
                            )
                            
                            if isinstance(emotion_analysis, list) and len(emotion_analysis) > 0:
                                self.emotion_result = emotion_analysis[0]
                                self.emotion_detected = True
                                print(f"Fallback emotion detected: {self.emotion_result['dominant_emotion']}")
                    except Exception as fallback_err:
                        print(f"Fallback emotion analysis error: {fallback_err}")
            
            # Sleep to prevent CPU overuse
            time.sleep(0.05)  # Reduced sleep time to process more frames
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream"""
        if status:
            print(f"Audio stream status: {status}")
        audio_data = indata[:, 0].copy()
        self.audio_queue.put(audio_data)

    def _process_audio(self):
        """Background thread to process audio and extract voice features"""
        buffer = np.array([], dtype=np.float32)
        window_size = int(self.audio_sampling_rate * 0.5)  # 0.5 second window
        while self.audio_thread_active:
            try:
                data = self.audio_queue.get(timeout=1)
                buffer = np.concatenate((buffer, data))
                if len(buffer) >= window_size:
                    segment = buffer[:window_size]
                    buffer = buffer[window_size:]
                    # Parselmouth for pitch and intensity
                    sound = parselmouth.Sound(segment, self.audio_sampling_rate)
                    pitch = sound.to_pitch()
                    f0_vals = pitch.selected_array['frequency']
                    f0_mean = np.mean(f0_vals[f0_vals>0]) if len(f0_vals[f0_vals>0])>0 else 0
                    intensity = sound.to_intensity().values[0]
                    # Jitter and shimmer
                    point_cc = sound.to_point_process()
                    jitter_local = parselmouth.praat.call(point_cc, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3)
                    shimmer_local = parselmouth.praat.call([sound, point_cc], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6)
                    # Librosa spectral features
                    spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=self.audio_sampling_rate).mean()
                    spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=self.audio_sampling_rate).mean()
                    zcr = librosa.feature.zero_crossing_rate(y=segment).mean()
                    bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=self.audio_sampling_rate).mean()
                    mfccs = librosa.feature.mfcc(y=segment, sr=self.audio_sampling_rate, n_mfcc=13)
                    mfcc_means = np.mean(mfccs, axis=1)
                    # Update metrics
                    self.voice_metrics = {
                        'f0_mean': f0_mean,
                        'intensity': intensity,
                        'jitter': jitter_local,
                        'shimmer': shimmer_local,
                        'spectral_centroid': spectral_centroid,
                        'spectral_rolloff': spectral_rolloff,
                        'zcr': zcr,
                        'bandwidth': bandwidth
                    }
                    for i, mfcc in enumerate(mfcc_means, start=1):
                        self.voice_metrics[f'mfcc_{i}'] = mfcc
            except queue.Empty:
                continue
    
    def _log_data_point(self, ear=0, threshold=0):
        """Log current analysis data to CSV file"""
        current_time = time.time()
        
        if self.log_data and (current_time - self.last_log_time) > self.log_interval:
            try:
                time_since_reset = max(1, (current_time - self.last_reset_time))
                blink_rate = self.blink_count / time_since_reset * 60.0
                
                # Calculate additional metrics
                avg_blink_duration = sum(self.blink_duration) / max(1, len(self.blink_duration))
                avg_blink_interval = sum(self.blink_intervals) / max(1, len(self.blink_intervals))
                
                # Update session stats
                self.session_stats['total_blinks'] = self.total_blink_count
                self.blink_rate_history.append(blink_rate)
                self.session_stats['avg_blink_rate'] = sum(self.blink_rate_history) / max(1, len(self.blink_rate_history))
                
                # Handle max/min
                if blink_rate > self.max_blink_rate:
                    self.max_blink_rate = blink_rate
                if blink_rate < self.min_blink_rate and blink_rate > 0:
                    self.min_blink_rate = blink_rate
                
                # Calculate eye asymmetry score (0-1, higher means more asymmetry)
                if len(self.left_ear_values) > 5 and len(self.right_ear_values) > 5:
                    left_avg = sum(self.left_ear_values) / len(self.left_ear_values)
                    right_avg = sum(self.right_ear_values) / len(self.right_ear_values)
                    self.eye_asymmetry = abs(left_avg - right_avg) / max((left_avg + right_avg) / 2, 0.001)
                
                # Write to CSV
                with open(self.log_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        self.current_emotion,
                        self.attention_state,
                        self.posture_state,
                        self.movement_level,
                        f"{blink_rate:.1f}",
                        self.total_blink_count,
                        f"{ear:.4f}",
                        f"{threshold:.4f}",
                        f"{self.eye_asymmetry:.4f}",
                        f"{avg_blink_duration:.4f}",
                        f"{avg_blink_interval:.4f}",
                        self.fatigue_level,
                        f"{self.drowsiness_frames / max(1, len(self.ear_values)):.4f}",
                        self.emotion_scores.get('angry', 0),
                        self.emotion_scores.get('disgust', 0),
                        self.emotion_scores.get('fear', 0),
                        self.emotion_scores.get('happy', 0),
                        self.emotion_scores.get('sad', 0),
                        self.emotion_scores.get('surprise', 0),
                        self.emotion_scores.get('neutral', 0),
                        self.voice_metrics.get('f0_mean', 0),
                        self.voice_metrics.get('intensity', 0),
                        self.voice_metrics.get('jitter', 0),
                        self.voice_metrics.get('shimmer', 0),
                        self.voice_metrics.get('spectral_centroid', 0),
                        self.voice_metrics.get('spectral_rolloff', 0),
                        self.voice_metrics.get('zcr', 0),
                        self.voice_metrics.get('bandwidth', 0),
                        self.voice_metrics.get('mfcc_1', 0),
                        self.voice_metrics.get('mfcc_2', 0),
                        self.voice_metrics.get('mfcc_3', 0),
                        self.voice_metrics.get('mfcc_4', 0),
                        self.voice_metrics.get('mfcc_5', 0),
                        self.voice_metrics.get('mfcc_6', 0),
                        self.voice_metrics.get('mfcc_7', 0),
                        self.voice_metrics.get('mfcc_8', 0),
                        self.voice_metrics.get('mfcc_9', 0),
                        self.voice_metrics.get('mfcc_10', 0),
                        self.voice_metrics.get('mfcc_11', 0),
                        self.voice_metrics.get('mfcc_12', 0),
                        self.voice_metrics.get('mfcc_13', 0)
                    ])
                    
                # Reset blink counter for rate calculation
                self.blink_count = 0
                self.last_reset_time = current_time
                self.last_log_time = current_time
                
                # Generate periodic charts if enabled
                if self.show_graphs:
                    self._generate_analytics_report()
                
            except Exception as e:
                print(f"Error logging data: {e}")
                
    def _log_raw_data(self, frame, ear, threshold, process_time):
        """Log detailed data for every frame"""
        if not hasattr(self, 'raw_data_file') or not self.log_data:
            return
        
        try:
            # Increment frame counter
            self.frame_counter += 1
            
            # Get current timestamp
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            # Get default values for movement metrics
            head_movement = 0
            torso_movement = 0
            left_arm_movement = 0
            right_arm_movement = 0
            
            # Extract movement data if available
            if hasattr(self, 'movement_details'):
                head_movement = self.movement_details.get('head', 0)
                torso_movement = self.movement_details.get('torso', 0)
                left_arm_movement = self.movement_details.get('left_arm', 0)
                right_arm_movement = self.movement_details.get('right_arm', 0)
            
            # Get face angles if available
            horizontal_angle = getattr(self, 'face_horizontal_angle', 0)
            vertical_angle = getattr(self, 'face_vertical_angle', 0)
            
            # Get distance from center (for attention analysis)
            distance_from_center = 0
            if hasattr(self, 'attention_calculated_state'):
                h, w, _ = frame.shape
                # Try to get nose position if landmarks were detected
                if hasattr(self, 'last_detected_nose'):
                    nose_x, nose_y = self.last_detected_nose
                    frame_center_x = w // 2
                    distance_from_center = abs(nose_x - frame_center_x) / (w * 0.5)
            
            # Get current eye state
            eye_state = "Closed" if self.eye_closed else "Open"
            
            # Get left and right EAR values if available
            left_ear = self.left_ear_values[-1] if self.left_ear_values else 0
            right_ear = self.right_ear_values[-1] if self.right_ear_values else 0
            
            # Write data to CSV
            with open(self.raw_data_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    self.frame_counter,
                    timestamp,
                    f"{self.current_fps:.1f}",
                    self.current_emotion,
                    self.attention_state,
                    self.posture_state,
                    self.movement_level,
                    f"{ear:.4f}",
                    f"{left_ear:.4f}",
                    f"{right_ear:.4f}",
                    f"{threshold:.4f}",
                    eye_state,
                    self.blink_count,
                    f"{horizontal_angle:.2f}",
                    f"{vertical_angle:.2f}",
                    f"{distance_from_center:.4f}",
                    f"{head_movement:.6f}",
                    f"{torso_movement:.6f}",
                    f"{left_arm_movement:.6f}",
                    f"{right_arm_movement:.6f}",
                    self.fatigue_level,
                    f"{self.drowsiness_frames / max(1, len(self.ear_values)):.4f}",
                    f"{self.emotion_scores.get('angry', 0):.4f}",
                    f"{self.emotion_scores.get('disgust', 0):.4f}",
                    f"{self.emotion_scores.get('fear', 0):.4f}",
                    f"{self.emotion_scores.get('happy', 0):.4f}",
                    f"{self.emotion_scores.get('sad', 0):.4f}",
                    f"{self.emotion_scores.get('surprise', 0):.4f}",
                    f"{self.emotion_scores.get('neutral', 0):.4f}",
                    f"{self.voice_metrics.get('f0_mean', 0):.4f}",
                    f"{self.voice_metrics.get('intensity', 0):.4f}",
                    f"{self.voice_metrics.get('jitter', 0):.4f}",
                    f"{self.voice_metrics.get('shimmer', 0):.4f}",
                    f"{self.voice_metrics.get('spectral_centroid', 0):.4f}",
                    f"{self.voice_metrics.get('spectral_rolloff', 0):.4f}",
                    f"{self.voice_metrics.get('zcr', 0):.4f}",
                    f"{self.voice_metrics.get('bandwidth', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_1', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_2', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_3', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_4', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_5', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_6', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_7', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_8', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_9', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_10', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_11', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_12', 0):.4f}",
                    f"{self.voice_metrics.get('mfcc_13', 0):.4f}",
                    f"{process_time:.4f}"
                ])
        except Exception as e:
            print(f"Error logging raw data: {e}")
    
    def analyze_frame(self, frame):
        """Main function to analyze a frame and extract all metrics"""
        # Start performance timing
        process_start = time.time()
        
        # Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if dt > 0:
            fps = 1 / dt
            self.fps_history.append(fps)
            self.current_fps = sum(self.fps_history) / len(self.fps_history)
        
        # Create copies for different analyses
        emotion_frame = None
        display_frame = frame.copy()
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with face mesh
        face_results = self.face_mesh.process(rgb_frame)
        
        # Process with pose detection
        pose_results = self.pose.process(rgb_frame)
        
        ear = 0
        threshold = 0
        
        # Process face landmarks if detected
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Draw face mesh landmarks if in debug mode
                if self.debug or self.show_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=display_frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                
                # Store nose position for attention logging
                nose_tip = face_landmarks.landmark[4]  # Nose tip
                h, w, _ = frame.shape
                self.last_detected_nose = (int(nose_tip.x * w), int(nose_tip.y * h))
                
                # Prepare emotion frame only when needed - more frequent updates for emotions
                if self.enable_emotion and (current_time - self.last_emotion_time > self.emotion_cooldown / 2):
                    # Extract face region for more efficient emotion processing
                    face_region = self._extract_face_region(frame, face_landmarks, expand_ratio=1.5)
                    if face_region is not None:
                        emotion_frame = face_region
                
                # Detect blinks with robust implementation
                ear, threshold = self.detect_blinks(face_landmarks, display_frame)
                
                # Analyze attention based on face orientation
                self.analyze_attention(face_landmarks, display_frame)
                
                # Analyze fatigue based on blink patterns and eye openness
                self.analyze_fatigue(ear, threshold)
        
        # Draw pose landmarks and analyze body language
        if pose_results.pose_landmarks:
            if self.debug or self.show_landmarks:
                self.mp_drawing.draw_landmarks(
                    display_frame,
                    pose_results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
                )
            
            # Analyze posture and movement
            self.analyze_posture(pose_results.pose_landmarks, display_frame)
            self.analyze_movement(pose_results.pose_landmarks, display_frame)
        
        # Queue frame for emotion analysis in background thread - more aggressive updates
        if self.enable_emotion and emotion_frame is not None and (current_time - self.last_emotion_time > self.emotion_cooldown / 2):
            self.emotion_queue.append(emotion_frame)
            self.last_emotion_time = current_time
            
        # Update emotion from background thread results
        if self.emotion_result:
            self.current_emotion = self.emotion_result['dominant_emotion']
            self.dominant_emotion_history.append(self.current_emotion)
            
            # Update emotion history counts
            self.emotion_history[self.current_emotion] += 1
            self.session_stats['emotion_counts'][self.current_emotion] += 1
            
            # Update emotion scores with proper normalization
            total_score = sum(self.emotion_result['emotion'].values())
            if total_score > 0:
                for emotion in self.emotion_labels:
                    if emotion in self.emotion_result['emotion']:
                        raw_score = self.emotion_result['emotion'][emotion]
                        # Normalize the score and ensure it's between 0 and 1
                        self.emotion_scores[emotion] = min(1.0, raw_score / max(1.0, total_score))
            
            # Clear the result to avoid processing it multiple times
            self.emotion_result = None
        
        # Log data periodically
        self._log_data_point(ear, threshold)
        
        # Update blink rate display
        time_since_reset = max(1, (current_time - self.last_reset_time))
        blink_rate = self.blink_count / time_since_reset * 60.0
        
        # Every 60 seconds, reset the counter for fresh rate calculation in display
        if time_since_reset >= 60:
            self.blink_count = 0
            self.last_reset_time = current_time
        
        # Update processing time metrics
        process_end = time.time()
        process_time = process_end - process_start
        self.processing_times.append(process_time)
        
        # Log raw data for this frame
        if self.log_data:
            self._log_raw_data(frame, ear, threshold, process_time)
        
        # Display results on frame
        self.display_results(display_frame, blink_rate, ear, threshold)
        
        # Handle video recording if active
        if self.recording and self.video_writer is not None:
            self.video_writer.write(display_frame)
        
        return display_frame
    
    def _extract_face_region(self, frame, face_landmarks, expand_ratio=1.2):
        """Extract the face region from the frame using landmarks for more efficient processing"""
        try:
            h, w, _ = frame.shape
            
            # Get the bounding box of the face
            x_min = w
            y_min = h
            x_max = 0
            y_max = 0
            
            for landmark in face_landmarks.landmark:
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                x_min = min(x_min, px)
                y_min = min(y_min, py)
                x_max = max(x_max, px)
                y_max = max(y_max, py)
            
            # Expand the bounding box
            width = x_max - x_min
            height = y_max - y_min
            
            x_min = max(0, int(x_min - width * (expand_ratio - 1) / 2))
            y_min = max(0, int(y_min - height * (expand_ratio - 1) / 2))
            x_max = min(w, int(x_max + width * (expand_ratio - 1) / 2))
            y_max = min(h, int(y_max + height * (expand_ratio - 1) / 2))
            
            # Extract the face region
            face_region = frame[y_min:y_max, x_min:x_max]
            
            # Ensure we have a valid region
            if face_region.shape[0] > 0 and face_region.shape[1] > 0:
                return face_region
                
        except Exception as e:
            if self.debug:
                print(f"Error extracting face region: {e}")
        
        return None
        
    def calculate_ear(self, landmarks, eye_points, frame=None, draw=False):
        """Calculate eye aspect ratio with multiple points for better accuracy"""
        h, w = None, None
        if frame is not None and draw:
            h, w, _ = frame.shape
        
        # Get primary points
        top = landmarks.landmark[eye_points['top']]
        bottom = landmarks.landmark[eye_points['bottom']]
        left = landmarks.landmark[eye_points['left']]
        right = landmarks.landmark[eye_points['right']]
        
        # Get additional points for better height measurement
        top_inner = landmarks.landmark[eye_points['top_inner']]
        top_outer = landmarks.landmark[eye_points['top_outer']]
        bottom_inner = landmarks.landmark[eye_points['bottom_inner']]
        bottom_outer = landmarks.landmark[eye_points['bottom_outer']]
        
        # Draw points if debug is enabled
        if frame is not None and draw:
            # Main points
            cv2.circle(frame, (int(top.x * w), int(top.y * h)), 2, (0, 255, 0), -1)
            cv2.circle(frame, (int(bottom.x * w), int(bottom.y * h)), 2, (0, 255, 0), -1)
            cv2.circle(frame, (int(left.x * w), int(left.y * h)), 2, (255, 0, 0), -1)
            cv2.circle(frame, (int(right.x * w), int(right.y * h)), 2, (255, 0, 0), -1)
            
            # Additional points
            cv2.circle(frame, (int(top_inner.x * w), int(top_inner.y * h)), 1, (0, 255, 255), -1)
            cv2.circle(frame, (int(top_outer.x * w), int(top_outer.y * h)), 1, (0, 255, 255), -1)
            cv2.circle(frame, (int(bottom_inner.x * w), int(bottom_inner.y * h)), 1, (0, 255, 255), -1)
            cv2.circle(frame, (int(bottom_outer.x * w), int(bottom_outer.y * h)), 1, (0, 255, 255), -1)
        
        # Calculate multiple height measurements for robustness
        h1 = abs(top.y - bottom.y)
        h2 = abs(top_inner.y - bottom_inner.y)
        h3 = abs(top_outer.y - bottom_outer.y)
        
        # Use average height
        height = (h1 + h2 + h3) / 3.0
        
        # Calculate width
        width = abs(right.x - left.x)
        
        # Enhanced EAR calculation with normalization factor
        # A simple EAR is height/width, but we can make it more robust
        ear = height / max(width, 0.001)  # Avoid division by zero
        
        # Apply smoothing to reduce noise
        if hasattr(self, 'prev_ear') and self.prev_ear is not None:
            # Exponential smoothing
            alpha = 0.3  # Smoothing factor (0-1)
            ear = alpha * ear + (1 - alpha) * self.prev_ear
        
        self.prev_ear = ear
        
        return ear, width  # Return width for face size normalization
    
    def detect_blinks(self, face_landmarks, frame):
        """Enhanced blink detection with multiple validation steps and adaptive thresholds"""
        try:
            h, w, _ = frame.shape
            
            # Calculate EAR for both eyes separately
            left_ear, left_width = self.calculate_ear(face_landmarks, self.left_eye_landmarks, frame, self.debug)
            right_ear, right_width = self.calculate_ear(face_landmarks, self.right_eye_landmarks, frame, self.debug)
            
            # Store individual eye values for asymmetry detection
            self.left_ear_values.append(left_ear)
            self.right_ear_values.append(right_ear)
            
            # Calculate eye asymmetry (can indicate fatigue or neurological issues)
            current_asymmetry = abs(left_ear - right_ear) / max((left_ear + right_ear) / 2, 0.001)
            
            # Average EAR from both eyes
            ear = (left_ear + right_ear) / 2.0
            
            # Calculate face width for normalization (distance from camera compensation)
            face_width = (left_width + right_width) / 2.0
            self.face_size_history.append(face_width)
            avg_face_width = sum(self.face_size_history) / len(self.face_size_history)
            
            # Normalize EAR by face width to make it distance-invariant
            normalized_ear = ear / max(avg_face_width, 0.001)
            self.normalized_ear_values.append(normalized_ear)
            
            # Add to moving average for noise reduction
            self.ear_moving_avg.append(ear)
            ear_avg = sum(self.ear_moving_avg) / len(self.ear_moving_avg)
            
            # Store in history for calibration
            self.ear_values.append(ear)
            
            # Dynamic threshold calculation - 2 approaches
            
            # 1. Percentile-based threshold
            if len(self.ear_values) > 15:  # Wait for enough samples
                # Get recent values for dynamic adaptation
                recent_ears = list(self.ear_values)[-50:]  # Look at more recent history
                
                # Apply Savitzky-Golay filter to smooth the values
                if len(recent_ears) > 10:
                    try:
                        window_length = min(len(recent_ears)-1, 7)
                        # Ensure window_length is odd
                        if window_length % 2 == 0:
                            window_length -= 1
                        if window_length >= 5:
                            recent_ears = savgol_filter(recent_ears, window_length, 2).tolist()
                    except:
                        pass
                
                # Calculate statistics
                sorted_ears = sorted(recent_ears)
                q10 = sorted_ears[len(sorted_ears)//10]  # 10% percentile
                q25 = sorted_ears[len(sorted_ears)//4]   # 25% percentile
                q75 = sorted_ears[3*len(sorted_ears)//4] # 75% percentile
                
                # Update min/max using percentiles to avoid outliers
                self.min_ear = q10 * 0.9  # Give some margin
                self.max_ear = q75 * 1.1
                
                # Calculate dynamic threshold based on percentiles
                percentile_threshold = q10 + (q75 - q10) * 0.3  # 30% between q10 and q75
            else:
                # Initial threshold before we have enough data
                percentile_threshold = self.adaptive_threshold
                
            # 2. Baseline-based threshold (using samples when eyes definitely open)
            if len(self.ear_baseline) > 5:
                baseline_mean = sum(self.ear_baseline) / len(self.ear_baseline)
                baseline_std = np.std(list(self.ear_baseline))
                
                # More sophisticated threshold using standard deviation
                # 2 standard deviations below the mean
                baseline_threshold = baseline_mean - (2.0 * baseline_std)
                
                # The system is calibrated once we have enough baseline samples
                if not self.is_calibrated and len(self.ear_baseline) > 15:
                    self.is_calibrated = True
                
            else:
                # Default when not enough baseline data
                baseline_threshold = 0.2
                
            # Combine both threshold approaches for the best results
            # When calibrated, trust the baseline more; otherwise, use percentile-based
            if self.is_calibrated:
                # Weighted average of both thresholds
                threshold = baseline_threshold * 0.7 + (percentile_threshold if 'percentile_threshold' in locals() else self.adaptive_threshold) * 0.3
            else:
                # Rely more on percentiles before calibration
                threshold = (percentile_threshold if 'percentile_threshold' in locals() else self.adaptive_threshold) * 0.7 + (baseline_threshold if 'baseline_threshold' in locals() else 0.2) * 0.3
            
            # Ensure threshold is reasonable
            threshold = max(min(threshold, self.max_ear * 0.8), self.min_ear * 1.2)
            
            # Drowsiness detection - more sensitive than blink detection
            # If eyes are consistently low but not quite closed, might be drowsy
            if ear_avg < threshold * 1.2 and ear_avg > threshold * 0.8:
                self.drowsiness_frames += 1
            else:
                self.drowsiness_frames = max(0, self.drowsiness_frames - 1)
                
            # Alert if drowsy for many consecutive frames
            drowsiness_frame_threshold = 20
            if self.drowsiness_frames > drowsiness_frame_threshold:
                if self.debug:
                    print("Drowsiness detected!")
                self.drowsiness_alerts += 1
                self.drowsiness_frames = 0
                
            # State machine with hysteresis for blink detection
            if ear_avg < threshold:  # Eye might be closed
                self.eye_closed_frames += 1
                self.eye_open_frames = 0
                
                # Add to closed samples for analysis
                self.ear_closed.append(ear)
                
                # Check if eye has been closed for enough frames
                if self.eye_closed_frames >= self.eye_closed_frames_threshold and not self.eye_closed:
                    self.eye_closed = True
                    # Record the time when eye closed for blink duration calculation
                    self.blink_start_time = time.time()
                    
                    if self.debug:
                        print(f"Eye closed: EAR={ear:.4f}, threshold={threshold:.4f}")
                
            else:  # Eye might be open
                # Only add to baseline if definitely open (well above threshold)
                if ear > threshold * 1.2:
                    self.ear_baseline.append(ear)
                
                self.eye_open_frames += 1
                
                # Check if eye has been open for enough frames to confirm the blink is complete
                if self.eye_open_frames >= self.eye_open_frames_threshold and self.eye_closed:
                    self.eye_closed = False
                    
                    # Add blink cooldown to prevent double-counting
                    current_time = time.time()
                    if current_time - self.last_blink_time > BLINK_COOLDOWN:
                        # Only now we count a complete blink (close->open cycle)
                        self.blink_count += 1
                        self.total_blink_count += 1
                        
                        # Calculate blink duration
                        blink_duration = current_time - self.blink_start_time
                        self.blink_duration.append(blink_duration)
                        
                        # Calculate interval since last blink
                        if self.blink_timestamps:
                            interval = current_time - self.blink_timestamps[-1]
                            self.blink_intervals.append(interval)
                            
                        # Record timestamp
                        self.blink_timestamps.append(current_time)
                        self.last_blink_time = current_time
                        
                        if self.debug:
                            print(f"Blink detected #{self.total_blink_count}: " + 
                                 f"EAR={ear:.4f}, duration={blink_duration:.3f}s, threshold={threshold:.4f}")
                
                self.eye_closed_frames = 0
            
            # Display current EAR and threshold if in debug mode
            if self.debug:
                cv2.putText(frame, f"EAR: {ear:.4f} Thresh: {threshold:.4f}", 
                           (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Display normalized EAR 
                if len(self.normalized_ear_values) > 0:
                    norm_ear_avg = sum(self.normalized_ear_values) / len(self.normalized_ear_values)
                    cv2.putText(frame, f"Norm EAR: {normalized_ear:.4f} Avg: {norm_ear_avg:.4f}", 
                               (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Display asymmetry
                cv2.putText(frame, f"Asym: {current_asymmetry:.4f}", 
                           (10, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw EAR graph with threshold line
                self.draw_ear_graph(frame, ear, threshold)
            
            # Recalibrate periodically
            self.calibration_counter += 1
            if self.calibration_counter >= self.recalibration_interval:
                self.recalibrate()
                self.calibration_counter = 0
                
            return ear, threshold
                
        except Exception as e:
            if self.debug:
                print(f"Error in blink detection: {e}")
            return 0, 0
    
    def recalibrate(self):
        """Recalibrate thresholds based on collected data with improved algorithm"""
        if len(self.ear_baseline) > 5 and len(self.ear_closed) > 2:
            # Calculate statistics for open and closed eyes
            open_mean = sum(self.ear_baseline) / len(self.ear_baseline)
            open_std = np.std(list(self.ear_baseline))
            closed_mean = sum(self.ear_closed) / len(self.ear_closed)
            closed_std = np.std(list(self.ear_closed))
            
            # Check if there's clear separation between open and closed distributions
            # Ideally there should be minimal overlap in the distributions
            separation_ratio = abs(open_mean - closed_mean) / max((open_std + closed_std), 0.001)
            
            # If there's clear separation, adjust the threshold
            if separation_ratio > 1.0:  # Reasonable separation
                # Set threshold between open and closed, closer to closed
                # Use a weighted approach based on the separation quality
                weight = min(0.5, max(0.2, 1.0 / separation_ratio))
                self.adaptive_threshold = closed_mean + (open_mean - closed_mean) * weight
                
                if self.debug:
                    print(f"Recalibrated: open={open_mean:.4f}±{open_std:.4f}, " +
                         f"closed={closed_mean:.4f}±{closed_std:.4f}, " +
                         f"separation={separation_ratio:.2f}, " +
                         f"new threshold={self.adaptive_threshold:.4f}")
                         
                # Reset calibration counter to allow for more data collection
                # if the separation is not very good
                if separation_ratio < 2.0:
                    self.ear_closed = deque(maxlen=20)  # Keep the size but clear the data
                    
            # If calibration isn't successful, reset and try again
            else:
                if self.debug:
                    print(f"Calibration skipped: Poor separation between open/closed " +
                         f"eyes. open={open_mean:.4f}±{open_std:.4f}, " +
                         f"closed={closed_mean:.4f}±{closed_std:.4f}, " +
                         f"separation={separation_ratio:.2f}")
                self.ear_closed = deque(maxlen=20)
                
    def analyze_fatigue(self, ear, threshold):
        """Analyze fatigue based on blink patterns, eye openness, and other factors"""
        # Calculate current blink rate (blinks per minute)
        time_since_reset = time.time() - self.last_reset_time
        if time_since_reset > 0:
            current_blink_rate = self.blink_count / time_since_reset * 60.0
        else:
            current_blink_rate = 0
            
        # Check for extended eye closure
        extended_eye_closure = self.eye_closed_frames > self.eye_closed_frames_threshold * 5
        
        # Check for abnormal blink rate (either too high or too low)
        abnormal_blink_rate = False
        
        if len(self.blink_rate_history) > 3:  # Need some history to determine normal
            avg_rate = sum(self.blink_rate_history) / len(self.blink_rate_history)
            std_rate = np.std(list(self.blink_rate_history))
            
            # Too many blinks (more than 1.5 std above average) indicates fatigue
            # Too few blinks (more than 1.5 std below average, but at least 3 bpm) also concerning
            if (current_blink_rate > (avg_rate + 1.5 * std_rate) and current_blink_rate > self.fatigue_threshold):
                abnormal_blink_rate = True
            elif (current_blink_rate < (avg_rate - 1.5 * std_rate) and current_blink_rate < 5):
                abnormal_blink_rate = True
        
        # Check for changes in blink duration
        abnormal_duration = False
        if len(self.blink_duration) > 5:
            avg_duration = sum(self.blink_duration) / len(self.blink_duration)
            recent_duration = sum(list(self.blink_duration)[-3:]) / 3
            
            # Longer blink duration indicates fatigue
            if recent_duration > avg_duration * 1.3:
                abnormal_duration = True
                
        # Check for abnormal eye asymmetry (can indicate extreme fatigue)
        abnormal_asymmetry = False
        if len(self.left_ear_values) > 10 and len(self.right_ear_values) > 10:
            # Get recent values
            recent_left = list(self.left_ear_values)[-10:]
            recent_right = list(self.right_ear_values)[-10:]
            
            # Calculate average asymmetry
            asymmetry_values = [abs(l - r) / ((l + r) / 2) for l, r in zip(recent_left, recent_right)]
            avg_asymmetry = sum(asymmetry_values) / len(asymmetry_values)
            
            # High asymmetry can indicate neurological fatigue
            if avg_asymmetry > 0.3:  # 30% difference between eyes
                abnormal_asymmetry = True
        
        # Determine overall fatigue level
        fatigue_indicators = sum([
            1 if extended_eye_closure else 0,
            1 if abnormal_blink_rate else 0,
            1 if abnormal_duration else 0,
            1 if abnormal_asymmetry else 0,
            1 if self.drowsiness_frames > 10 else 0
        ])
        
        # Update fatigue level
        if fatigue_indicators >= 3:
            self.fatigue_level = "Severe"
            self.session_stats['fatigue_incidents'] += 1
        elif fatigue_indicators >= 2:
            self.fatigue_level = "Moderate"
        elif fatigue_indicators >= 1:
            self.fatigue_level = "Mild"
        else:
            self.fatigue_level = "Normal"
            
        # Add indicator to debug frame
        if self.debug and fatigue_indicators > 0:
            indicators_text = []
            if extended_eye_closure: indicators_text.append("closure")
            if abnormal_blink_rate: indicators_text.append("rate")
            if abnormal_duration: indicators_text.append("duration")
            if abnormal_asymmetry: indicators_text.append("asymmetry")
            if self.drowsiness_frames > 10: indicators_text.append("drowsy")
            
            print(f"Fatigue indicators: {', '.join(indicators_text)}, level: {self.fatigue_level}")
    
    def draw_ear_graph(self, frame, current_ear, threshold):
        """Draw EAR history graph with dynamic threshold line and improved visualization"""
        h, w, _ = frame.shape
        
        # Graph dimensions and position
        graph_width = PLOT_WIDTH
        graph_height = PLOT_HEIGHT
        graph_x = w - graph_width - 10
        graph_y = h - graph_height - 10
        
        # Draw graph background
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), (20, 20, 20), -1)
        cv2.rectangle(frame, (graph_x, graph_y), 
                     (graph_x + graph_width, graph_y + graph_height), (80, 80, 80), 1)
                     
        # Title and labels
        cv2.putText(frame, "EAR History", (graph_x + 5, graph_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Y-axis labels
        cv2.putText(frame, f"{self.max_ear:.2f}", (graph_x - 30, graph_y + 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        cv2.putText(frame, f"{self.min_ear:.2f}", (graph_x - 30, graph_y + graph_height - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
        
        # Calculate y-scale based on min/max EAR values
        y_min = max(0.001, min(self.min_ear, current_ear) * 0.8)
        y_max = max(self.max_ear, current_ear) * 1.2
        y_range = y_max - y_min
        
        # Draw grid lines
        for i in range(1, 4):  # Draw 3 horizontal grid lines
            y_level = graph_y + (graph_height * i) // 4
            cv2.line(frame, (graph_x, y_level), 
                    (graph_x + graph_width, y_level), (50, 50, 50), 1)
            
        # Draw threshold line
        threshold_y = graph_y + graph_height - int((threshold - y_min) / max(0.001, y_range) * graph_height)
        threshold_y = max(graph_y, min(threshold_y, graph_y + graph_height))
        cv2.line(frame, (graph_x, threshold_y), 
                (graph_x + graph_width, threshold_y), (0, 180, 255), 1)
        
        # Draw min-max range lines
        min_y = graph_y + graph_height - int((self.min_ear - y_min) / max(0.001, y_range) * graph_height)
        max_y = graph_y + graph_height - int((self.max_ear - y_min) / max(0.001, y_range) * graph_height)
        min_y = max(graph_y, min(min_y, graph_y + graph_height))
        max_y = max(graph_y, min(max_y, graph_y + graph_height))
        
        # Draw min/max lines as dashed
        for i in range(0, graph_width, 6):  # Draw dashed lines
            cv2.line(frame, (graph_x + i, min_y), (graph_x + i + 3, min_y), (100, 100, 180), 1)
            cv2.line(frame, (graph_x + i, max_y), (graph_x + i + 3, max_y), (100, 180, 100), 1)
        
        # Draw EAR values from history with enhanced visibility
        if len(self.ear_values) > 1:
            values_to_draw = list(self.ear_values)[-graph_width:] if len(self.ear_values) > graph_width else list(self.ear_values)
            
            if not values_to_draw:  # Safety check for empty list
                return
                
            # Get step size - ensure we don't divide by zero
            if graph_width > 0 and len(values_to_draw) > graph_width:
                step = max(1, len(values_to_draw) // graph_width)
                sampled_values = values_to_draw[::step]
            else:
                sampled_values = values_to_draw
            
            # Safety check
            if not sampled_values:
                return
                
            # Calculate starting point
            prev_x = graph_x
            prev_y = graph_y + graph_height - int((sampled_values[0] - y_min) / max(0.001, y_range) * graph_height)
            prev_y = max(graph_y, min(prev_y, graph_y + graph_height))
            
            # Draw line segments with color based on threshold
            for i, ear_val in enumerate(sampled_values[1:]):
                # Calculate x position based on available width
                x = graph_x + int((i + 1) * graph_width / max(1, len(sampled_values)))
                y = graph_y + graph_height - int((ear_val - y_min) / max(0.001, y_range) * graph_height)
                
                # Keep y within graph bounds
                y = max(graph_y, min(y, graph_y + graph_height))
                
                # Color based on whether above/below threshold
                if ear_val < threshold:
                    line_color = (50, 50, 240)  # Red for below threshold (closed)
                else:
                    line_color = (50, 240, 50)  # Green for above threshold (open)
                
                cv2.line(frame, (prev_x, prev_y), (x, y), line_color, 1)
                
                # Add a circle at each data point for visibility
                if i % 5 == 0:  # Only draw circles at every 5th point to avoid clutter
                    cv2.circle(frame, (x, y), 1, (200, 200, 200), -1)
                    
                prev_x, prev_y = x, y
            
        # Mark the most recent value with a more visible point
        if len(self.ear_values) > 0:
            latest_y = graph_y + graph_height - int((current_ear - y_min) / max(0.001, y_range) * graph_height)
            latest_y = max(graph_y, min(latest_y, graph_y + graph_height))
            latest_x = graph_x + graph_width - 2
            
            # Highlight current value
            cv2.circle(frame, (latest_x, latest_y), 3, (50, 50, 240) if current_ear < threshold else (50, 240, 50), -1)
            cv2.circle(frame, (latest_x, latest_y), 3, (255, 255, 255), 1)
    
    def analyze_attention(self, face_landmarks, frame):
        """Enhanced attention analysis using face orientation and eye tracking"""
        try:
            h, w, _ = frame.shape
            
            # Extract key landmarks for attention analysis
            nose_tip = face_landmarks.landmark[4]  # Nose tip
            left_eye = face_landmarks.landmark[33]  # Left eye outer corner
            right_eye = face_landmarks.landmark[263]  # Right eye outer corner
            forehead = face_landmarks.landmark[10]  # Forehead/top of face
            chin = face_landmarks.landmark[152]  # Bottom of chin
            
            # Convert to pixel coordinates
            nose_x, nose_y = int(nose_tip.x * w), int(nose_tip.y * h)
            
            # Calculate face orientation vectors
            horizontal_vector = (right_eye.x - left_eye.x, right_eye.y - left_eye.y)
            vertical_vector = (chin.y - forehead.y, forehead.x - chin.x)  # Perpendicular to face plane
            
            # Normalize vectors
            h_magnitude = math.sqrt(horizontal_vector[0]**2 + horizontal_vector[1]**2)
            horizontal_vector = (horizontal_vector[0]/max(h_magnitude, 0.001), 
                                horizontal_vector[1]/max(h_magnitude, 0.001))
            
            v_magnitude = math.sqrt(vertical_vector[0]**2 + vertical_vector[1]**2)
            vertical_vector = (vertical_vector[0]/max(v_magnitude, 0.001), 
                              vertical_vector[1]/max(v_magnitude, 0.001))
            
            # Calculate face direction angles
            # Horizontal angle: 0 means facing straight, negative is left, positive is right
            horizontal_angle = math.degrees(math.atan2(horizontal_vector[1], horizontal_vector[0]))
            horizontal_angle = (horizontal_angle + 90) % 360  # Adjust to make 0 = straight ahead
            if horizontal_angle > 180:
                horizontal_angle -= 360  # Convert to -180 to 180 range
                
            # Vertical angle: 0 means facing straight, negative is down, positive is up
            vertical_angle = math.degrees(math.atan2(vertical_vector[1], vertical_vector[0]))
            
            # Store face orientation for visualization
            self.face_horizontal_angle = horizontal_angle
            self.face_vertical_angle = vertical_angle
            
            # Add a position-based factor using nose position relative to frame center
            # This helps account for people who might be at an angle to the camera
            frame_center_x = w // 2
            distance_from_center = abs(nose_x - frame_center_x) / (w * 0.5)  # Normalized 0-1
            
            print(f"Face angles - H: {horizontal_angle:.1f}°, V: {vertical_angle:.1f}°, Dist: {distance_from_center:.2f}")
            
            # Classify horizontal attention (left-right) - more lenient
            if abs(horizontal_angle) < 25:  # More lenient
                horizontal_attention = "Centered"
            elif abs(horizontal_angle) < 45:
                horizontal_attention = "Slightly Off" + (" Right" if horizontal_angle > 0 else " Left")
            else:
                horizontal_attention = "Looking" + (" Right" if horizontal_angle > 0 else " Left")
                
            # Classify vertical attention (up-down) - more lenient
            if abs(vertical_angle) < 25:  # More lenient
                vertical_attention = "Level"
            elif vertical_angle > 25:
                vertical_attention = "Looking Up"
            else:
                vertical_attention = "Looking Down"
                
            # Store current state for potential rollback
            previous_state = self.attention_state
            
            # Determine new state with more lenient thresholds for "Focused"
            new_state = "Unknown"  # Default initialization
            
            # Use OR conditions for Focused to make it more achievable
            # Either good face angles OR good position can qualify as Focused
            if ((horizontal_attention == "Centered" and vertical_attention == "Level") or
                (distance_from_center < 0.20)):  # Much more lenient - was 0.12
                new_state = "Focused"
            # Partially attentive has moderate conditions
            elif (("Slightly Off" in horizontal_attention and abs(vertical_angle) < 30) or  # More lenient vertical
                  (horizontal_attention == "Centered" and vertical_attention != "Level") or
                  (distance_from_center < 0.35)):  # More lenient - was 0.25
                new_state = "Partially Attentive"
            # Everything else is distracted
            else:
                new_state = "Distracted"
            
            # Apply a very light temporal smoothing
            # Only require 1 frame to confirm state changes
            if hasattr(self, 'prev_calculated_state'):
                # Keep state transitions fluid - no complex conditions
                self.attention_state = new_state
            else:
                # Initialize if we don't have history
                self.attention_state = new_state
            
            # Store calculated state for next frame
            self.prev_calculated_state = new_state
                    
            # Add to attention history for trend analysis
            self.attention_history.append(self.attention_state)
            
            # Update session statistics
            if self.attention_state in self.session_stats['attention_states']:
                self.session_stats['attention_states'][self.attention_state] += 1
            else:
                self.session_stats['attention_states'][self.attention_state] = 1
                
            # Draw attention vectors and labels if debug is enabled
            if self.debug:
                # Draw face direction vector
                face_center_x, face_center_y = int((left_eye.x + right_eye.x) * w / 2), int((left_eye.y + right_eye.y) * h / 2)
                vector_length = 50
                vector_end_x = int(face_center_x + vector_length * horizontal_vector[0])
                vector_end_y = int(face_center_y + vector_length * horizontal_vector[1])
                
                # Draw horizontal attention vector
                cv2.arrowedLine(frame, (face_center_x, face_center_y), 
                               (vector_end_x, vector_end_y), (0, 255, 255), 2)
                
                # Draw labels for horizontal and vertical attention with more info
                cv2.putText(frame, f"Horz: {horizontal_attention} ({horizontal_angle:.1f}°)", 
                           (10, h - 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Vert: {vertical_attention} ({vertical_angle:.1f}°)", 
                           (10, h - 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"Center dist: {distance_from_center:.2f}", 
                           (10, h - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(frame, f"New state: {new_state}", 
                           (10, h - 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw center reference lines
                cv2.line(frame, (w//2, 0), (w//2, h), (50, 50, 50), 1)  # Vertical center line
                cv2.line(frame, (0, h//2), (w, h//2), (50, 50, 50), 1)  # Horizontal center line
                
                # Show focused region
                focused_radius = int(0.20 * w * 0.5)  # 0.20 is our threshold
                cv2.circle(frame, (w//2, h//2), focused_radius, (0, 255, 0), 1)
                
        except Exception as e:
            print(f"Error in attention analysis: {e}")
            self.attention_state = "Unknown"
    
    def analyze_posture(self, pose_landmarks, frame):
        """Enhanced posture analysis with detailed biomechanical assessment"""
        try:
            h, w, _ = frame.shape
            
            # Extract key landmarks for posture analysis
            # Upper body landmarks
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_HIP]
            left_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            print(f"Posture landmarks detected - Shoulders: ({left_shoulder.visibility:.2f}, {right_shoulder.visibility:.2f}), "
                  f"Hips: ({left_hip.visibility:.2f}, {right_hip.visibility:.2f})")
                  
            # Check visibility of key landmarks
            min_visibility = 0.5
            if (left_shoulder.visibility < min_visibility or 
                right_shoulder.visibility < min_visibility or
                left_hip.visibility < min_visibility or
                right_hip.visibility < min_visibility):
                print("Low visibility for key posture landmarks")
                self.posture_state = "Partially Visible"
                return
                
            # Calculate key metrics
            
            # 1. Shoulder alignment (horizontal level)
            shoulder_dx = right_shoulder.x - left_shoulder.x
            shoulder_dy = right_shoulder.y - left_shoulder.y
            shoulder_angle = math.degrees(math.atan2(shoulder_dy, shoulder_dx))
            
            # Adjust angle to be relative to horizontal (0 degrees)
            shoulder_angle = abs(shoulder_angle)
            
            # 2. Upper body inclination (torso angle)
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            hip_center_x = (left_hip.x + right_hip.x) / 2
            hip_center_y = (left_hip.y + right_hip.y) / 2
            
            # Detect if hip points are visible enough for valid calculation
            if (left_hip.visibility < 0.3 or right_hip.visibility < 0.3):
                # Use an approximation if hips aren't clearly visible
                # Approximate torso angle using shoulders and head position
                torso_angle = math.degrees(math.atan2(
                    shoulder_center_y - nose.y,
                    shoulder_center_x - nose.x
                ))
                print("Using approximated torso angle with head position")
            else:
                torso_angle = math.degrees(math.atan2(
                    shoulder_center_y - hip_center_y,
                    shoulder_center_x - hip_center_x
                ))
            
            # Adjust torso angle to be 0 when vertical (convert from -180 to 180 range)
            torso_angle = (torso_angle + 90) % 360
            if torso_angle > 180:
                torso_angle -= 360
                
            # Convert to absolute tilt from vertical
            torso_angle = abs(torso_angle)
            
            # 3. Head alignment (vertical alignment of head relative to shoulders)
            ear_center_x = (left_ear.x + right_ear.x) / 2
            ear_center_y = (left_ear.y + right_ear.y) / 2
            
            # Head forward tilt - how far forward the head is compared to shoulders
            # Higher values indicate head forward posture (potential "text neck")
            head_forward_ratio = (ear_center_x - shoulder_center_x) / max(0.001, nose.y - shoulder_center_y)
            
            # Calculate head tilt angle (from vertical)
            if left_ear.visibility > 0.5 and right_ear.visibility > 0.5:
                head_tilt_angle = math.degrees(math.atan2(
                    right_ear.y - left_ear.y,
                    right_ear.x - left_ear.x
                ))
            else:
                # Use nose and shoulder center if ears aren't visible
                head_tilt_angle = math.degrees(math.atan2(
                    nose.y - shoulder_center_y,
                    nose.x - shoulder_center_x
                ))
            
            # Normalize and take abs value
            head_tilt_angle = abs(head_tilt_angle)
            
            # Print computed values for debugging
            print(f"Posture metrics - Shoulder angle: {shoulder_angle:.1f}°, Torso angle: {torso_angle:.1f}°, "
                  f"Head tilt: {head_tilt_angle:.1f}°, Head forward ratio: {head_forward_ratio:.2f}")
            
            # Score each component (lower is better)
            shoulder_score = min(10, shoulder_angle / 3)  # 0-10 scale
            torso_score = min(10, torso_angle / 5)        # 0-10 scale
            head_score = min(10, head_tilt_angle / 5)     # 0-10 scale
            
            # Combine into overall posture assessment
            total_score = shoulder_score * 0.3 + torso_score * 0.4 + head_score * 0.3
            
            # Classify posture
            if total_score < 3:
                self.posture_state = "Excellent"
            elif total_score < 5:
                self.posture_state = "Good"
            elif total_score < 7:
                self.posture_state = "Fair"
            else:
                self.posture_state = "Poor"
                # Increment posture issues counter for session stats
                self.session_stats['posture_issues'] += 1
            
            # Update posture history
            self.posture_history.append(self.posture_state)
            
            # Update session statistics
            if self.posture_state in self.session_stats['posture_states']:
                self.session_stats['posture_states'][self.posture_state] += 1
            else:
                self.session_stats['posture_states'][self.posture_state] = 1
            
            # Draw posture indicators if debug is enabled
            if self.debug:
                # Draw shoulder line
                left_shoulder_px = int(left_shoulder.x * w), int(left_shoulder.y * h)
                right_shoulder_px = int(right_shoulder.x * w), int(right_shoulder.y * h)
                cv2.line(frame, left_shoulder_px, right_shoulder_px, (255, 0, 0), 2)
                
                # Draw torso line
                shoulder_center = int(shoulder_center_x * w), int(shoulder_center_y * h)
                hip_center = int(hip_center_x * w), int(hip_center_y * h)
                cv2.line(frame, shoulder_center, hip_center, (0, 255, 0), 2)
                
                # Draw head alignment indicator
                nose_px = int(nose.x * w), int(nose.y * h)
                ear_center = int(ear_center_x * w), int(ear_center_y * h)
                cv2.line(frame, ear_center, nose_px, (0, 0, 255), 2)
                
                # Display scores
                cv2.putText(frame, f"Shoulder: {shoulder_score:.1f}", 
                           (w - 220, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Torso: {torso_score:.1f}", 
                           (w - 220, h - 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(frame, f"Head: {head_score:.1f}", 
                           (w - 220, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
        except Exception as e:
            print(f"Error in posture analysis: {e}")
            self.posture_state = "Unknown"
    
    def analyze_movement(self, pose_landmarks, frame):
        """Enhanced movement analysis with segmentation by body part and behavior patterns"""
        try:
            # Get key points for movement analysis
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Create position vectors for different body parts
            head_pos = np.array([nose.x, nose.y, nose.z])
            left_arm_pos = np.array([left_wrist.x, left_wrist.y, left_wrist.z])
            right_arm_pos = np.array([right_wrist.x, right_wrist.y, right_wrist.z])
            shoulders_pos = np.array([
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2,
                (left_shoulder.z + right_shoulder.z) / 2
            ])
            
            # Combine into a full position array with weights for importance
            current_pos = np.concatenate([
                head_pos * 0.4,        # Head movement is important
                left_arm_pos * 0.3,    # Arm movements
                right_arm_pos * 0.3,   # Arm movements
                shoulders_pos * 0.4    # Torso stability
            ])
            
            # Calculate movement for each body part separately
            if len(self.movement_history) > 0:
                prev_pos = self.movement_history[-1]
                
                # Calculate position differences
                pos_diff = current_pos - prev_pos
                
                # Calculate movement magnitude
                movement_magnitude = np.linalg.norm(pos_diff)
                self.movement_magnitude.append(movement_magnitude)
                
                # Store movement components (for advanced analytics)
                head_movement = np.linalg.norm(pos_diff[:3])  # Head
                left_arm_movement = np.linalg.norm(pos_diff[3:6])  # Left arm
                right_arm_movement = np.linalg.norm(pos_diff[6:9])  # Right arm
                torso_movement = np.linalg.norm(pos_diff[9:12])  # Torso
                
                # Calculate recent average movement
                recent_movements = list(self.movement_magnitude)[-10:]
                avg_movement = sum(recent_movements) / len(recent_movements)
                
                # Detect patterns in movement
                if len(self.movement_magnitude) > 20:
                    # Detect fidgeting (alternating small movements)
                    recent_patterns = list(self.movement_magnitude)[-20:]
                    
                    # Calculate variance and zero-crossings
                    movement_mean = sum(recent_patterns) / len(recent_patterns)
                    zero_centered = [m - movement_mean for m in recent_patterns]
                    sign_changes = sum(1 for i in range(1, len(zero_centered)) 
                                     if zero_centered[i] * zero_centered[i-1] < 0)
                    
                    # Detect fidgeting pattern (high frequency, low amplitude movements)
                    fidgeting = False
                    if sign_changes > 12 and avg_movement < 0.05:  # Many sign changes with small movement
                        fidgeting = True
                    
                    # Store fidgeting detection
                    self.fidgeting_detected = fidgeting
                
                # Classify movement level
                if avg_movement < 0.01:
                    movement_level = "Still"
                elif avg_movement < 0.03:
                    movement_level = "Low"
                elif avg_movement < 0.07:
                    movement_level = "Moderate"
                else:
                    movement_level = "High"
                    
                # Add qualifier if fidgeting is detected
                if hasattr(self, 'fidgeting_detected') and self.fidgeting_detected:
                    movement_level = f"{movement_level} (Fidgeting)"
                
                # Store detailed movement data for visualization
                self.movement_details = {
                    'head': head_movement,
                    'left_arm': left_arm_movement,
                    'right_arm': right_arm_movement,
                    'torso': torso_movement,
                    'total': movement_magnitude,
                    'average': avg_movement
                }
                
                # Update the class attribute
                self.movement_level = movement_level
            
            # Add current position to history
            self.movement_history.append(current_pos)
            
            # Debug visualization
            if self.debug and hasattr(self, 'movement_details'):
                h, w, _ = frame.shape
                
                # Show movement values
                cv2.putText(frame, f"Movement: {self.movement_details['average']:.4f}", 
                           (10, h - 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                cv2.putText(frame, f"Head: {self.movement_details['head']:.4f}", 
                           (10, h - 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                cv2.putText(frame, f"Arms: {(self.movement_details['left_arm'] + self.movement_details['right_arm'])/2:.4f}", 
                           (10, h - 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                if hasattr(self, 'fidgeting_detected') and self.fidgeting_detected:
                    cv2.putText(frame, "Fidgeting Detected", 
                               (10, h - 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
                
        except Exception as e:
            if self.debug:
                print(f"Error in movement analysis: {e}")
            self.movement_level = "Unknown"
    
    def display_results(self, frame, blink_rate, ear=0, threshold=0):
        """Display comprehensive analysis results with improved visual design"""
        h, w, _ = frame.shape
        
        # Determine color indicators based on states
        attention_color = {
            'Focused': (0, 255, 0),       # Green
            'Partially Attentive': (0, 165, 255),  # Orange
            'Distracted': (0, 0, 255),    # Red
            'Unknown': (200, 200, 200)    # Gray
        }.get(self.attention_state, (200, 200, 200))
        
        posture_color = {
            'Excellent': (0, 255, 0),     # Green
            'Good': (0, 255, 0),          # Green
            'Fair': (0, 165, 255),        # Orange
            'Poor': (0, 0, 255),          # Red
            'Unknown': (200, 200, 200)    # Gray
        }.get(self.posture_state, (200, 200, 200))
        
        fatigue_color = {
            'Normal': (0, 255, 0),        # Green
            'Mild': (0, 255, 255),        # Yellow
            'Moderate': (0, 165, 255),    # Orange
            'Severe': (0, 0, 255),        # Red
            'Unknown': (200, 200, 200)    # Gray
        }.get(self.fatigue_level, (200, 200, 200))
        
        # Background semi-transparent overlay for all text
        overlay = frame.copy()
        
        # Main panel background - INCREASED SIZE
        cv2.rectangle(overlay, (10, 10), (340, 280), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        cv2.rectangle(frame, (10, 10), (340, 280), (100, 100, 100), 1)
        
        # Title bar
        cv2.rectangle(frame, (10, 10), (340, 40), (40, 40, 40), -1)
        cv2.putText(frame, "Integrated Analysis", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Section: Emotion Analysis with better visuals
        y_pos = 70
        
        # Show a message when emotion detection is still initializing
        if not self.emotion_detected and self.enable_emotion:
            emotion_text = "Initializing emotion..."
            emotion_color = (100, 100, 255)  # Light blue for initializing
        else:
            emotion_text = f"Emotion: {self.current_emotion}"
            emotion_color = (255, 255, 255)
            
        cv2.putText(frame, emotion_text, 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, emotion_color, 2)
        
        # If we have emotion scores, show the top 3 with better visualization
        if any(self.emotion_scores.values()):
            # Get top 3 emotions
            top_emotions = sorted(self.emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Define max bar width
            max_bar_width = 100
            bar_height = 10
            bar_start_x = 140
            bar_spacing = 20  # Increased spacing between bars
            
            for i, (emotion, score) in enumerate(top_emotions):
                if score > 0.01:  # Only show if score is significant
                    # Ensure score is between 0-1
                    norm_score = min(1.0, max(0.0, score))
                    
                    # Calculate bar length based on score (proportional to max width)
                    bar_length = int(norm_score * max_bar_width)
                    
                    # Different colors for different emotions
                    emotion_bar_colors = {
                        'angry': (0, 0, 255),      # Red
                        'disgust': (0, 140, 255),  # Orange-ish
                        'fear': (0, 69, 255),      # Red-Orange
                        'happy': (0, 255, 0),      # Green
                        'sad': (255, 0, 0),        # Blue
                        'surprise': (255, 255, 0), # Cyan
                        'neutral': (200, 200, 200) # Gray
                    }
                    bar_color = emotion_bar_colors.get(emotion, (100, 100, 255))
                    
                    # Calculate positions - Using consistent spacing
                    y_bar = y_pos + 15 + i * bar_spacing
                    
                    # Draw background for bar (full width)
                    cv2.rectangle(frame, 
                                 (bar_start_x, y_bar), 
                                 (bar_start_x + max_bar_width, y_bar + bar_height), 
                                 (40, 40, 40), -1)
                    
                    # Draw colored bar (proportional to score)
                    if bar_length > 0:
                        cv2.rectangle(frame, 
                                     (bar_start_x, y_bar), 
                                     (bar_start_x + bar_length, y_bar + bar_height), 
                                     bar_color, -1)
                    
                    # Draw label with percentages
                    cv2.putText(frame, f"{emotion}: {norm_score*100:.0f}%", 
                               (30, y_bar + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Section: Attention and Posture - MOVED DOWN to avoid overlap
        y_pos = 150  # Increased vertical position to make room for emotion bars
        cv2.putText(frame, f"Attention: {self.attention_state}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, attention_color, 2)
        
        y_pos += 30
        cv2.putText(frame, f"Posture: {self.posture_state}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, posture_color, 2)
        
        y_pos += 30
        cv2.putText(frame, f"Movement: {self.movement_level}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Section: Blink Analysis
        y_pos += 30
        cv2.putText(frame, f"Blink Rate: {blink_rate:.1f} bpm", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add small text for total blinks
        cv2.putText(frame, f"Total: {self.total_blink_count}", 
                   (210, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Fatigue level with color coding
        y_pos += 30
        cv2.putText(frame, f"Fatigue: {self.fatigue_level}", 
                   (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fatigue_color, 2)
        
        # Status bar
        # Right side status panel for system information
        status_x = w - 220
        status_y = 10
        cv2.rectangle(frame, (status_x, status_y), (w-10, status_y+90), (0, 0, 0), -1)
        cv2.rectangle(frame, (status_x, status_y), (w-10, status_y+90), (100, 100, 100), 1)
        
        # System status info
        cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                   (status_x + 10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Calibration status
        status = "Calibrated" if self.is_calibrated else f"Calibrating {min(100, len(self.ear_baseline)*2)}%"
        status_color = (0, 255, 0) if self.is_calibrated else (0, 165, 255)
        cv2.putText(frame, f"Status: {status}", 
                  (status_x + 10, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        
        # Session duration
        session_time = time.time() - self.session_start_time
        minutes, seconds = divmod(int(session_time), 60)
        hours, minutes = divmod(minutes, 60)
        
        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        else:
            time_str = f"{minutes}m {seconds}s"
            
        cv2.putText(frame, f"Session: {time_str}", 
                  (status_x + 10, status_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Eye state indicator - use a clear visual indicator
        eye_state_x = w - 100
        eye_state_y = 130  # Adjusted to be below status panel
        
        # Create eye state indicator
        cv2.rectangle(frame, (eye_state_x - 60, eye_state_y - 30), 
                     (eye_state_x + 60, eye_state_y + 30), (40, 40, 40), -1)
        
        if self.eye_closed:
            # Draw closed eye
            cv2.line(frame, (eye_state_x - 30, eye_state_y), 
                    (eye_state_x + 30, eye_state_y), (0, 0, 255), 3)
            state_text = "CLOSED"
            state_color = (0, 0, 255)
        else:
            # Draw open eye
            cv2.ellipse(frame, (eye_state_x, eye_state_y), (30, 15), 
                       0, 0, 360, (0, 255, 0), 2)
            cv2.circle(frame, (eye_state_x, eye_state_y), 10, (0, 255, 0), -1)
            state_text = "OPEN"
            state_color = (0, 255, 0)
            
        cv2.putText(frame, state_text, 
                   (eye_state_x - 40, eye_state_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        
        # Draw EAR graph if debug is enabled
        if self.debug:
            self.draw_ear_graph(frame, ear, threshold)
            
        # Add interface instructions at the bottom
        instruction_y = h - 20
        cv2.putText(frame, "Press 'q' to quit, 'd' to toggle debug, 'r' to record", 
                   (20, instruction_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                   
        # Add recording indicator if active
        if self.recording and hasattr(self, 'recording_start_time'):
            rec_time = time.time() - self.recording_start_time
            minutes, seconds = divmod(int(rec_time), 60)
            
            # Flashing red circle for recording
            if int(time.time() * 2) % 2 == 0:  # Flash at 2Hz
                cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)
                
            cv2.putText(frame, f"REC {minutes:02d}:{seconds:02d}", 
                       (45, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def cleanup(self):
        """Clean up resources before closing"""
        # Stop the emotion thread
        self.emotion_thread_active = False
        if hasattr(self, 'emotion_thread') and self.emotion_thread.is_alive():
            self.emotion_thread.join(timeout=1.0)
        
        # Clean up other resources
        self.face_mesh.close()
        self.pose.close()
        
        # Clean up audio resources
        self.audio_thread_active = False
        if hasattr(self, 'audio_stream'):
            self.audio_stream.stop()
            self.audio_stream.close()
        
        # Close video writer if recording
        if self.recording and self.video_writer is not None:
            self.video_writer.release()
            
        # Generate final analytics report and log files
        if self.log_data:
            try:
                # Calculate total session duration
                session_duration_seconds = time.time() - self.session_start_time
                hours, remainder = divmod(int(session_duration_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                # Convert session stats to percentages
                attention_stats = {}
                total_attention_frames = sum(self.session_stats['attention_states'].values())
                if total_attention_frames > 0:
                    for state, count in self.session_stats['attention_states'].items():
                        attention_stats[state] = (count / total_attention_frames) * 100
                
                posture_stats = {}
                total_posture_frames = sum(self.session_stats['posture_states'].values())
                if total_posture_frames > 0:
                    for state, count in self.session_stats['posture_states'].items():
                        posture_stats[state] = (count / total_posture_frames) * 100
                
                emotion_stats = {}
                total_emotion_detections = sum(self.session_stats['emotion_counts'].values())
                if total_emotion_detections > 0:
                    for emotion, count in self.session_stats['emotion_counts'].items():
                        emotion_stats[emotion] = (count / total_emotion_detections) * 100
                
                # Update the session summary with final stats
                with open(self.summary_file, 'a') as file:
                    file.write("\n\n--- FINAL SESSION SUMMARY ---\n")
                    file.write(f"Session completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    file.write(f"Total duration: {hours:02d}:{minutes:02d}:{seconds:02d}\n")
                    file.write(f"Frames processed: {self.frame_counter}\n")
                    file.write(f"Average FPS: {sum(self.fps_history) / max(1, len(self.fps_history)):.1f}\n\n")
                    
                    file.write("=== BLINK STATISTICS ===\n")
                    file.write(f"Total blinks: {self.total_blink_count}\n")
                    file.write(f"Blink rate: {self.total_blink_count / max(1, session_duration_seconds / 60):.1f} blinks per minute\n")
                    if len(self.blink_rate_history) > 0:
                        file.write(f"Average blink rate: {sum(self.blink_rate_history) / max(1, len(self.blink_rate_history)):.1f} blinks per minute\n")
                        file.write(f"Peak blink rate: {self.max_blink_rate:.1f} blinks per minute\n")
                    
                    if len(self.blink_duration) > 0:
                        file.write(f"Average blink duration: {sum(self.blink_duration) / max(1, len(self.blink_duration)):.3f}s\n")
                        
                    file.write("\n=== ATTENTION STATISTICS ===\n")
                    for state, percentage in sorted(attention_stats.items(), key=lambda x: x[1], reverse=True):
                        file.write(f"{state}: {percentage:.1f}% ({self.session_stats['attention_states'].get(state, 0)} frames)\n")
                    
                    file.write("\n=== POSTURE STATISTICS ===\n")
                    for state, percentage in sorted(posture_stats.items(), key=lambda x: x[1], reverse=True):
                        file.write(f"{state}: {percentage:.1f}% ({self.session_stats['posture_states'].get(state, 0)} frames)\n")
                            
                    file.write(f"Posture issues: {self.session_stats['posture_issues']}\n")
                    
                    file.write("\n=== FATIGUE STATISTICS ===\n")
                    file.write(f"Fatigue incidents: {self.session_stats['fatigue_incidents']}\n")
                    file.write(f"Drowsiness alerts: {self.drowsiness_alerts}\n")
                    
                    if self.enable_emotion:
                        file.write("\n=== EMOTION STATISTICS ===\n")
                        for emotion, percentage in sorted(emotion_stats.items(), key=lambda x: x[1], reverse=True):
                            if percentage > 1:  # Only show emotions with more than 1% presence
                                file.write(f"{emotion}: {percentage:.1f}% ({self.session_stats['emotion_counts'].get(emotion, 0)} detections)\n")
                
                # Create final session analytics JSON
                final_stats = {
                    "session": {
                        "start_time": datetime.datetime.fromtimestamp(self.session_start_time).strftime('%Y-%m-%d %H:%M:%S'),
                        "end_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        "duration_seconds": session_duration_seconds,
                        "frames_processed": self.frame_counter,
                        "average_fps": sum(self.fps_history) / max(1, len(self.fps_history))
                    },
                    "blink_analysis": {
                        "total_blinks": self.total_blink_count,
                        "blink_rate_per_minute": self.total_blink_count / max(1, session_duration_seconds / 60),
                        "average_blink_duration": sum(self.blink_duration) / max(1, len(self.blink_duration)) if self.blink_duration else 0,
                        "max_blink_rate": self.max_blink_rate,
                        "min_blink_rate": self.min_blink_rate
                    },
                    "attention_analysis": {
                        "states": attention_stats,
                    },
                    "posture_analysis": {
                        "states": posture_stats,
                        "posture_issues": self.session_stats['posture_issues']
                    },
                    "fatigue_analysis": {
                        "incidents": self.session_stats['fatigue_incidents'],
                        "drowsiness_alerts": self.drowsiness_alerts
                    },
                    "emotion_analysis": {
                        "emotions": emotion_stats
                    },
                    "performance": {
                        "average_processing_time": sum(self.processing_times) / max(1, len(self.processing_times)),
                        "max_processing_time": max(self.processing_times) if self.processing_times else 0,
                        "min_processing_time": min(self.processing_times) if self.processing_times else 0
                    }
                }
                
                # Save final stats
                with open(self.session_info_file.replace('.json', '_final.json'), 'w') as file:
                    json.dump(final_stats, file, indent=4)
                
                # Generate comprehensive PDF report if requested
                report_file = None
                if hasattr(self, 'generate_report') and self.generate_report:
                    report_file = self._generate_comprehensive_report()
                
                # Print completion message
                print(f"\nSession completed. Log files saved to:")
                print(f"  Summary: {self.summary_file}")
                print(f"  Raw data: {self.raw_data_file}")
                print(f"  Session info: {self.session_info_file}")
                print(f"  Analytics: {self.log_file}")
                
                if report_file:
                    print(f"  Comprehensive report: {report_file}")
                    
            except Exception as e:
                print(f"Error generating final report: {e}")
                import traceback
                traceback.print_exc()
    
    def toggle_recording(self, frame):
        """Start or stop recording the analyzed video"""
        if not self.recording:
            # Start recording
            try:
                # Create videos directory if it doesn't exist
                if not os.path.exists('videos'):
                    os.makedirs('videos')
                
                # Create video file with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                video_file = f"videos/session_recording_{timestamp}.mp4"
                
                # Get frame dimensions
                h, w, _ = frame.shape
                
                # Initialize video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(video_file, fourcc, 20.0, (w, h))
                
                self.recording = True
                self.recording_start_time = time.time()
                print(f"Recording started: {video_file}")
                
            except Exception as e:
                print(f"Error starting recording: {e}")
                self.recording = False
                
        else:
            # Stop recording
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
                
            recording_duration = time.time() - self.recording_start_time
            print(f"Recording stopped after {recording_duration:.1f} seconds")
            self.recording = False

    def _generate_comprehensive_report(self):
        """Generate a comprehensive PDF report with all analytics data and visualizations"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            from matplotlib.gridspec import GridSpec
            import pandas as pd
            import numpy as np
            
            # Create reports directory if it doesn't exist
            if not os.path.exists('logs/reports'):
                os.makedirs('logs/reports')
            
            # Generate timestamp for file naming
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = f"logs/reports/comprehensive_report_{timestamp}.pdf"
            
            # Create a PDF file
            with PdfPages(report_file) as pdf:
                # Load the CSV data
                try:
                    df = pd.read_csv(self.log_file)
                    raw_df = pd.read_csv(self.raw_data_file)
                except Exception as e:
                    print(f"Error loading CSV data for report: {e}")
                    df = None
                    raw_df = None
                
                # Calculate session duration
                session_duration_seconds = time.time() - self.session_start_time
                hours, remainder = divmod(int(session_duration_seconds), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                # ----- Title Page -----
                plt.figure(figsize=(8.5, 11))
                plt.axis('off')
                
                # Title
                plt.text(0.5, 0.8, 'Integrated Feature Analysis', 
                         ha='center', fontsize=24, fontweight='bold')
                plt.text(0.5, 0.7, 'Comprehensive Session Report', 
                         ha='center', fontsize=18)
                
                # Session info
                plt.text(0.5, 0.5, f"Session Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", 
                         ha='center', fontsize=14)
                plt.text(0.5, 0.45, f"Session Duration: {hours:02d}:{minutes:02d}:{seconds:02d}", 
                         ha='center', fontsize=14)
                plt.text(0.5, 0.4, f"Frames Processed: {self.frame_counter}", 
                         ha='center', fontsize=14)
                
                # Footer
                plt.text(0.5, 0.1, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                         ha='center', fontsize=10)
                plt.text(0.5, 0.05, "Integrated Feature Analyzer v1.0", 
                         ha='center', fontsize=10)
                
                pdf.savefig()
                plt.close()
                
                # ----- Summary Page -----
                plt.figure(figsize=(8.5, 11))
                plt.axis('off')
                plt.text(0.5, 0.95, 'Session Summary', ha='center', fontsize=18, fontweight='bold')
                
                # Key statistics
                summary_text = [
                    f"Total Session Duration: {hours:02d}:{minutes:02d}:{seconds:02d}",
                    f"Total Frames Analyzed: {self.frame_counter}",
                    f"Average FPS: {sum(self.fps_history) / max(1, len(self.fps_history)):.1f}",
                    f"Total Blinks Detected: {self.total_blink_count}",
                    f"Average Blink Rate: {self.total_blink_count / max(1, session_duration_seconds / 60):.1f} blinks per minute",
                    f"Drowsiness Alerts: {self.drowsiness_alerts}",
                    f"Fatigue Incidents: {self.session_stats['fatigue_incidents']}",
                    f"Posture Issues: {self.session_stats['posture_issues']}"
                ]
                
                for i, text in enumerate(summary_text):
                    plt.text(0.1, 0.85 - i*0.05, text, fontsize=12)
                
                # Calculate distribution percentages
                attention_stats = {}
                total_attention_frames = sum(self.session_stats['attention_states'].values())
                if total_attention_frames > 0:
                    for state, count in self.session_stats['attention_states'].items():
                        attention_stats[state] = (count / total_attention_frames) * 100
                
                posture_stats = {}
                total_posture_frames = sum(self.session_stats['posture_states'].values())
                if total_posture_frames > 0:
                    for state, count in self.session_stats['posture_states'].items():
                        posture_stats[state] = (count / total_posture_frames) * 100
                
                emotion_stats = {}
                total_emotion_detections = sum(self.session_stats['emotion_counts'].values())
                if total_emotion_detections > 0:
                    for emotion, count in self.session_stats['emotion_counts'].items():
                        emotion_stats[emotion] = (count / total_emotion_detections) * 100
                
                # Add small summary charts
                gs = GridSpec(2, 2, left=0.1, right=0.9, bottom=0.25, top=0.55, wspace=0.4, hspace=0.3)
                
                # Attention distribution
                if attention_stats:
                    ax1 = plt.subplot(gs[0, 0])
                    ax1.pie([attention_stats.get(x, 0) for x in ['Focused', 'Partially Attentive', 'Distracted', 'Unknown']], 
                           labels=['Focused', 'Partial', 'Distracted', 'Unknown'],
                           autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Attention Distribution')
                
                # Posture distribution
                if posture_stats:
                    ax2 = plt.subplot(gs[0, 1])
                    ax2.pie([posture_stats.get(x, 0) for x in ['Excellent', 'Good', 'Fair', 'Poor', 'Unknown']], 
                           labels=['Excellent', 'Good', 'Fair', 'Poor', 'Unknown'],
                           autopct='%1.1f%%', startangle=90)
                    ax2.set_title('Posture Distribution')
                
                # Emotion distribution
                if emotion_stats and self.enable_emotion:
                    ax3 = plt.subplot(gs[1, 0:])
                    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                    ax3.bar(emotions, [emotion_stats.get(e, 0) for e in emotions])
                    ax3.set_title('Emotion Distribution')
                    plt.xticks(rotation=45)
                
                # Conclusions and recommendations
                plt.text(0.1, 0.2, 'Overall Assessment:', fontsize=14, fontweight='bold')
                
                # Generate overall assessment
                assessment_text = []
                
                # Attention assessment
                if attention_stats.get('Focused', 0) > 70:
                    assessment_text.append("Excellent attention focus throughout the session.")
                elif attention_stats.get('Focused', 0) > 50:
                    assessment_text.append("Good attention focus with some distractions.")
                elif attention_stats.get('Distracted', 0) > 30:
                    assessment_text.append("Significant attention issues detected. Consider taking more breaks.")
                
                # Posture assessment
                if posture_stats.get('Poor', 0) > 30:
                    assessment_text.append("Poor posture detected frequently. Consider ergonomic improvements.")
                elif posture_stats.get('Excellent', 0) + posture_stats.get('Good', 0) > 70:
                    assessment_text.append("Maintained good posture throughout most of the session.")
                
                # Fatigue assessment
                if self.session_stats['fatigue_incidents'] > 5:
                    assessment_text.append("Multiple fatigue incidents detected. Consider more rest periods.")
                elif self.drowsiness_alerts > 10:
                    assessment_text.append("Drowsiness detected frequently. Consider taking a longer break.")
                
                # Emotion assessment
                if self.enable_emotion:
                    dominant_emotion = max(emotion_stats.items(), key=lambda x: x[1])[0] if emotion_stats else "neutral"
                    if dominant_emotion == 'neutral':
                        assessment_text.append("Maintained neutral emotional state throughout session.")
                    elif dominant_emotion == 'happy':
                        assessment_text.append("Positive emotional state detected for most of the session.")
                    elif dominant_emotion in ['angry', 'sad', 'fear']:
                        assessment_text.append(f"Predominantly {dominant_emotion} emotional state detected.")
                
                # Add default assessment if empty
                if not assessment_text:
                    assessment_text.append("Insufficient data to generate detailed assessment.")
                
                for i, text in enumerate(assessment_text):
                    plt.text(0.1, 0.15 - i*0.05, text, fontsize=12)
                
                pdf.savefig()
                plt.close()
                
                # ----- Detailed Analytics Pages -----
                if df is not None and len(df) > 3:
                    # Blink Analysis Page
                    plt.figure(figsize=(8.5, 11))
                    plt.subplot(2, 1, 1)
                    plt.plot(df['Blink_Rate'], marker='o', linestyle='-')
                    plt.axhline(y=self.fatigue_threshold, color='r', linestyle='--', alpha=0.7, label='Fatigue Threshold')
                    plt.title('Blink Rate Over Time')
                    plt.xlabel('Time Intervals')
                    plt.ylabel('Blinks per Minute')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    plt.subplot(2, 1, 2)
                    if 'Blink_Duration' in df.columns and 'Blink_Interval' in df.columns:
                        if len(df) > 1:
                            plt.plot(df['Blink_Duration'], marker='s', linestyle='-', label='Duration')
                            plt.plot(df['Blink_Interval'], marker='^', linestyle='-', label='Interval')
                            plt.title('Blink Duration and Interval')
                            plt.xlabel('Time Intervals')
                            plt.ylabel('Seconds')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    pdf.savefig()
                    plt.close()
                    
                    # Emotion Analysis Page
                    if self.enable_emotion:
                        plt.figure(figsize=(8.5, 11))
                        
                        # Emotion trends over time
                        plt.subplot(2, 1, 1)
                        emotion_cols = ['Angry_Score', 'Disgust_Score', 'Fear_Score', 
                                       'Happy_Score', 'Sad_Score', 'Surprise_Score', 'Neutral_Score']
                        
                        for col in emotion_cols:
                            plt.plot(df[col], label=col.replace('_Score', ''))
                            
                        plt.title('Emotion Scores Over Time')
                        plt.xlabel('Time Intervals')
                        plt.ylabel('Emotion Score')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        # Average emotion distribution
                        plt.subplot(2, 1, 2)
                        emotion_avgs = df[emotion_cols].mean()
                        emotion_avgs.plot(kind='bar', color='skyblue')
                        plt.title('Average Emotion Distribution')
                        plt.ylabel('Average Score')
                        plt.grid(True, axis='y', alpha=0.3)
                        
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                    
                    # Attention and Posture Analysis
                    if raw_df is not None and len(raw_df) > 10:
                        plt.figure(figsize=(8.5, 11))
                        
                        # Head angle analysis
                        if 'Horizontal_Angle' in raw_df.columns and 'Vertical_Angle' in raw_df.columns:
                            plt.subplot(2, 1, 1)
                            # Downsample if needed to avoid overcrowding
                            sample_rate = max(1, len(raw_df) // 100)
                            plt.plot(raw_df['Horizontal_Angle'][::sample_rate], label='Horizontal Angle')
                            plt.plot(raw_df['Vertical_Angle'][::sample_rate], label='Vertical Angle')
                            plt.title('Head Angle Over Time')
                            plt.xlabel('Frames')
                            plt.ylabel('Angle (degrees)')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                        
                        # Movement analysis
                        if 'Head_Movement' in raw_df.columns and 'Torso_Movement' in raw_df.columns:
                            plt.subplot(2, 1, 2)
                            # Downsample if needed to avoid overcrowding
                            sample_rate = max(1, len(raw_df) // 100)
                            plt.plot(raw_df['Head_Movement'][::sample_rate], label='Head Movement')
                            plt.plot(raw_df['Torso_Movement'][::sample_rate], label='Torso Movement')
                            plt.title('Movement Over Time')
                            plt.xlabel('Frames')
                            plt.ylabel('Movement Magnitude')
                            plt.legend()
                            plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
                    
                    # Performance Metrics Page
                    if raw_df is not None and 'FPS' in raw_df.columns and 'Processing_Time' in raw_df.columns:
                        plt.figure(figsize=(8.5, 11))
                        
                        plt.subplot(2, 1, 1)
                        # Downsample if needed to avoid overcrowding
                        sample_rate = max(1, len(raw_df) // 100)
                        plt.plot(raw_df['FPS'][::sample_rate])
                        plt.title('FPS Over Time')
                        plt.xlabel('Frames')
                        plt.ylabel('Frames Per Second')
                        plt.grid(True, alpha=0.3)
                        
                        plt.subplot(2, 1, 2)
                        plt.plot(raw_df['Processing_Time'][::sample_rate] * 1000)  # Convert to milliseconds
                        plt.title('Processing Time Per Frame')
                        plt.xlabel('Frames')
                        plt.ylabel('Processing Time (ms)')
                        plt.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        pdf.savefig()
                        plt.close()
            
            print(f"\nComprehensive report generated: {report_file}")
            return report_file
            
        except Exception as e:
            print(f"Error generating comprehensive report: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    """Main function to run the integrated analyzer with command line options"""
    import argparse
    import sys
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Integrated Feature Analysis')
    parser.add_argument('--no-emotion', action='store_true', help='Disable emotion detection')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--log', action='store_true', default=True, help='Enable data logging (enabled by default)')
    parser.add_argument('--no-log', action='store_true', help='Disable data logging')
    parser.add_argument('--log-interval', type=int, default=15, help='Logging interval in seconds (default: 15)')
    parser.add_argument('--report', action='store_true', help='Generate comprehensive PDF report at end of session')
    parser.add_argument('--debug', action='store_true', help='Start in debug mode')
    parser.add_argument('--resolution', type=str, default='640x480', help='Camera resolution (default: 640x480)')
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Invalid resolution format: {args.resolution}. Using default 640x480.")
        resolution = (640, 480)
    
    # Try to initialize the webcam with the specified index
    camera_id = args.camera
    cap = cv2.VideoCapture(camera_id)
    
    # If the specified camera fails, try other indices
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_id}")
        
        # Try alternative camera indices (0, 1, 2)
        for alt_camera_id in [0, 1, 2]:
            if alt_camera_id == camera_id:
                continue
                
            print(f"Trying camera index {alt_camera_id}...")
            cap = cv2.VideoCapture(alt_camera_id)
            
            if cap.isOpened():
                camera_id = alt_camera_id
                print(f"Successfully connected to camera {camera_id}")
                break
        
        # If no camera could be opened, exit
        if not cap.isOpened():
            print("Error: Could not open any webcam")
            print("\nTroubleshooting steps:")
            print("1. Run 'python Feature_extractor/list_cameras.py' to identify available cameras")
            print("2. Check if your camera is connected and not used by another application")
            
            # Mac-specific advice
            if platform.system() == 'Darwin':  # macOS
                print("\nOn macOS:")
                print("1. Check System Settings > Privacy & Security > Camera")
                print("2. Ensure Terminal has camera permissions")
                print("3. Try disconnecting and reconnecting your camera")
                print("4. Restart your computer if the issue persists")
            
            return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # Check if --no-log was provided (overrides the default --log)
    log_data = True
    if args.no_log:
        log_data = False
    
    # Initialize analyzer with appropriate settings
    analyzer = IntegratedFeatureAnalyzer(
        log_data=log_data, 
        log_interval=args.log_interval,
        camera_id=camera_id,
        resolution=resolution,
        enable_emotion=not args.no_emotion
    )
    
    # Store report flag for later use
    analyzer.generate_report = args.report
    
    # Enable debug mode if requested
    if args.debug:
        analyzer.debug = True
    
    print("\n===== Integrated Feature Analysis =====")
    print("Press 'q' to quit")
    print("Press 'd' to toggle debug visualizations")
    print("Press 'l' to toggle landmark display")
    print("Press 'r' to start/stop recording")
    print("Press 'g' to toggle analytics graphs")
    if log_data:
        print(f"Data will be logged to the 'logs' directory every {args.log_interval} seconds")
        if args.report:
            print("A comprehensive PDF report will be generated at the end of the session")
    else:
        print("Data logging is disabled")
    if not args.no_emotion:
        print("Emotion detection enabled (may require a few moments to initialize)")
    else:
        print("Emotion detection disabled")
    print("=========================================\n")
    
    try:
        consecutive_failures = 0
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            
            if not ret:
                consecutive_failures += 1
                print(f"Warning: Failed to capture image (attempt {consecutive_failures})")
                
                # If we have multiple consecutive failures, try to recover
                if consecutive_failures >= 5:
                    print("Attempting to recover connection to camera...")
                    cap.release()
                    cap = cv2.VideoCapture(camera_id)
                    
                    if not cap.isOpened():
                        print(f"Error: Could not reconnect to camera {camera_id}")
                        break
                    
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                    consecutive_failures = 0
                    
                # Skip this iteration and try again
                time.sleep(0.1)
                continue
            
            # Reset failure counter on successful capture
            consecutive_failures = 0
            
            # Mirror frame horizontally (more natural for webcam)
            frame = cv2.flip(frame, 1)
            
            # Process frame
            result_frame = analyzer.analyze_frame(frame)
            
            # Display the resulting frame
            cv2.imshow('Integrated Feature Analysis', result_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                # Toggle debug mode
                analyzer.debug = not analyzer.debug
                print(f"Debug mode {'enabled' if analyzer.debug else 'disabled'}")
            elif key == ord('l'):
                # Toggle landmark display
                analyzer.show_landmarks = not analyzer.show_landmarks
                print(f"Landmarks {'enabled' if analyzer.show_landmarks else 'disabled'}")
            elif key == ord('r'):
                # Start or stop recording
                analyzer.toggle_recording(frame)
            elif key == ord('g'):
                # Toggle analytics graphs
                analyzer.show_graphs = not analyzer.show_graphs
                print(f"Analytics graphs {'enabled' if analyzer.show_graphs else 'disabled'}")
    finally:
        # Clean up resources
        analyzer.cleanup()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
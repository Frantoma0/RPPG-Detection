#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    rPPG LIE DETECTION SYSTEM v2.1                            ║
║                    Professional Deception Analysis                            ║
║══════════════════════════════════════════════════════════════════════════════║
║  Използва Remote Photoplethysmography (rPPG) за извличане на                 ║
║  сърдечен ритъм от видео и анализ на физиологични маркери за стрес          ║
║                                                                              ║
║  Технологии: OpenCV, MediaPipe, NumPy, SciPy, Flask-SocketIO                 ║
║  Оптимизирано за: NVIDIA RTX 4070 + AMD Ryzen 9 HX                           ║
║                                                                              ║
║  v2.1 - Fixed MediaPipe compatibility (works with or without MediaPipe)      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
import threading
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Flask imports
from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit

# ══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE COMPATIBILITY LAYER
# ══════════════════════════════════════════════════════════════════════════════

MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_LEGACY = False
mp = None

try:
    import mediapipe
    mp = mediapipe
    # Check if legacy API is available (solutions.face_mesh)
    if hasattr(mp, 'solutions'):
        solutions = getattr(mp, 'solutions', None)
        if solutions and hasattr(solutions, 'face_mesh'):
            MEDIAPIPE_AVAILABLE = True
            MEDIAPIPE_LEGACY = True
            print("[INFO] MediaPipe Legacy API detected (solutions.face_mesh)")
except ImportError:
    print("[INFO] MediaPipe not installed")
except Exception as e:
    print(f"[INFO] MediaPipe check failed: {e}")

if not MEDIAPIPE_AVAILABLE:
    print("[INFO] Using OpenCV Haar Cascade for face detection (fallback mode)")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Системна конфигурация"""
    # Video settings
    CAMERA_ID: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    FPS: int = 30
    
    # rPPG settings
    BUFFER_SIZE: int = 300  # 10 seconds at 30fps
    MIN_HR: int = 40        # Minimum heart rate (BPM)
    MAX_HR: int = 180       # Maximum heart rate (BPM)
    
    # Signal processing
    BANDPASS_LOW: float = 0.7   # Hz (42 BPM)
    BANDPASS_HIGH: float = 3.0  # Hz (180 BPM)
    FILTER_ORDER: int = 5
    
    # HRV Analysis windows
    HRV_WINDOW: int = 30        # seconds for HRV calculation
    STRESS_WINDOW: int = 10     # seconds for stress averaging
    
    # Face ROI settings
    FOREHEAD_RATIO: float = 0.3  # Portion of forehead to use
    CHEEK_RATIO: float = 0.25    # Portion of cheeks to use
    
    # Stress thresholds
    STRESS_LOW: float = 0.3
    STRESS_MEDIUM: float = 0.5
    STRESS_HIGH: float = 0.7
    
    # Deception indicators
    HR_SPIKE_THRESHOLD: float = 15.0    # BPM increase
    HRV_DROP_THRESHOLD: float = 0.3     # 30% decrease
    BLINK_RATE_THRESHOLD: float = 25.0  # blinks per minute


config = Config()

# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PhysiologicalState:
    """Текущо физиологично състояние"""
    timestamp: float = 0.0
    
    # Heart Rate
    heart_rate: float = 0.0
    heart_rate_raw: float = 0.0
    hr_confidence: float = 0.0
    
    # HRV Metrics
    hrv_sdnn: float = 0.0       # Standard deviation of NN intervals
    hrv_rmssd: float = 0.0      # Root mean square of successive differences
    hrv_lf_hf_ratio: float = 0.0  # Low/High frequency ratio
    
    # Respiratory
    respiratory_rate: float = 0.0
    
    # Stress & Deception
    stress_level: float = 0.0
    deception_probability: float = 0.0
    
    # Face metrics
    blink_rate: float = 0.0
    micro_expressions: List[str] = None
    face_detected: bool = False
    
    # Signal quality
    signal_quality: float = 0.0
    
    def __post_init__(self):
        if self.micro_expressions is None:
            self.micro_expressions = []


@dataclass 
class QuestionSession:
    """Сесия с въпрос"""
    question_id: int
    question_text: str
    question_type: str  # 'neutral', 'control', 'relevant'
    start_time: float
    end_time: float = 0.0
    baseline_hr: float = 0.0
    peak_hr: float = 0.0
    avg_stress: float = 0.0
    deception_score: float = 0.0
    physiological_data: List[PhysiologicalState] = None
    
    def __post_init__(self):
        if self.physiological_data is None:
            self.physiological_data = []


# ══════════════════════════════════════════════════════════════════════════════
# rPPG SIGNAL PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════

class RPPGProcessor:
    """
    Remote Photoplethysmography процесор
    
    Извлича сърдечен ритъм от видео чрез анализ на микро-промени
    в цвета на кожата, причинени от кръвния поток.
    """
    
    def __init__(self):
        # Signal buffers
        self.green_signal = deque(maxlen=config.BUFFER_SIZE)
        self.red_signal = deque(maxlen=config.BUFFER_SIZE)
        self.blue_signal = deque(maxlen=config.BUFFER_SIZE)
        self.timestamps = deque(maxlen=config.BUFFER_SIZE)
        
        # Peak detection for HRV
        self.peak_times = deque(maxlen=100)
        self.rr_intervals = deque(maxlen=100)
        
        # Kalman filter state
        self.kalman_hr = None
        self.init_kalman_filter()
        
        # Results
        self.current_hr = 0.0
        self.current_hrv = {}
        self.signal_quality = 0.0
        
        # Processing stats
        self.frame_count = 0
        self.last_process_time = time.time()
        
    def init_kalman_filter(self):
        """Инициализация на Калман филтър за изглаждане на HR"""
        self.kalman_hr = cv2.KalmanFilter(2, 1)
        self.kalman_hr.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kalman_hr.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.kalman_hr.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 0.03
        self.kalman_hr.measurementNoiseCov = np.array([[1]], np.float32) * 0.5
        self.kalman_hr.statePost = np.array([[70], [0]], np.float32)
        
    def extract_roi_signal_from_rect(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]) -> Tuple[float, float, float]:
        """
        Извлича средни цветови стойности от ROI базирано на правоъгълник на лицето
        """
        x, y, w, h = face_rect
        frame_h, frame_w = frame.shape[:2]
        
        # Forehead region (top 30% of face)
        forehead_y = max(0, y)
        forehead_h = int(h * 0.3)
        forehead_x = x + int(w * 0.2)
        forehead_w = int(w * 0.6)
        
        forehead_x = max(0, min(forehead_x, frame_w - 1))
        forehead_w = min(forehead_w, frame_w - forehead_x)
        forehead_y = max(0, min(forehead_y, frame_h - 1))
        forehead_h = min(forehead_h, frame_h - forehead_y)
        
        if forehead_w <= 0 or forehead_h <= 0:
            return 0, 0, 0
            
        forehead_roi = frame[forehead_y:forehead_y+forehead_h, forehead_x:forehead_x+forehead_w]
        
        if forehead_roi.size == 0:
            return 0, 0, 0
            
        mean_color = np.mean(forehead_roi, axis=(0, 1))
        return mean_color[2], mean_color[1], mean_color[0]  # BGR to RGB
        
    def extract_roi_signal_from_landmarks(self, frame: np.ndarray, landmarks) -> Tuple[float, float, float]:
        """
        Извлича средни цветови стойности от Region of Interest (ROI)
        използвайки MediaPipe landmarks
        """
        h, w = frame.shape[:2]
        
        # Forehead ROI (между веждите)
        forehead_points = [10, 67, 69, 104, 108, 151, 337, 338, 297, 299]
        
        # Cheek ROIs
        left_cheek_points = [50, 101, 118, 117, 116, 123, 147, 213]
        right_cheek_points = [280, 330, 347, 346, 345, 352, 376, 433]
        
        def get_roi_mean(points):
            coords = []
            for idx in points:
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    x_coord, y_coord = int(lm.x * w), int(lm.y * h)
                    if 0 <= x_coord < w and 0 <= y_coord < h:
                        coords.append((x_coord, y_coord))
            
            if len(coords) < 3:
                return None
                
            coords = np.array(coords)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [coords], 255)
            
            # Extract mean color
            pixels = frame[mask > 0]
            
            if len(pixels) > 100:
                return np.mean(pixels, axis=0)
            return None
        
        # Get signals from multiple ROIs
        signals = []
        for points in [forehead_points, left_cheek_points, right_cheek_points]:
            mean_color = get_roi_mean(points)
            if mean_color is not None:
                signals.append(mean_color)
        
        if not signals:
            return 0, 0, 0
            
        # Average across ROIs
        avg_signal = np.mean(signals, axis=0)
        return avg_signal[2], avg_signal[1], avg_signal[0]  # BGR to RGB
        
    def add_sample(self, r: float, g: float, b: float, timestamp: float):
        """Добавя нова проба към буфера"""
        self.red_signal.append(r)
        self.green_signal.append(g)
        self.blue_signal.append(b)
        self.timestamps.append(timestamp)
        self.frame_count += 1
        
    def calculate_heart_rate(self) -> Tuple[float, float]:
        """
        Изчислява сърдечния ритъм чрез FFT анализ
        
        Returns:
            (heart_rate, confidence)
        """
        if len(self.green_signal) < config.BUFFER_SIZE // 2:
            return 0, 0
            
        # Use green channel (best for rPPG)
        signal_data = np.array(self.green_signal)
        timestamps = np.array(self.timestamps)
        
        # Calculate actual sampling rate
        if len(timestamps) < 2:
            return 0, 0
        dt = np.mean(np.diff(timestamps))
        if dt <= 0:
            return 0, 0
        fs = 1.0 / dt
        
        # Detrend signal
        signal_detrended = signal.detrend(signal_data)
        
        # Normalize
        signal_normalized = (signal_detrended - np.mean(signal_detrended)) / (np.std(signal_detrended) + 1e-10)
        
        # Bandpass filter
        try:
            nyq = fs / 2
            low = config.BANDPASS_LOW / nyq
            high = min(config.BANDPASS_HIGH / nyq, 0.99)
            
            if low >= high or low <= 0:
                return 0, 0
                
            b, a = signal.butter(config.FILTER_ORDER, [low, high], btype='band')
            signal_filtered = signal.filtfilt(b, a, signal_normalized)
        except Exception:
            return 0, 0
        
        # FFT
        n = len(signal_filtered)
        yf = fft(signal_filtered)
        xf = fftfreq(n, dt)
        
        # Get positive frequencies
        positive_mask = xf > 0
        xf_pos = xf[positive_mask]
        yf_pos = np.abs(yf[positive_mask])
        
        # Filter to valid HR range
        hr_mask = (xf_pos >= config.BANDPASS_LOW) & (xf_pos <= config.BANDPASS_HIGH)
        if not np.any(hr_mask):
            return 0, 0
            
        xf_hr = xf_pos[hr_mask]
        yf_hr = yf_pos[hr_mask]
        
        # Find dominant frequency
        peak_idx = np.argmax(yf_hr)
        peak_freq = xf_hr[peak_idx]
        peak_power = yf_hr[peak_idx]
        
        # Calculate confidence based on peak prominence
        total_power = np.sum(yf_hr)
        confidence = peak_power / (total_power + 1e-10) if total_power > 0 else 0
        
        # Convert to BPM
        heart_rate = peak_freq * 60
        
        # Apply Kalman filter
        self.kalman_hr.correct(np.array([[np.float32(heart_rate)]]))
        filtered_hr = self.kalman_hr.predict()[0, 0]
        
        # Update signal quality
        self.signal_quality = min(confidence * 2, 1.0)
        
        return float(filtered_hr), float(confidence)
        
    def detect_peaks(self, signal_data: np.ndarray, fs: float) -> np.ndarray:
        """Детектира R-пикове в сигнала за HRV анализ"""
        # Bandpass filter
        nyq = fs / 2
        low = config.BANDPASS_LOW / nyq
        high = min(config.BANDPASS_HIGH / nyq, 0.99)
        
        try:
            b, a = signal.butter(3, [low, high], btype='band')
            filtered = signal.filtfilt(b, a, signal_data)
        except:
            return np.array([])
        
        # Find peaks
        min_distance = int(fs * 0.5)  # Minimum 0.5 sec between beats
        peaks, _ = signal.find_peaks(filtered, distance=min_distance, prominence=0.1)
        
        return peaks
        
    def calculate_hrv_metrics(self) -> Dict[str, float]:
        """
        Изчислява HRV метрики
        """
        if len(self.green_signal) < config.BUFFER_SIZE:
            return {'sdnn': 0, 'rmssd': 0, 'lf_hf': 0}
            
        signal_data = np.array(self.green_signal)
        timestamps = np.array(self.timestamps)
        
        dt = np.mean(np.diff(timestamps))
        if dt <= 0:
            return {'sdnn': 0, 'rmssd': 0, 'lf_hf': 0}
        fs = 1.0 / dt
        
        # Detect peaks
        peaks = self.detect_peaks(signal_data, fs)
        
        if len(peaks) < 5:
            return {'sdnn': 0, 'rmssd': 0, 'lf_hf': 0}
            
        # Calculate RR intervals in ms
        peak_times = timestamps[peaks]
        rr_intervals = np.diff(peak_times) * 1000  # Convert to ms
        
        # Filter outliers
        rr_mean = np.mean(rr_intervals)
        rr_std = np.std(rr_intervals)
        valid_rr = rr_intervals[(rr_intervals > rr_mean - 2*rr_std) & 
                                (rr_intervals < rr_mean + 2*rr_std)]
        
        if len(valid_rr) < 3:
            return {'sdnn': 0, 'rmssd': 0, 'lf_hf': 0}
            
        # Time domain metrics
        sdnn = np.std(valid_rr)
        rmssd = np.sqrt(np.mean(np.diff(valid_rr)**2))
        
        # Frequency domain (LF/HF ratio)
        try:
            rr_interp = np.interp(
                np.linspace(peak_times[0], peak_times[-1], len(valid_rr)*4),
                peak_times[:-1],
                valid_rr
            )
            
            f, psd = signal.welch(rr_interp, fs=4, nperseg=min(len(rr_interp), 256))
            
            lf_mask = (f >= 0.04) & (f < 0.15)
            hf_mask = (f >= 0.15) & (f < 0.4)
            
            lf_power = np.trapz(psd[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0
            hf_power = np.trapz(psd[hf_mask], f[hf_mask]) if np.any(hf_mask) else 0
            
            lf_hf_ratio = lf_power / (hf_power + 1e-10)
        except:
            lf_hf_ratio = 1.0
            
        return {
            'sdnn': float(sdnn),
            'rmssd': float(rmssd),
            'lf_hf': float(lf_hf_ratio)
        }
        
    def calculate_respiratory_rate(self) -> float:
        """Изчислява дихателна честота от модулацията на rPPG сигнала"""
        if len(self.green_signal) < config.BUFFER_SIZE:
            return 0
            
        signal_data = np.array(self.green_signal)
        timestamps = np.array(self.timestamps)
        
        dt = np.mean(np.diff(timestamps))
        if dt <= 0:
            return 0
        fs = 1.0 / dt
        
        try:
            nyq = fs / 2
            low = 0.1 / nyq
            high = min(0.5 / nyq, 0.99)
            
            if low >= high or low <= 0:
                return 0
                
            b, a = signal.butter(3, [low, high], btype='band')
            resp_signal = signal.filtfilt(b, a, signal_data)
            
            n = len(resp_signal)
            yf = np.abs(fft(resp_signal))
            xf = fftfreq(n, dt)
            
            resp_mask = (xf > 0.1) & (xf < 0.5)
            if not np.any(resp_mask):
                return 0
                
            xf_resp = xf[resp_mask]
            yf_resp = yf[resp_mask]
            
            peak_freq = xf_resp[np.argmax(yf_resp)]
            return float(peak_freq * 60)
        except:
            return 0


# ══════════════════════════════════════════════════════════════════════════════
# FACE ANALYZER (with OpenCV fallback)
# ══════════════════════════════════════════════════════════════════════════════

class FaceAnalyzer:
    """
    Анализатор на лицеви изражения и микро-движения
    
    Поддържа:
    - MediaPipe Face Mesh (ако е наличен)
    - OpenCV Haar Cascade (fallback)
    """
    
    def __init__(self):
        self.use_mediapipe = MEDIAPIPE_AVAILABLE and MEDIAPIPE_LEGACY
        self.face_mesh = None
        self.face_cascade = None
        self.eye_cascade = None
        
        if self.use_mediapipe:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7
                )
                print("[INFO] Using MediaPipe Face Mesh")
            except Exception as e:
                print(f"[WARNING] MediaPipe Face Mesh failed: {e}")
                self.use_mediapipe = False
        
        if not self.use_mediapipe:
            # Fallback to OpenCV Haar Cascade
            print("[INFO] Using OpenCV Haar Cascade (fallback)")
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        
        # Blink detection
        self.blink_timestamps = deque(maxlen=100)
        self.last_blink_time = 0
        self.eye_closed = False
        self.blink_threshold = 0.2
        
        # Eye landmarks (MediaPipe)
        self.LEFT_EYE_TOP = 159
        self.LEFT_EYE_BOTTOM = 145
        self.RIGHT_EYE_TOP = 386
        self.RIGHT_EYE_BOTTOM = 374
        
        # Eye state for OpenCV fallback
        self.prev_eyes_detected = True
        
        # Micro-expression tracking
        self.expression_buffer = deque(maxlen=30)
        
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[object], Dict]:
        """Обработва кадър и връща landmarks/face_rect + метрики"""
        metrics = {
            'face_detected': False,
            'blink_detected': False,
            'blink_rate': 0,
            'ear_left': 0,
            'ear_right': 0,
            'expression_asymmetry': 0,
            'micro_expressions': [],
            'face_rect': None
        }
        
        if self.use_mediapipe:
            return self._process_mediapipe(frame, metrics)
        else:
            return self._process_opencv(frame, metrics)
            
    def _process_mediapipe(self, frame: np.ndarray, metrics: Dict) -> Tuple[Optional[object], Dict]:
        """Обработка с MediaPipe"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            metrics['face_detected'] = True
            
            # Eye Aspect Ratio (EAR) for blink detection
            metrics['ear_left'] = self._calculate_ear_mediapipe(landmarks, 'left')
            metrics['ear_right'] = self._calculate_ear_mediapipe(landmarks, 'right')
            
            avg_ear = (metrics['ear_left'] + metrics['ear_right']) / 2
            
            # Blink detection
            current_time = time.time()
            if avg_ear < self.blink_threshold:
                if not self.eye_closed:
                    self.eye_closed = True
                    if current_time - self.last_blink_time > 0.1:
                        self.blink_timestamps.append(current_time)
                        self.last_blink_time = current_time
                        metrics['blink_detected'] = True
            else:
                self.eye_closed = False
                
            # Calculate blink rate (blinks per minute)
            recent_blinks = [t for t in self.blink_timestamps 
                          if current_time - t < 60]
            metrics['blink_rate'] = len(recent_blinks)
            
            # Expression asymmetry
            metrics['expression_asymmetry'] = self._calculate_asymmetry_mediapipe(landmarks)
            
            # Micro-expression detection
            metrics['micro_expressions'] = self._detect_micro_expressions_mediapipe(landmarks)
            
            return landmarks, metrics
            
        return None, metrics
        
    def _process_opencv(self, frame: np.ndarray, metrics: Dict) -> Tuple[Optional[Tuple], Dict]:
        """Обработка с OpenCV Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(100, 100)
        )
        
        if len(faces) > 0:
            # Use largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            metrics['face_detected'] = True
            metrics['face_rect'] = (x, y, w, h)
            
            # Detect eyes within face region
            roi_gray = gray[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, minSize=(20, 20))
            
            # Blink detection based on eye visibility
            current_time = time.time()
            eyes_detected = len(eyes) >= 2
            
            if self.prev_eyes_detected and not eyes_detected:
                if current_time - self.last_blink_time > 0.1:
                    self.blink_timestamps.append(current_time)
                    self.last_blink_time = current_time
                    metrics['blink_detected'] = True
                    
            self.prev_eyes_detected = eyes_detected
            
            # Calculate blink rate
            recent_blinks = [t for t in self.blink_timestamps 
                          if current_time - t < 60]
            metrics['blink_rate'] = len(recent_blinks)
            
            # Simplified EAR based on eye detection
            metrics['ear_left'] = 0.3 if eyes_detected else 0.1
            metrics['ear_right'] = 0.3 if eyes_detected else 0.1
            
            return face, metrics
            
        return None, metrics
        
    def _calculate_ear_mediapipe(self, landmarks, side: str) -> float:
        """Calculate Eye Aspect Ratio (MediaPipe)"""
        if side == 'left':
            top = landmarks.landmark[self.LEFT_EYE_TOP]
            bottom = landmarks.landmark[self.LEFT_EYE_BOTTOM]
        else:
            top = landmarks.landmark[self.RIGHT_EYE_TOP]
            bottom = landmarks.landmark[self.RIGHT_EYE_BOTTOM]
            
        return abs(top.y - bottom.y)
        
    def _calculate_asymmetry_mediapipe(self, landmarks) -> float:
        """Изчислява асиметрия на лицето (MediaPipe)"""
        left_mouth = landmarks.landmark[61]
        right_mouth = landmarks.landmark[291]
        
        left_brow = landmarks.landmark[70]
        right_brow = landmarks.landmark[300]
        
        mouth_asymmetry = abs(left_mouth.y - right_mouth.y)
        brow_asymmetry = abs(left_brow.y - right_brow.y)
        
        return (mouth_asymmetry + brow_asymmetry) / 2
        
    def _detect_micro_expressions_mediapipe(self, landmarks) -> List[str]:
        """Детектира микро-изражения (MediaPipe)"""
        expressions = []
        
        left_brow = landmarks.landmark[70]
        right_brow = landmarks.landmark[300]
        brow_center = landmarks.landmark[9]
        
        if (left_brow.y < brow_center.y - 0.02 and 
            right_brow.y < brow_center.y - 0.02):
            expressions.append('brow_furrow')
            
        upper_lip = landmarks.landmark[13]
        lower_lip = landmarks.landmark[14]
        if abs(upper_lip.y - lower_lip.y) < 0.01:
            expressions.append('lip_compression')
            
        ear_left = self._calculate_ear_mediapipe(landmarks, 'left')
        ear_right = self._calculate_ear_mediapipe(landmarks, 'right')
        if ear_left < 0.15 and ear_right < 0.15:
            expressions.append('eye_squint')
            
        return expressions
        
    def close(self):
        if self.face_mesh:
            self.face_mesh.close()


# ══════════════════════════════════════════════════════════════════════════════
# STRESS & DECEPTION ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class DeceptionAnalyzer:
    """Анализатор на стрес и индикатори за измама"""
    
    def __init__(self):
        self.baseline_hr = None
        self.baseline_hrv = None
        self.baseline_blink_rate = None
        
        self.calibration_data = []
        self.is_calibrated = False
        
        self.stress_history = deque(maxlen=300)
        self.hr_history = deque(maxlen=300)
        
    def calibrate(self, states: List[PhysiologicalState]):
        """Калибрира базовите стойности от неутрално състояние"""
        if len(states) < 30:
            return False
            
        valid_states = [s for s in states if s.heart_rate > 0]
        if len(valid_states) < 15:
            return False
            
        self.baseline_hr = np.mean([s.heart_rate for s in valid_states])
        self.baseline_hrv = np.mean([s.hrv_sdnn for s in valid_states])
        self.baseline_blink_rate = np.mean([s.blink_rate for s in valid_states])
        
        self.is_calibrated = True
        return True
        
    def analyze(self, state: PhysiologicalState) -> Tuple[float, float]:
        """Анализира текущото състояние"""
        if not self.is_calibrated or state.heart_rate <= 0:
            return 0.0, 0.0
            
        self.hr_history.append(state.heart_rate)
        
        indicators = []
        
        # HR elevation
        hr_deviation = (state.heart_rate - self.baseline_hr) / self.baseline_hr
        hr_stress = min(max(hr_deviation * 2, 0), 1)
        indicators.append(('hr_elevation', hr_stress, 0.25))
        
        # HRV decrease
        if self.baseline_hrv > 0 and state.hrv_rmssd > 0:
            hrv_ratio = state.hrv_rmssd / self.baseline_hrv
            hrv_stress = 1 - min(hrv_ratio, 1)
            indicators.append(('hrv_decrease', hrv_stress, 0.2))
        
        # LF/HF ratio
        if state.hrv_lf_hf_ratio > 0:
            lf_hf_stress = min(state.hrv_lf_hf_ratio / 4, 1)
            indicators.append(('lf_hf_ratio', lf_hf_stress, 0.15))
        
        # Blink rate anomaly
        if self.baseline_blink_rate > 0:
            blink_deviation = abs(state.blink_rate - self.baseline_blink_rate)
            blink_stress = min(blink_deviation / self.baseline_blink_rate, 1)
            indicators.append(('blink_anomaly', blink_stress, 0.15))
        
        # Micro-expressions
        micro_score = len(state.micro_expressions) / 3
        indicators.append(('micro_expressions', min(micro_score, 1), 0.15))
        
        # HR variability
        if len(self.hr_history) > 10:
            recent_hr = list(self.hr_history)[-10:]
            hr_variability = np.std(recent_hr) / (np.mean(recent_hr) + 1e-10)
            var_stress = min(hr_variability * 5, 1)
            indicators.append(('hr_variability', var_stress, 0.1))
        
        total_weight = sum(i[2] for i in indicators)
        stress_level = sum(i[1] * i[2] for i in indicators) / total_weight
        
        self.stress_history.append(stress_level)
        
        # Deception probability
        deception_indicators = []
        
        if len(self.hr_history) > 30:
            recent_avg = np.mean(list(self.hr_history)[-30:])
            hr_spike = max(0, state.heart_rate - recent_avg - 5) / config.HR_SPIKE_THRESHOLD
            deception_indicators.append(min(hr_spike, 1))
        
        deception_indicators.append(stress_level)
        
        suspicious_expressions = ['lip_compression', 'eye_squint']
        suspicious_count = sum(1 for e in state.micro_expressions 
                              if e in suspicious_expressions)
        deception_indicators.append(min(suspicious_count / 2, 1))
        
        if self.baseline_blink_rate > 0:
            blink_ratio = state.blink_rate / self.baseline_blink_rate
            if blink_ratio < 0.5 or blink_ratio > 2:
                deception_indicators.append(0.7)
            else:
                deception_indicators.append(0.2)
        
        deception_probability = np.mean(deception_indicators) if deception_indicators else 0
        
        return float(stress_level), float(deception_probability)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class LieDetectorApp:
    """Главно приложение"""
    
    def __init__(self):
        self.rppg = RPPGProcessor()
        self.face_analyzer = FaceAnalyzer()
        self.deception_analyzer = DeceptionAnalyzer()
        
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.processed_frame = None
        
        self.current_state = PhysiologicalState()
        self.state_history = deque(maxlen=1800)
        
        self.session_start = None
        self.questions = []
        self.current_question: Optional[QuestionSession] = None
        
        self.calibration_states = []
        self.is_calibrating = False
        
        self.lock = threading.Lock()
        
    def start_camera(self) -> bool:
        """Стартира камерата"""
        self.cap = cv2.VideoCapture(config.CAMERA_ID, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(config.CAMERA_ID)
            
        if not self.cap.isOpened():
            return False
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
        
        self.is_running = True
        self.session_start = time.time()
        
        return True
        
    def stop_camera(self):
        """Спира камерата"""
        self.is_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
            
    def process_frame(self) -> Optional[np.ndarray]:
        """Обработва един кадър"""
        if not self.cap or not self.is_running:
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        timestamp = time.time()
        self.current_frame = frame.copy()
        
        face_data, face_metrics = self.face_analyzer.process_frame(frame)
        
        state = PhysiologicalState(timestamp=timestamp)
        state.face_detected = face_metrics['face_detected']
        state.blink_rate = face_metrics['blink_rate']
        state.micro_expressions = face_metrics.get('micro_expressions', [])
        
        if face_data is not None:
            if self.face_analyzer.use_mediapipe:
                r, g, b = self.rppg.extract_roi_signal_from_landmarks(frame, face_data)
            else:
                r, g, b = self.rppg.extract_roi_signal_from_rect(frame, face_metrics['face_rect'])
                
            if g > 0:
                self.rppg.add_sample(r, g, b, timestamp)
                
                hr, confidence = self.rppg.calculate_heart_rate()
                state.heart_rate = hr
                state.hr_confidence = confidence
                state.signal_quality = self.rppg.signal_quality
                
                hrv = self.rppg.calculate_hrv_metrics()
                state.hrv_sdnn = hrv['sdnn']
                state.hrv_rmssd = hrv['rmssd']
                state.hrv_lf_hf_ratio = hrv['lf_hf']
                
                state.respiratory_rate = self.rppg.calculate_respiratory_rate()
                
                if self.deception_analyzer.is_calibrated:
                    stress, deception = self.deception_analyzer.analyze(state)
                    state.stress_level = stress
                    state.deception_probability = deception
                    
        with self.lock:
            self.current_state = state
            self.state_history.append(state)
            
            if self.is_calibrating:
                self.calibration_states.append(state)
                
            if self.current_question:
                self.current_question.physiological_data.append(state)
                
        self.processed_frame = self._draw_visualization(frame, face_data, face_metrics, state)
        
        return self.processed_frame
        
    def _draw_visualization(self, frame: np.ndarray, face_data, face_metrics: Dict,
                           state: PhysiologicalState) -> np.ndarray:
        """Рисува визуализация върху кадъра"""
        vis_frame = frame.copy()
        h, w = vis_frame.shape[:2]
        
        if face_data is not None:
            if self.face_analyzer.use_mediapipe:
                self._draw_roi_areas_mediapipe(vis_frame, face_data)
            else:
                self._draw_roi_areas_opencv(vis_frame, face_metrics['face_rect'])
        
        overlay = vis_frame.copy()
        
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0, vis_frame)
        
        hr_text = f"HR: {state.heart_rate:.0f} BPM" if state.heart_rate > 0 else "HR: --"
        hr_color = (0, 255, 0) if 60 < state.heart_rate < 100 else (0, 255, 255)
        cv2.putText(vis_frame, hr_text, (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, hr_color, 2)
        
        quality_color = (0, 255, 0) if state.signal_quality > 0.5 else (0, 165, 255)
        cv2.putText(vis_frame, f"Signal: {state.signal_quality*100:.0f}%", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, quality_color, 1)
        
        cv2.putText(vis_frame, f"HRV: {state.hrv_rmssd:.1f}ms", (200, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        cv2.putText(vis_frame, f"RR: {state.respiratory_rate:.0f}/min", (200, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        stress_x = 400
        cv2.putText(vis_frame, "STRESS:", (stress_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        bar_width = 150
        bar_height = 20
        cv2.rectangle(vis_frame, (stress_x, 40), (stress_x + bar_width, 40 + bar_height),
                     (100, 100, 100), 1)
        
        stress_fill = int(state.stress_level * bar_width)
        stress_color = (0, 255, 0) if state.stress_level < 0.3 else \
                      (0, 255, 255) if state.stress_level < 0.6 else (0, 0, 255)
        cv2.rectangle(vis_frame, (stress_x, 40), 
                     (stress_x + stress_fill, 40 + bar_height), stress_color, -1)
        
        dec_x = 600
        cv2.putText(vis_frame, "DECEPTION:", (dec_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        dec_fill = int(state.deception_probability * bar_width)
        dec_color = (0, 255, 0) if state.deception_probability < 0.3 else \
                   (0, 255, 255) if state.deception_probability < 0.6 else (0, 0, 255)
        cv2.rectangle(vis_frame, (dec_x, 40), (dec_x + bar_width, 40 + bar_height),
                     (100, 100, 100), 1)
        cv2.rectangle(vis_frame, (dec_x, 40), 
                     (dec_x + dec_fill, 40 + bar_height), dec_color, -1)
        
        cv2.putText(vis_frame, f"Blinks: {state.blink_rate:.0f}/min", (800, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        if state.micro_expressions:
            cv2.putText(vis_frame, f"Micro: {', '.join(state.micro_expressions)}", 
                       (800, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)
        
        method = "MediaPipe" if self.face_analyzer.use_mediapipe else "OpenCV"
        cv2.putText(vis_frame, f"[{method}]", (w - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        if not state.face_detected:
            cv2.putText(vis_frame, "NO FACE DETECTED", (w//2 - 150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                       
        if self.is_calibrating:
            cv2.putText(vis_frame, "CALIBRATING...", (w//2 - 100, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif not self.deception_analyzer.is_calibrated:
            cv2.putText(vis_frame, "Press 'C' to calibrate", (w//2 - 120, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        return vis_frame
        
    def _draw_roi_areas_mediapipe(self, frame: np.ndarray, landmarks):
        """Рисува ROI областите за rPPG (MediaPipe)"""
        h, w = frame.shape[:2]
        
        forehead_points = [10, 67, 69, 104, 108, 151, 337, 338, 297, 299]
        coords = []
        for idx in forehead_points:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                x, y = int(lm.x * w), int(lm.y * h)
                coords.append((x, y))
        
        if len(coords) > 2:
            coords = np.array(coords)
            cv2.polylines(frame, [coords], True, (0, 255, 0), 1)
            
    def _draw_roi_areas_opencv(self, frame: np.ndarray, face_rect: Tuple[int, int, int, int]):
        """Рисува ROI областите за rPPG (OpenCV)"""
        x, y, w, h = face_rect
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
        
        forehead_y = y
        forehead_h = int(h * 0.3)
        forehead_x = x + int(w * 0.2)
        forehead_w = int(w * 0.6)
        cv2.rectangle(frame, (forehead_x, forehead_y), 
                     (forehead_x + forehead_w, forehead_y + forehead_h), 
                     (0, 255, 255), 1)
            
    def start_calibration(self):
        self.calibration_states = []
        self.is_calibrating = True
        
    def finish_calibration(self) -> bool:
        self.is_calibrating = False
        return self.deception_analyzer.calibrate(self.calibration_states)
        
    def start_question(self, question_id: int, text: str, q_type: str):
        self.current_question = QuestionSession(
            question_id=question_id,
            question_text=text,
            question_type=q_type,
            start_time=time.time(),
            baseline_hr=self.current_state.heart_rate
        )
        
    def end_question(self):
        if self.current_question:
            self.current_question.end_time = time.time()
            
            if self.current_question.physiological_data:
                hrs = [s.heart_rate for s in self.current_question.physiological_data 
                      if s.heart_rate > 0]
                stresses = [s.stress_level for s in self.current_question.physiological_data]
                deceptions = [s.deception_probability for s in self.current_question.physiological_data]
                
                if hrs:
                    self.current_question.peak_hr = max(hrs)
                if stresses:
                    self.current_question.avg_stress = np.mean(stresses)
                if deceptions:
                    self.current_question.deception_score = np.mean(deceptions)
                    
            self.questions.append(self.current_question)
            self.current_question = None
            
    def get_state_dict(self) -> dict:
        with self.lock:
            return asdict(self.current_state)
            
    def cleanup(self):
        self.stop_camera()
        self.face_analyzer.close()


# ══════════════════════════════════════════════════════════════════════════════
# FLASK APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rppg-lie-detector-secret-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

detector: Optional[LieDetectorApp] = None
streaming_thread: Optional[threading.Thread] = None


def stream_data():
    global detector
    
    while detector and detector.is_running:
        frame = detector.process_frame()
        
        if frame is not None:
            state = detector.get_state_dict()
            socketio.emit('physiological_update', state)
            
        socketio.sleep(0.033)


@app.route('/')
def index():
    return render_template('dashboard.html')


@app.route('/video_feed')
def video_feed():
    def generate():
        global detector
        while detector and detector.is_running:
            if detector.processed_frame is not None:
                ret, buffer = cv2.imencode('.jpg', detector.processed_frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           buffer.tobytes() + b'\r\n')
            time.sleep(0.033)
            
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_status', {'status': 'connected'})


@socketio.on('start_session')
def handle_start():
    global detector, streaming_thread
    
    if detector is None:
        detector = LieDetectorApp()
        
    if detector.start_camera():
        streaming_thread = threading.Thread(target=stream_data)
        streaming_thread.daemon = True
        streaming_thread.start()
        emit('session_status', {'status': 'started'})
    else:
        emit('session_status', {'status': 'error', 'message': 'Camera not found'})


@socketio.on('stop_session')
def handle_stop():
    global detector
    if detector:
        detector.cleanup()
        detector = None
    emit('session_status', {'status': 'stopped'})


@socketio.on('start_calibration')
def handle_calibration_start():
    global detector
    if detector:
        detector.start_calibration()
        emit('calibration_status', {'status': 'started'})
        
        def finish():
            time.sleep(5)
            if detector:
                success = detector.finish_calibration()
                socketio.emit('calibration_status', {
                    'status': 'completed' if success else 'failed',
                    'baseline_hr': detector.deception_analyzer.baseline_hr if success else 0
                })
                
        threading.Thread(target=finish).start()


@socketio.on('start_question')
def handle_question_start(data):
    global detector
    if detector:
        detector.start_question(
            data.get('id', 0),
            data.get('text', ''),
            data.get('type', 'neutral')
        )
        emit('question_status', {'status': 'started', 'id': data.get('id')})


@socketio.on('end_question')
def handle_question_end():
    global detector
    if detector:
        detector.end_question()
        if detector.questions:
            last_q = detector.questions[-1]
            emit('question_result', {
                'id': last_q.question_id,
                'avg_stress': last_q.avg_stress,
                'deception_score': last_q.deception_score,
                'peak_hr': last_q.peak_hr
            })


@socketio.on('get_session_report')
def handle_report():
    global detector
    if detector and detector.questions:
        report = {
            'questions': [
                {
                    'id': q.question_id,
                    'text': q.question_text,
                    'type': q.question_type,
                    'deception_score': q.deception_score,
                    'avg_stress': q.avg_stress,
                    'peak_hr': q.peak_hr
                }
                for q in detector.questions
            ]
        }
        emit('session_report', report)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                    rPPG LIE DETECTION SYSTEM v2.1                            ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  Стартиране на сървър...                                                     ║
    ║  Отвори браузър на: http://localhost:5000                                    ║
    ║                                                                              ║
    ║  Функции:                                                                    ║
    ║  • rPPG анализ на сърдечен ритъм от камера                                   ║
    ║  • HRV метрики (SDNN, RMSSD, LF/HF)                                          ║
    ║  • Детекция на микро-изражения                                               ║
    ║  • Анализ на стрес и индикатори за измама                                    ║
    ║  • CQT протокол за въпроси                                                   ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if MEDIAPIPE_AVAILABLE:
        print("    [✓] Face detection: MediaPipe Face Mesh (468 landmarks)")
    else:
        print("    [!] Face detection: OpenCV Haar Cascade (fallback)")
    print()
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    main()

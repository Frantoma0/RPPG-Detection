#!/usr/bin/env python3
"""
rPPG LIE DETECTION SYSTEM v3.0 - STREAMLIT EDITION

Optimized for NVIDIA RTX 4070 Laptop GPU + AMD Ryzen 9 HX
Features:
- GPU-accelerated signal processing (CUDA/CuPy)
- Streamlit for smooth, real-time UI
- Optimized MediaPipe face detection
- Advanced FFT with GPU acceleration
- Real-time charts with Plotly
- Multi-threaded processing pipeline
"""

import cv2
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
import warnings
import queue

warnings.filterwarnings('ignore')

# Streamlit imports
import streamlit as st

# Check for GPU acceleration
GPU_AVAILABLE = False
cp = None

try:
    import cupy as cp
    GPU_AVAILABLE = cp.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"[GPU] CuPy CUDA acceleration enabled - {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError:
    pass

# Try PyTorch as fallback for GPU
TORCH_GPU = False
torch = None
try:
    import torch
    TORCH_GPU = torch.cuda.is_available()
    if TORCH_GPU and not GPU_AVAILABLE:
        print(f"[GPU] PyTorch CUDA enabled - {torch.cuda.get_device_name(0)}")
except (ImportError, OSError, Exception):
    # PyTorch may be corrupted or incompletely installed - skip it
    torch = None
    TORCH_GPU = False

# MediaPipe
MEDIAPIPE_AVAILABLE = False
mp = None
try:
    import mediapipe
    mp = mediapipe
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        MEDIAPIPE_AVAILABLE = True
except:
    pass

# ==============================================================================
# CONFIGURATION - Optimized for RTX 4070
# ==============================================================================

@dataclass
class Config:
    """System configuration optimized for RTX 4070"""
    # Video settings - Higher resolution for better accuracy
    CAMERA_ID: int = 0
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    TARGET_FPS: int = 30

    # rPPG settings - Larger buffer for better accuracy
    BUFFER_SIZE: int = 300  # 10 seconds at 30fps
    MIN_SAMPLES_FOR_HR: int = 90  # Minimum 3 seconds

    # Heart rate range
    MIN_HR: int = 40
    MAX_HR: int = 180

    # Signal processing - Optimized bandpass
    BANDPASS_LOW: float = 0.7   # Hz (42 BPM)
    BANDPASS_HIGH: float = 3.0  # Hz (180 BPM)
    FILTER_ORDER: int = 4

    # GPU batch processing
    GPU_BATCH_SIZE: int = 64

    # Kalman filter parameters - Fine-tuned
    KALMAN_PROCESS_NOISE: float = 0.01
    KALMAN_MEASUREMENT_NOISE: float = 0.3

    # ROI weights for multi-region fusion
    FOREHEAD_WEIGHT: float = 0.5
    LEFT_CHEEK_WEIGHT: float = 0.25
    RIGHT_CHEEK_WEIGHT: float = 0.25

    # Stress thresholds
    STRESS_LOW: float = 0.3
    STRESS_MEDIUM: float = 0.5
    STRESS_HIGH: float = 0.7


config = Config()


# ==============================================================================
# GPU-ACCELERATED SIGNAL PROCESSING
# ==============================================================================

class GPUSignalProcessor:
    """GPU-accelerated signal processing for RTX 4070"""

    def __init__(self):
        self.use_gpu = GPU_AVAILABLE or TORCH_GPU
        self.device = 'cuda' if TORCH_GPU else 'cpu'

    def to_gpu(self, arr: np.ndarray):
        """Transfer array to GPU"""
        if GPU_AVAILABLE and cp is not None:
            return cp.asarray(arr)
        elif TORCH_GPU and torch is not None:
            return torch.from_numpy(arr).cuda()
        return arr

    def to_cpu(self, arr) -> np.ndarray:
        """Transfer array back to CPU"""
        if GPU_AVAILABLE and cp is not None and hasattr(arr, 'get'):
            return arr.get()
        elif TORCH_GPU and torch is not None and hasattr(arr, 'cpu'):
            return arr.cpu().numpy()
        return np.asarray(arr)

    def fft_gpu(self, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated FFT"""
        n = len(signal_data)

        if GPU_AVAILABLE and cp is not None:
            signal_gpu = cp.asarray(signal_data)
            fft_result = cp.fft.fft(signal_gpu)
            magnitudes = cp.abs(fft_result)
            return self.to_cpu(magnitudes), None
        elif TORCH_GPU and torch is not None:
            signal_torch = torch.from_numpy(signal_data.astype(np.float32)).cuda()
            fft_result = torch.fft.fft(signal_torch)
            magnitudes = torch.abs(fft_result)
            return magnitudes.cpu().numpy(), None
        else:
            fft_result = fft(signal_data)
            return np.abs(fft_result), None

    def bandpass_filter(self, data: np.ndarray, fs: float,
                        low: float, high: float, order: int = 4) -> np.ndarray:
        """Optimized bandpass filter"""
        nyq = fs / 2
        low_norm = low / nyq
        high_norm = min(high / nyq, 0.99)

        if low_norm <= 0 or low_norm >= high_norm:
            return data

        try:
            # Use second-order sections for numerical stability
            sos = signal.butter(order, [low_norm, high_norm], btype='band', output='sos')
            return signal.sosfiltfilt(sos, data)
        except:
            return data

    def detrend_normalize(self, data: np.ndarray) -> np.ndarray:
        """Detrend and normalize signal"""
        # Remove linear trend
        detrended = signal.detrend(data)
        # Normalize
        std = np.std(detrended)
        if std > 1e-10:
            return (detrended - np.mean(detrended)) / std
        return detrended


gpu_processor = GPUSignalProcessor()


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class PhysiologicalState:
    """Current physiological state"""
    timestamp: float = 0.0

    # Heart Rate
    heart_rate: float = 0.0
    heart_rate_raw: float = 0.0
    hr_confidence: float = 0.0

    # HRV Metrics
    hrv_sdnn: float = 0.0
    hrv_rmssd: float = 0.0
    hrv_lf_hf_ratio: float = 0.0

    # Respiratory
    respiratory_rate: float = 0.0

    # Stress & Deception
    stress_level: float = 0.0
    deception_probability: float = 0.0

    # Face metrics
    blink_rate: float = 0.0
    micro_expressions: List[str] = field(default_factory=list)
    face_detected: bool = False

    # Signal quality
    signal_quality: float = 0.0


# ==============================================================================
# OPTIMIZED rPPG PROCESSOR
# ==============================================================================

class OptimizedRPPGProcessor:
    """
    GPU-Optimized rPPG Processor for RTX 4070

    Features:
    - Multi-ROI signal fusion
    - GPU-accelerated FFT
    - Advanced Kalman filtering
    - Adaptive signal quality estimation
    """

    def __init__(self):
        # Multi-channel signal buffers
        self.green_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.red_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.blue_buffer = deque(maxlen=config.BUFFER_SIZE)
        self.timestamps = deque(maxlen=config.BUFFER_SIZE)

        # Multi-ROI buffers for fusion
        self.forehead_signal = deque(maxlen=config.BUFFER_SIZE)
        self.left_cheek_signal = deque(maxlen=config.BUFFER_SIZE)
        self.right_cheek_signal = deque(maxlen=config.BUFFER_SIZE)

        # Kalman filter for HR smoothing
        self.kalman = self._init_kalman()
        self.kalman_initialized = False

        # Results cache
        self.current_hr = 0.0
        self.hr_confidence = 0.0
        self.signal_quality = 0.0

        # Peak detection for HRV
        self.peak_times = deque(maxlen=100)

        # Adaptive baseline
        self.baseline_green = None
        self.frame_count = 0

    def _init_kalman(self):
        """Initialize Kalman filter for HR smoothing"""
        kalman = cv2.KalmanFilter(2, 1)
        kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        kalman.processNoiseCov = np.eye(2, dtype=np.float32) * config.KALMAN_PROCESS_NOISE
        kalman.measurementNoiseCov = np.array([[config.KALMAN_MEASUREMENT_NOISE]], np.float32)
        kalman.statePost = np.array([[70], [0]], np.float32)
        return kalman

    def add_sample(self, r: float, g: float, b: float, timestamp: float,
                   forehead: float = None, left_cheek: float = None, right_cheek: float = None):
        """Add new sample with optional multi-ROI data"""
        self.red_buffer.append(r)
        self.green_buffer.append(g)
        self.blue_buffer.append(b)
        self.timestamps.append(timestamp)

        # Multi-ROI signals
        if forehead is not None:
            self.forehead_signal.append(forehead)
        if left_cheek is not None:
            self.left_cheek_signal.append(left_cheek)
        if right_cheek is not None:
            self.right_cheek_signal.append(right_cheek)

        self.frame_count += 1

    def calculate_heart_rate(self) -> Tuple[float, float]:
        """
        Calculate heart rate using GPU-accelerated FFT

        Returns:
            (heart_rate_bpm, confidence)
        """
        if len(self.green_buffer) < config.MIN_SAMPLES_FOR_HR:
            return 0.0, 0.0

        # Get signals
        green = np.array(self.green_buffer)
        timestamps = np.array(self.timestamps)

        # Calculate sampling rate
        if len(timestamps) < 2:
            return 0.0, 0.0
        dt = np.mean(np.diff(timestamps))
        if dt <= 0:
            return 0.0, 0.0
        fs = 1.0 / dt

        # Fuse multi-ROI signals if available
        if len(self.forehead_signal) == len(green):
            forehead = np.array(self.forehead_signal)
            left = np.array(self.left_cheek_signal) if len(self.left_cheek_signal) == len(green) else green
            right = np.array(self.right_cheek_signal) if len(self.right_cheek_signal) == len(green) else green

            # Weighted fusion
            green = (forehead * config.FOREHEAD_WEIGHT +
                    left * config.LEFT_CHEEK_WEIGHT +
                    right * config.RIGHT_CHEEK_WEIGHT)

        # Preprocessing
        processed = gpu_processor.detrend_normalize(green)
        filtered = gpu_processor.bandpass_filter(
            processed, fs, config.BANDPASS_LOW, config.BANDPASS_HIGH, config.FILTER_ORDER
        )

        # GPU-accelerated FFT
        n = len(filtered)
        magnitudes, _ = gpu_processor.fft_gpu(filtered)
        freqs = fftfreq(n, dt)

        # Get positive frequencies in HR range
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        mags_pos = magnitudes[pos_mask]

        hr_mask = (freqs_pos >= config.BANDPASS_LOW) & (freqs_pos <= config.BANDPASS_HIGH)
        if not np.any(hr_mask):
            return 0.0, 0.0

        freqs_hr = freqs_pos[hr_mask]
        mags_hr = mags_pos[hr_mask]

        # Find dominant frequency
        peak_idx = np.argmax(mags_hr)
        peak_freq = freqs_hr[peak_idx]
        peak_power = mags_hr[peak_idx]

        # Calculate confidence from spectral purity
        total_power = np.sum(mags_hr)
        confidence = peak_power / (total_power + 1e-10)

        # Secondary peak check for harmonic rejection
        mags_hr_copy = mags_hr.copy()
        mags_hr_copy[max(0, peak_idx-3):min(len(mags_hr_copy), peak_idx+4)] = 0
        if len(mags_hr_copy) > 0:
            second_peak = np.max(mags_hr_copy)
            if second_peak > 0:
                ratio = peak_power / second_peak
                confidence *= min(ratio / 2, 1.0)

        # Convert to BPM
        hr_raw = peak_freq * 60

        # Apply Kalman filter
        if not self.kalman_initialized and 40 < hr_raw < 180:
            self.kalman.statePost = np.array([[hr_raw], [0]], np.float32)
            self.kalman_initialized = True

        self.kalman.correct(np.array([[np.float32(hr_raw)]]))
        hr_filtered = self.kalman.predict()[0, 0]

        # Clamp to valid range
        hr_filtered = np.clip(hr_filtered, config.MIN_HR, config.MAX_HR)

        self.current_hr = float(hr_filtered)
        self.hr_confidence = float(min(confidence * 2, 1.0))
        self.signal_quality = self.hr_confidence

        return self.current_hr, self.hr_confidence

    def calculate_hrv_metrics(self) -> Dict[str, float]:
        """Calculate HRV metrics"""
        if len(self.green_buffer) < config.BUFFER_SIZE:
            return {'sdnn': 0, 'rmssd': 0, 'lf_hf': 1.0}

        green = np.array(self.green_buffer)
        timestamps = np.array(self.timestamps)

        dt = np.mean(np.diff(timestamps))
        if dt <= 0:
            return {'sdnn': 0, 'rmssd': 0, 'lf_hf': 1.0}
        fs = 1.0 / dt

        # Filter and find peaks
        filtered = gpu_processor.bandpass_filter(green, fs, 0.7, 2.5, 3)

        min_dist = int(fs * 0.4)
        peaks, properties = signal.find_peaks(filtered, distance=min_dist, prominence=0.05)

        if len(peaks) < 5:
            return {'sdnn': 0, 'rmssd': 0, 'lf_hf': 1.0}

        # RR intervals in ms
        peak_times = timestamps[peaks]
        rr = np.diff(peak_times) * 1000

        # Remove outliers
        rr_median = np.median(rr)
        rr_valid = rr[(rr > rr_median * 0.5) & (rr < rr_median * 1.5)]

        if len(rr_valid) < 3:
            return {'sdnn': 0, 'rmssd': 0, 'lf_hf': 1.0}

        # Time domain
        sdnn = float(np.std(rr_valid))
        rmssd = float(np.sqrt(np.mean(np.diff(rr_valid)**2)))

        # Frequency domain (LF/HF ratio)
        try:
            rr_interp_times = np.linspace(peak_times[0], peak_times[-1], len(rr_valid) * 4)
            rr_interp = np.interp(rr_interp_times, peak_times[:-1], rr_valid)

            f, psd = signal.welch(rr_interp - np.mean(rr_interp), fs=4, nperseg=min(256, len(rr_interp)))

            lf_mask = (f >= 0.04) & (f < 0.15)
            hf_mask = (f >= 0.15) & (f < 0.4)

            lf = np.trapz(psd[lf_mask], f[lf_mask]) if np.any(lf_mask) else 0
            hf = np.trapz(psd[hf_mask], f[hf_mask]) if np.any(hf_mask) else 1e-10

            lf_hf = float(lf / hf)
        except:
            lf_hf = 1.0

        return {'sdnn': sdnn, 'rmssd': rmssd, 'lf_hf': lf_hf}

    def calculate_respiratory_rate(self) -> float:
        """Calculate respiratory rate from signal modulation"""
        if len(self.green_buffer) < config.BUFFER_SIZE:
            return 0.0

        green = np.array(self.green_buffer)
        timestamps = np.array(self.timestamps)

        dt = np.mean(np.diff(timestamps))
        if dt <= 0:
            return 0.0
        fs = 1.0 / dt

        # Respiratory band: 0.1-0.5 Hz (6-30 breaths/min)
        filtered = gpu_processor.bandpass_filter(green, fs, 0.1, 0.5, 2)

        # FFT for respiratory peak
        n = len(filtered)
        mags, _ = gpu_processor.fft_gpu(filtered)
        freqs = fftfreq(n, dt)

        resp_mask = (freqs > 0.1) & (freqs < 0.5)
        if not np.any(resp_mask):
            return 0.0

        freqs_resp = freqs[resp_mask]
        mags_resp = mags[resp_mask]

        peak_freq = freqs_resp[np.argmax(mags_resp)]
        return float(peak_freq * 60)


# ==============================================================================
# OPTIMIZED FACE ANALYZER
# ==============================================================================

class OptimizedFaceAnalyzer:
    """Optimized face detection and analysis"""

    def __init__(self):
        self.use_mediapipe = MEDIAPIPE_AVAILABLE
        self.face_mesh = None
        self.face_cascade = None

        if self.use_mediapipe:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.7
                )
            except:
                self.use_mediapipe = False

        if not self.use_mediapipe:
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )

        # Blink detection
        self.blink_times = deque(maxlen=100)
        self.last_blink = 0
        self.eyes_closed = False

        # Landmark indices
        self.FOREHEAD_POINTS = [10, 67, 69, 104, 108, 151, 337, 338, 297, 299]
        self.LEFT_CHEEK_POINTS = [50, 101, 118, 117, 116, 123, 147, 213]
        self.RIGHT_CHEEK_POINTS = [280, 330, 347, 346, 345, 352, 376, 433]
        self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM = 159, 145
        self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM = 386, 374

    def process(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame for face detection and ROI extraction"""
        result = {
            'face_detected': False,
            'landmarks': None,
            'face_rect': None,
            'forehead_signal': 0,
            'left_cheek_signal': 0,
            'right_cheek_signal': 0,
            'mean_rgb': (0, 0, 0),
            'blink_rate': 0,
            'blink_detected': False,
            'micro_expressions': []
        }

        if self.use_mediapipe:
            return self._process_mediapipe(frame, result)
        else:
            return self._process_opencv(frame, result)

    def _process_mediapipe(self, frame: np.ndarray, result: Dict) -> Dict:
        """Process with MediaPipe"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection = self.face_mesh.process(rgb)

        if not detection.multi_face_landmarks:
            return result

        landmarks = detection.multi_face_landmarks[0]
        result['face_detected'] = True
        result['landmarks'] = landmarks

        h, w = frame.shape[:2]

        # Extract multi-ROI signals
        forehead = self._get_roi_mean(frame, landmarks, self.FOREHEAD_POINTS, h, w)
        left_cheek = self._get_roi_mean(frame, landmarks, self.LEFT_CHEEK_POINTS, h, w)
        right_cheek = self._get_roi_mean(frame, landmarks, self.RIGHT_CHEEK_POINTS, h, w)

        if forehead is not None:
            result['forehead_signal'] = forehead[1]  # Green channel
            result['mean_rgb'] = (forehead[2], forehead[1], forehead[0])
        if left_cheek is not None:
            result['left_cheek_signal'] = left_cheek[1]
        if right_cheek is not None:
            result['right_cheek_signal'] = right_cheek[1]

        # Blink detection
        ear_left = self._calc_ear(landmarks, 'left')
        ear_right = self._calc_ear(landmarks, 'right')
        avg_ear = (ear_left + ear_right) / 2

        now = time.time()
        if avg_ear < 0.2:
            if not self.eyes_closed and now - self.last_blink > 0.1:
                self.blink_times.append(now)
                self.last_blink = now
                result['blink_detected'] = True
            self.eyes_closed = True
        else:
            self.eyes_closed = False

        recent_blinks = [t for t in self.blink_times if now - t < 60]
        result['blink_rate'] = len(recent_blinks)

        # Micro-expressions
        result['micro_expressions'] = self._detect_micro_expressions(landmarks)

        return result

    def _process_opencv(self, frame: np.ndarray, result: Dict) -> Dict:
        """Process with OpenCV Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

        if len(faces) == 0:
            return result

        face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = face
        result['face_detected'] = True
        result['face_rect'] = (x, y, w, h)

        # Forehead ROI
        fh_y = max(0, y)
        fh_h = int(h * 0.3)
        fh_x = x + int(w * 0.2)
        fh_w = int(w * 0.6)

        if fh_w > 0 and fh_h > 0:
            forehead_roi = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
            if forehead_roi.size > 0:
                mean = np.mean(forehead_roi, axis=(0, 1))
                result['mean_rgb'] = (mean[2], mean[1], mean[0])
                result['forehead_signal'] = mean[1]

        return result

    def _get_roi_mean(self, frame: np.ndarray, landmarks, points: List[int], h: int, w: int):
        """Extract mean color from ROI defined by landmarks"""
        coords = []
        for idx in points:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                if 0 <= px < w and 0 <= py < h:
                    coords.append((px, py))

        if len(coords) < 3:
            return None

        coords = np.array(coords)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [coords], 255)

        pixels = frame[mask > 0]
        if len(pixels) > 50:
            return np.mean(pixels, axis=0)
        return None

    def _calc_ear(self, landmarks, side: str) -> float:
        """Calculate Eye Aspect Ratio"""
        if side == 'left':
            top = landmarks.landmark[self.LEFT_EYE_TOP]
            bottom = landmarks.landmark[self.LEFT_EYE_BOTTOM]
        else:
            top = landmarks.landmark[self.RIGHT_EYE_TOP]
            bottom = landmarks.landmark[self.RIGHT_EYE_BOTTOM]
        return abs(top.y - bottom.y)

    def _detect_micro_expressions(self, landmarks) -> List[str]:
        """Detect micro-expressions"""
        expressions = []

        # Brow furrow
        left_brow = landmarks.landmark[70]
        right_brow = landmarks.landmark[300]
        center = landmarks.landmark[9]
        if left_brow.y < center.y - 0.02 and right_brow.y < center.y - 0.02:
            expressions.append('brow_furrow')

        # Lip compression
        upper = landmarks.landmark[13]
        lower = landmarks.landmark[14]
        if abs(upper.y - lower.y) < 0.01:
            expressions.append('lip_compression')

        return expressions

    def close(self):
        if self.face_mesh:
            self.face_mesh.close()


# ==============================================================================
# STRESS & DECEPTION ANALYZER
# ==============================================================================

class StressAnalyzer:
    """Stress and deception analysis"""

    def __init__(self):
        self.baseline_hr = None
        self.baseline_hrv = None
        self.baseline_blink = None
        self.calibrated = False

        self.hr_history = deque(maxlen=300)
        self.stress_history = deque(maxlen=300)

    def calibrate(self, states: List[PhysiologicalState]) -> bool:
        """Calibrate baseline values"""
        valid = [s for s in states if s.heart_rate > 0]
        if len(valid) < 15:
            return False

        self.baseline_hr = np.mean([s.heart_rate for s in valid])
        self.baseline_hrv = np.mean([s.hrv_rmssd for s in valid]) or 30
        self.baseline_blink = np.mean([s.blink_rate for s in valid]) or 15
        self.calibrated = True
        return True

    def analyze(self, state: PhysiologicalState) -> Tuple[float, float]:
        """Analyze stress and deception probability"""
        if not self.calibrated or state.heart_rate <= 0:
            return 0.0, 0.0

        self.hr_history.append(state.heart_rate)

        # Stress indicators
        indicators = []

        # HR elevation
        hr_dev = (state.heart_rate - self.baseline_hr) / self.baseline_hr
        indicators.append(min(max(hr_dev * 2, 0), 1) * 0.3)

        # HRV decrease
        if self.baseline_hrv > 0 and state.hrv_rmssd > 0:
            hrv_ratio = state.hrv_rmssd / self.baseline_hrv
            indicators.append((1 - min(hrv_ratio, 1)) * 0.25)

        # LF/HF ratio
        if state.hrv_lf_hf_ratio > 0:
            indicators.append(min(state.hrv_lf_hf_ratio / 4, 1) * 0.2)

        # Blink anomaly
        if self.baseline_blink > 0:
            blink_dev = abs(state.blink_rate - self.baseline_blink) / self.baseline_blink
            indicators.append(min(blink_dev, 1) * 0.15)

        # Micro-expressions
        indicators.append(min(len(state.micro_expressions) / 3, 1) * 0.1)

        stress = sum(indicators)
        self.stress_history.append(stress)

        # Deception probability
        deception_indicators = [stress]

        if len(self.hr_history) > 30:
            recent_avg = np.mean(list(self.hr_history)[-30:])
            spike = max(0, state.heart_rate - recent_avg - 5) / 15
            deception_indicators.append(min(spike, 1))

        deception = float(np.mean(deception_indicators))

        return float(stress), deception


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

class RPPGApp:
    """Main application class"""

    def __init__(self):
        self.rppg = OptimizedRPPGProcessor()
        self.face = OptimizedFaceAnalyzer()
        self.stress = StressAnalyzer()

        self.cap = None
        self.running = False

        self.current_state = PhysiologicalState()
        self.state_history = deque(maxlen=1800)
        self.calibration_states = []
        self.calibrating = False

        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)

        self.lock = threading.Lock()

    def start_camera(self) -> bool:
        """Start camera capture"""
        # Try DirectShow first (Windows)
        self.cap = cv2.VideoCapture(config.CAMERA_ID, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(config.CAMERA_ID)

        if not self.cap.isOpened():
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency

        self.running = True
        return True

    def stop_camera(self):
        """Stop camera"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def process_frame(self) -> Tuple[Optional[np.ndarray], PhysiologicalState]:
        """Process single frame"""
        if not self.cap or not self.running:
            return None, self.current_state

        ret, frame = self.cap.read()
        if not ret:
            return None, self.current_state

        timestamp = time.time()

        # Face detection and ROI extraction
        face_result = self.face.process(frame)

        # Create state
        state = PhysiologicalState(timestamp=timestamp)
        state.face_detected = face_result['face_detected']
        state.blink_rate = face_result['blink_rate']
        state.micro_expressions = face_result['micro_expressions']

        if face_result['face_detected']:
            r, g, b = face_result['mean_rgb']

            if g > 0:
                # Add sample with multi-ROI data
                self.rppg.add_sample(
                    r, g, b, timestamp,
                    face_result.get('forehead_signal', g),
                    face_result.get('left_cheek_signal', g),
                    face_result.get('right_cheek_signal', g)
                )

                # Calculate HR
                hr, conf = self.rppg.calculate_heart_rate()
                state.heart_rate = hr
                state.hr_confidence = conf
                state.signal_quality = self.rppg.signal_quality

                # HRV metrics
                hrv = self.rppg.calculate_hrv_metrics()
                state.hrv_sdnn = hrv['sdnn']
                state.hrv_rmssd = hrv['rmssd']
                state.hrv_lf_hf_ratio = hrv['lf_hf']

                # Respiratory rate
                state.respiratory_rate = self.rppg.calculate_respiratory_rate()

                # Stress analysis
                if self.stress.calibrated:
                    stress, deception = self.stress.analyze(state)
                    state.stress_level = stress
                    state.deception_probability = deception

        with self.lock:
            self.current_state = state
            self.state_history.append(state)
            if self.calibrating:
                self.calibration_states.append(state)

        # Draw visualization
        vis_frame = self._draw_overlay(frame, face_result, state)

        return vis_frame, state

    def _draw_overlay(self, frame: np.ndarray, face_result: Dict,
                      state: PhysiologicalState) -> np.ndarray:
        """Draw visualization overlay"""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Draw ROI
        if face_result['landmarks'] and self.face.use_mediapipe:
            self._draw_roi_mediapipe(vis, face_result['landmarks'])
        elif face_result['face_rect']:
            self._draw_roi_opencv(vis, face_result['face_rect'])

        # Top panel background
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

        # Heart rate
        hr_color = (0, 255, 0) if 60 <= state.heart_rate <= 100 else (0, 255, 255)
        hr_text = f"HR: {state.heart_rate:.0f} BPM" if state.heart_rate > 0 else "HR: --"
        cv2.putText(vis, hr_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, hr_color, 2)

        # Signal quality
        sq_color = (0, 255, 0) if state.signal_quality > 0.5 else (0, 165, 255)
        cv2.putText(vis, f"Quality: {state.signal_quality*100:.0f}%", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, sq_color, 1)

        # HRV
        cv2.putText(vis, f"HRV: {state.hrv_rmssd:.1f}ms", (220, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.putText(vis, f"RR: {state.respiratory_rate:.0f}/min", (220, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Stress bar
        self._draw_bar(vis, 420, 20, "STRESS", state.stress_level)

        # Deception bar
        self._draw_bar(vis, 620, 20, "DECEPTION", state.deception_probability)

        # Blinks
        cv2.putText(vis, f"Blinks: {state.blink_rate:.0f}/min", (820, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Micro-expressions
        if state.micro_expressions:
            cv2.putText(vis, f"Micro: {', '.join(state.micro_expressions)}",
                       (820, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

        # Method indicator
        method = "MediaPipe" if self.face.use_mediapipe else "OpenCV"
        gpu_txt = " + CUDA" if (GPU_AVAILABLE or TORCH_GPU) else ""
        cv2.putText(vis, f"[{method}{gpu_txt}]", (w - 180, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

        # Status messages
        if not state.face_detected:
            cv2.putText(vis, "NO FACE DETECTED", (w//2 - 150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if self.calibrating:
            cv2.putText(vis, "CALIBRATING...", (w//2 - 100, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        elif not self.stress.calibrated:
            cv2.putText(vis, "Click 'Calibrate' to start", (w//2 - 140, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        return vis

    def _draw_bar(self, frame: np.ndarray, x: int, y: int, label: str, value: float):
        """Draw a progress bar"""
        cv2.putText(frame, f"{label}:", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        bar_w, bar_h = 150, 20
        cv2.rectangle(frame, (x, y + 20), (x + bar_w, y + 20 + bar_h), (100, 100, 100), 1)

        fill = int(value * bar_w)
        color = (0, 255, 0) if value < 0.3 else (0, 255, 255) if value < 0.6 else (0, 0, 255)
        cv2.rectangle(frame, (x, y + 20), (x + fill, y + 20 + bar_h), color, -1)

    def _draw_roi_mediapipe(self, frame: np.ndarray, landmarks):
        """Draw ROI for MediaPipe"""
        h, w = frame.shape[:2]
        coords = []
        for idx in self.face.FOREHEAD_POINTS:
            if idx < len(landmarks.landmark):
                lm = landmarks.landmark[idx]
                coords.append((int(lm.x * w), int(lm.y * h)))
        if len(coords) > 2:
            cv2.polylines(frame, [np.array(coords)], True, (0, 255, 0), 1)

    def _draw_roi_opencv(self, frame: np.ndarray, rect: Tuple):
        """Draw ROI for OpenCV"""
        x, y, w, h = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # Forehead
        fh_y, fh_h = y, int(h * 0.3)
        fh_x, fh_w = x + int(w * 0.2), int(w * 0.6)
        cv2.rectangle(frame, (fh_x, fh_y), (fh_x + fh_w, fh_y + fh_h), (0, 255, 255), 1)

    def start_calibration(self):
        """Start calibration"""
        self.calibration_states = []
        self.calibrating = True

    def finish_calibration(self) -> bool:
        """Finish calibration"""
        self.calibrating = False
        return self.stress.calibrate(self.calibration_states)

    def cleanup(self):
        """Cleanup resources"""
        self.stop_camera()
        self.face.close()


# ==============================================================================
# STREAMLIT UI
# ==============================================================================

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="rPPG Lie Detection System v3.0",
        page_icon="heartbeat",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for smooth UI
    st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stMetric { background-color: #1e1e1e; padding: 15px; border-radius: 10px; }
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 20px; border-radius: 15px; margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .stress-low { color: #00ff00; }
    .stress-medium { color: #ffff00; }
    .stress-high { color: #ff0000; }
    .stButton>button {
        width: 100%; border-radius: 10px; padding: 10px;
        font-weight: bold; transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: scale(1.02); }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'app' not in st.session_state:
        st.session_state.app = None
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'hr_history' not in st.session_state:
        st.session_state.hr_history = []
    if 'stress_history' not in st.session_state:
        st.session_state.stress_history = []

    # Sidebar
    with st.sidebar:
        st.title("rPPG Lie Detection")
        st.markdown("**v3.0 - GPU Optimized**")

        st.divider()

        # System info
        st.subheader("System Status")
        col1, col2 = st.columns(2)
        with col1:
            if GPU_AVAILABLE:
                st.success("CuPy CUDA")
            elif TORCH_GPU:
                st.success("PyTorch CUDA")
            else:
                st.warning("CPU Mode")
        with col2:
            if MEDIAPIPE_AVAILABLE:
                st.success("MediaPipe")
            else:
                st.warning("OpenCV")

        st.divider()

        # Controls
        st.subheader("Controls")

        col1, col2 = st.columns(2)
        with col1:
            start_btn = st.button("Start", type="primary", use_container_width=True)
        with col2:
            stop_btn = st.button("Stop", type="secondary", use_container_width=True)

        calibrate_btn = st.button("Calibrate (5s)", use_container_width=True)

        st.divider()

        # Settings
        st.subheader("Settings")
        config.CAMERA_ID = st.selectbox("Camera", [0, 1, 2], index=0)
        config.TARGET_FPS = st.slider("Target FPS", 15, 60, 30)

    # Main content
    st.title("Real-Time Physiological Analysis")

    # Create columns for layout
    video_col, metrics_col = st.columns([2, 1])

    with video_col:
        video_placeholder = st.empty()

    with metrics_col:
        # Metrics placeholders
        hr_metric = st.empty()
        hrv_metric = st.empty()
        stress_metric = st.empty()
        deception_metric = st.empty()

    # Charts row
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        hr_chart = st.empty()
    with chart_col2:
        stress_chart = st.empty()

    # Handle button actions
    if start_btn and not st.session_state.running:
        st.session_state.app = RPPGApp()
        if st.session_state.app.start_camera():
            st.session_state.running = True
            st.session_state.hr_history = []
            st.session_state.stress_history = []
            st.rerun()
        else:
            st.error("Failed to open camera!")
            st.session_state.app = None

    if stop_btn and st.session_state.running:
        if st.session_state.app:
            st.session_state.app.cleanup()
            st.session_state.app = None
        st.session_state.running = False
        st.rerun()

    if calibrate_btn and st.session_state.running and st.session_state.app:
        st.session_state.app.start_calibration()
        st.toast("Calibrating... Please stay still for 5 seconds")

        # Run calibration for 5 seconds
        start_time = time.time()
        while time.time() - start_time < 5:
            if st.session_state.app:
                st.session_state.app.process_frame()
            time.sleep(0.033)

        if st.session_state.app:
            if st.session_state.app.finish_calibration():
                st.toast("Calibration successful!", icon="check")
            else:
                st.toast("Calibration failed - not enough data", icon="cross")

    # Main processing loop
    if st.session_state.running and st.session_state.app:
        app = st.session_state.app

        # Process frame
        frame, state = app.process_frame()

        if frame is not None:
            # Display video
            video_placeholder.image(frame, channels="BGR", use_container_width=True)

            # Update metrics
            hr_color = "normal" if 60 <= state.heart_rate <= 100 else "off"
            hr_metric.metric("Heart Rate", f"{state.heart_rate:.0f} BPM",
                           delta=f"Confidence: {state.hr_confidence*100:.0f}%")

            hrv_metric.metric("HRV (RMSSD)", f"{state.hrv_rmssd:.1f} ms",
                            delta=f"SDNN: {state.hrv_sdnn:.1f} ms")

            stress_pct = state.stress_level * 100
            stress_delta = "Low" if stress_pct < 30 else "Medium" if stress_pct < 60 else "High"
            stress_metric.metric("Stress Level", f"{stress_pct:.0f}%", delta=stress_delta)

            dec_pct = state.deception_probability * 100
            deception_metric.metric("Deception Index", f"{dec_pct:.0f}%")

            # Update history for charts
            if state.heart_rate > 0:
                st.session_state.hr_history.append(state.heart_rate)
                st.session_state.stress_history.append(state.stress_level * 100)

                # Keep last 100 values
                st.session_state.hr_history = st.session_state.hr_history[-100:]
                st.session_state.stress_history = st.session_state.stress_history[-100:]

            # Update charts
            if len(st.session_state.hr_history) > 2:
                hr_chart.line_chart(st.session_state.hr_history, height=200)
                stress_chart.line_chart(st.session_state.stress_history, height=200)

        # Rerun for continuous update
        time.sleep(0.033)
        st.rerun()
    else:
        # Show placeholder when not running
        video_placeholder.info("Click 'Start' to begin camera capture")
        hr_metric.metric("Heart Rate", "-- BPM")
        hrv_metric.metric("HRV (RMSSD)", "-- ms")
        stress_metric.metric("Stress Level", "--%")
        deception_metric.metric("Deception Index", "--%")


if __name__ == "__main__":
    main()

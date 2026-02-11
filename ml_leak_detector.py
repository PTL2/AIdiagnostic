"""
ML-Powered Leak Detector

Uses trained machine learning models to detect valve leaks.
Can be used standalone or as part of a hybrid system with physics-based detection.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class MLDetectionResult:
    """Result from ML-based leak detection"""
    is_leak: bool
    leak_probability: float  # 0.0 to 100.0
    confidence: float  # 0.0 to 1.0
    model_agreement: float  # 0.0 to 1.0 (how much XGB and RF agree)
    xgb_probability: float
    rf_probability: float
    feature_values: Dict[str, float]


class MLLeakDetector:
    """
    Machine Learning-based leak detector using trained XGBoost and Random Forest models.
    """

    def __init__(self, model_path: str = None):
        """
        Initialize ML detector.

        Args:
            model_path: Path to trained model file (.pkl)
                       If None, looks for 'leak_detection_model_latest.pkl'
        """
        if model_path is None:
            model_path = Path(__file__).parent / 'leak_detection_model_latest.pkl'
        else:
            model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please train a model first using: python train_ml_model.py"
            )

        # Load model
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.xgb_model = model_data['xgb_model']
        self.rf_model = model_data['rf_model']
        self.feature_columns = model_data['feature_columns']
        self.training_stats = model_data.get('training_stats', {})

        print(f"ML Leak Detector loaded from: {model_path}")
        if 'timestamp' in model_data:
            print(f"Model trained: {model_data['timestamp']}")

    def extract_features_from_waveform(self, amplitudes: np.ndarray) -> Dict[str, float]:
        """
        Extract ML features from waveform.

        Args:
            amplitudes: Waveform amplitude array

        Returns:
            Dict of features matching self.feature_columns
        """
        # Basic statistics
        mean_amp = np.mean(amplitudes)
        median_amp = np.median(amplitudes)
        std_amp = np.std(amplitudes)
        max_amp = np.max(amplitudes)
        min_amp = np.min(amplitudes)

        # Percentiles
        p25 = np.percentile(amplitudes, 25)
        p75 = np.percentile(amplitudes, 75)
        p90 = np.percentile(amplitudes, 90)
        p95 = np.percentile(amplitudes, 95)
        p99 = np.percentile(amplitudes, 99)

        # Threshold-based ratios
        above_1g = np.sum(np.abs(amplitudes) > 1.0) / len(amplitudes)
        above_2g = np.sum(np.abs(amplitudes) > 2.0) / len(amplitudes)
        above_3g = np.sum(np.abs(amplitudes) > 3.0) / len(amplitudes)
        above_5g = np.sum(np.abs(amplitudes) > 5.0) / len(amplitudes)

        # Peak counts
        peaks_above_5g = np.sum(np.abs(amplitudes) > 5.0)
        peaks_above_10g = np.sum(np.abs(amplitudes) > 10.0)

        # Range and spread
        amplitude_range = max_amp - min_amp
        iqr = p75 - p25

        # RMS
        rms = np.sqrt(np.mean(amplitudes ** 2))

        # Crest factor
        crest_factor = max_amp / rms if rms > 0 else 0

        return {
            'mean_amplitude': mean_amp,
            'median_amplitude': median_amp,
            'std_amplitude': std_amp,
            'max_amplitude': max_amp,
            'min_amplitude': min_amp,
            'percentile_25': p25,
            'percentile_75': p75,
            'percentile_90': p90,
            'percentile_95': p95,
            'percentile_99': p99,
            'above_1g_ratio': above_1g,
            'above_2g_ratio': above_2g,
            'above_3g_ratio': above_3g,
            'above_5g_ratio': above_5g,
            'peaks_above_5g': float(peaks_above_5g),
            'peaks_above_10g': float(peaks_above_10g),
            'amplitude_range': amplitude_range,
            'iqr': iqr,
            'rms': rms,
            'crest_factor': crest_factor
        }

    def detect_leak(self, amplitudes: np.ndarray) -> MLDetectionResult:
        """
        Detect leak using ML models.

        Args:
            amplitudes: Waveform amplitude values (numpy array)

        Returns:
            MLDetectionResult with predictions
        """
        # Extract features
        features = self.extract_features_from_waveform(amplitudes)

        # Create feature vector in correct order
        X = np.array([[features[col] for col in self.feature_columns]])

        # Get predictions from both models
        xgb_proba = self.xgb_model.predict_proba(X)[0, 1]  # Probability of leak
        rf_proba = self.rf_model.predict_proba(X)[0, 1]

        # Ensemble: average probabilities
        ensemble_proba = (xgb_proba + rf_proba) / 2

        # Model agreement (how close the two models are)
        model_agreement = 1.0 - abs(xgb_proba - rf_proba)

        # Final prediction
        is_leak = ensemble_proba >= 0.5
        leak_probability = ensemble_proba * 100

        # Confidence based on:
        # 1. Distance from decision boundary (0.5)
        # 2. Model agreement
        boundary_confidence = abs(ensemble_proba - 0.5) * 2  # 0 at boundary, 1 at extremes
        confidence = (boundary_confidence + model_agreement) / 2

        return MLDetectionResult(
            is_leak=is_leak,
            leak_probability=leak_probability,
            confidence=confidence,
            model_agreement=model_agreement,
            xgb_probability=xgb_proba * 100,
            rf_probability=rf_proba * 100,
            feature_values=features
        )


class HybridLeakDetector:
    """
    Hybrid detector combining physics-based and ML approaches.

    Uses both methods and provides:
    - Individual results from each approach
    - Combined/ensemble result
    - Confidence scoring based on agreement
    """

    def __init__(self, ml_model_path: str = None):
        """
        Initialize hybrid detector.

        Args:
            ml_model_path: Path to ML model file (optional)
        """
        # Import physics-based detector
        from leak_detector import PhysicsBasedLeakDetector

        self.physics_detector = PhysicsBasedLeakDetector()
        self.ml_detector = None

        # Try to load ML model
        try:
            self.ml_detector = MLLeakDetector(ml_model_path)
            self.has_ml = True
        except FileNotFoundError as e:
            print(f"ML model not found - using physics-based only")
            print(f"To enable ML: {str(e)}")
            self.has_ml = False

    def detect_leak(self, amplitudes: np.ndarray) -> Dict:
        """
        Detect leak using both physics and ML approaches.

        Args:
            amplitudes: Waveform amplitude values

        Returns:
            Dict with results from both methods and ensemble
        """
        # Get physics-based result
        physics_result = self.physics_detector.detect_leak(amplitudes)

        result = {
            'physics': {
                'is_leak': physics_result.is_leak,
                'leak_probability': physics_result.leak_probability,
                'confidence': physics_result.confidence,
                'explanation': physics_result.explanation
            }
        }

        # Get ML result if available
        if self.has_ml:
            ml_result = self.ml_detector.detect_leak(amplitudes)

            result['ml'] = {
                'is_leak': ml_result.is_leak,
                'leak_probability': ml_result.leak_probability,
                'confidence': ml_result.confidence,
                'model_agreement': ml_result.model_agreement,
                'xgb_probability': ml_result.xgb_probability,
                'rf_probability': ml_result.rf_probability
            }

            # Ensemble result (weighted average)
            # Weight ML higher if it has high model agreement
            ml_weight = 0.5 + (ml_result.model_agreement * 0.2)  # 0.5 to 0.7
            physics_weight = 1.0 - ml_weight

            ensemble_prob = (
                physics_result.leak_probability * physics_weight +
                ml_result.leak_probability * ml_weight
            )

            # Agreement between physics and ML
            method_agreement = 1.0 - abs(
                physics_result.leak_probability - ml_result.leak_probability
            ) / 100.0

            result['ensemble'] = {
                'is_leak': ensemble_prob >= 50,
                'leak_probability': ensemble_prob,
                'confidence': (physics_result.confidence + ml_result.confidence) / 2,
                'method_agreement': method_agreement,
                'recommendation': self._get_recommendation(
                    physics_result, ml_result, method_agreement
                )
            }
        else:
            # No ML - just use physics result
            result['ensemble'] = result['physics'].copy()
            result['ensemble']['recommendation'] = 'Physics-based detection only (ML not trained yet)'

        return result

    def _get_recommendation(self, physics_result, ml_result, agreement):
        """Generate recommendation based on results."""
        if agreement > 0.8:
            if physics_result.is_leak and ml_result.is_leak:
                return "HIGH CONFIDENCE LEAK - Both methods agree strongly"
            elif not physics_result.is_leak and not ml_result.is_leak:
                return "HIGH CONFIDENCE NORMAL - Both methods agree strongly"

        if agreement > 0.5:
            return "MODERATE CONFIDENCE - Methods show reasonable agreement"

        # Low agreement
        if physics_result.is_leak or ml_result.is_leak:
            return "UNCERTAIN - Methods disagree, recommend manual inspection"
        else:
            return "LIKELY NORMAL - Despite some disagreement, both lean toward normal"

"""
Physics-Based Valve Leak Detection for Ultrasonic Sensors

Based on Acoustic Emission (AE) physics and actual XML waveform analysis:
- ULTRASONIC sensors (36-44 KHz narrow band) measure acoustic emissions
- NORMAL valve: Brief acoustic spike during valve event only = LOW mean amplitude (~1-2G)
- LEAKING valve: Sustained "smear" pattern from gas escaping = HIGH mean amplitude (~4-5G)

Key insight: For ultrasonic AE sensors, HIGH sustained amplitude indicates leak!
This is because gas escaping through gaps creates continuous acoustic noise (smear pattern).

Actual XML file analysis confirms:
- C402 Cyl 3 CD (known LEAK): mean 4.59G, 99% above 1G, 92% above 2G
- C402 Cyl 2 CD (normal): mean 1.27G, 59% above 1G, 17% above 2G
- LEAK has 3.6x higher mean amplitude with sustained elevation
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LeakDetectionResult:
    """Result of physics-based leak detection"""
    is_leak: bool
    confidence: float  # 0.0 to 1.0
    leak_probability: float  # 0.0 to 100.0
    criteria_met: Dict[str, bool]
    feature_values: Dict[str, float]
    explanation: str


class PhysicsBasedLeakDetector:
    """
    Physics-based leak detection using ultrasonic sensor characteristics.

    Based on actual XML file analysis:
    - Normal valves: mean amplitude ~1-2G (brief spikes only)
    - Leak valves: mean amplitude ~4-5G (sustained smear pattern)
    - Threshold: ~2.5-3G separates normal from leak
    """

    def __init__(self):
        # Thresholds based on actual XML file analysis
        # Normal: ~1-2G mean (Cyl 2 CD = 1.27G)
        # Leak: ~4-5G mean (Cyl 3 CD = 4.59G)
        self.amplitude_threshold_low = 2.0   # Below this = likely normal
        self.amplitude_threshold_mid = 3.0   # Above this = possible leak
        self.amplitude_threshold_high = 4.0  # Above this = likely leak

        # Pattern features for leak detection
        self.elevated_ratio_threshold = 0.8  # >80% above 1G = likely leak
        self.medium_elevation_threshold = 0.5  # >50% above 2G = likely leak
        self.high_elevation_threshold = 0.2   # >20% above 5G = severe leak

        # Confidence scaling
        self.leak_severe_threshold = 5.0   # Very high = severe leak
        self.leak_moderate_threshold = 3.5  # Moderately high = moderate leak

    def extract_features(self, amplitudes: np.ndarray) -> Dict[str, float]:
        """Extract physics-relevant features from ultrasonic amplitude data."""

        features = {
            'mean_amplitude': float(np.mean(amplitudes)),
            'max_amplitude': float(np.max(amplitudes)),
            'min_amplitude': float(np.min(amplitudes)),
            'std_amplitude': float(np.std(amplitudes)),
            'median_amplitude': float(np.median(amplitudes)),
            'amplitude_range': float(np.max(amplitudes) - np.min(amplitudes)),

            # Percentile features (robust to outliers)
            'p25_amplitude': float(np.percentile(amplitudes, 25)),
            'p75_amplitude': float(np.percentile(amplitudes, 75)),
            'iqr': float(np.percentile(amplitudes, 75) - np.percentile(amplitudes, 25)),

            # Pattern features
            'low_amplitude_ratio': float((amplitudes < 10.0).sum() / len(amplitudes)),
            'high_amplitude_ratio': float((amplitudes > 20.0).sum() / len(amplitudes)),
            'coefficient_of_variation': float(np.std(amplitudes) / np.mean(amplitudes)) if np.mean(amplitudes) > 0 else 0,
        }

        return features

    def detect_leak(self, amplitudes: np.ndarray) -> LeakDetectionResult:
        """
        Detect valve leak using physics-based thresholds.

        Logic (CORRECTED based on actual XML analysis):
        - HIGH mean amplitude + HIGH sustained elevation = LEAK (gas escaping = smear pattern)
        - LOW mean amplitude + brief spikes only = NORMAL (clean valve closure)

        Args:
            amplitudes: Array of ultrasonic amplitude values (in G units)

        Returns:
            LeakDetectionResult with detection outcome and explanation
        """

        features = self.extract_features(amplitudes)

        # Core detection criteria (INVERTED: HIGH = LEAK)
        criteria = {}

        # Primary criterion: Mean amplitude (HIGH = LEAK)
        mean_amp = features['mean_amplitude']
        criteria['high_mean_amplitude'] = mean_amp > self.amplitude_threshold_mid

        # Secondary criterion: High amplitude ratio (sustained elevation)
        # Leak valves have most readings above 1G
        high_ratio = features['high_amplitude_ratio']
        criteria['high_sustained_elevation'] = high_ratio > 0.1  # >10% above 20G not realistic, use different metric

        # Calculate actual sustained elevation ratios
        above_1g_ratio = (amplitudes > 1.0).sum() / len(amplitudes)
        above_2g_ratio = (amplitudes > 2.0).sum() / len(amplitudes)
        above_5g_ratio = (amplitudes > 5.0).sum() / len(amplitudes)

        features['above_1g_ratio'] = float(above_1g_ratio)
        features['above_2g_ratio'] = float(above_2g_ratio)
        features['above_5g_ratio'] = float(above_5g_ratio)

        criteria['sustained_above_1g'] = above_1g_ratio > self.elevated_ratio_threshold
        criteria['sustained_above_2g'] = above_2g_ratio > self.medium_elevation_threshold
        criteria['has_high_activity'] = above_5g_ratio > self.high_elevation_threshold

        # Calculate leak probability based on weighted criteria
        leak_score = 0.0
        max_score = 0.0

        # Mean amplitude is most important (weight: 0.35)
        max_score += 0.35
        if mean_amp > self.leak_severe_threshold:
            leak_score += 0.35  # Severe leak indicator (>5G)
        elif mean_amp > self.leak_moderate_threshold:
            leak_score += 0.28  # Moderate leak indicator (>3.5G)
        elif mean_amp > self.amplitude_threshold_high:
            leak_score += 0.21  # Likely leak (>4G)
        elif mean_amp > self.amplitude_threshold_mid:
            leak_score += 0.14  # Possible leak (>3G)
        elif mean_amp > self.amplitude_threshold_low:
            leak_score += 0.07  # Marginal (>2G)

        # Sustained elevation above 1G (weight: 0.25)
        max_score += 0.25
        if above_1g_ratio > 0.95:
            leak_score += 0.25  # Nearly all readings above 1G
        elif above_1g_ratio > 0.85:
            leak_score += 0.20
        elif above_1g_ratio > 0.70:
            leak_score += 0.15
        elif above_1g_ratio > 0.50:
            leak_score += 0.10

        # Sustained elevation above 2G (weight: 0.25)
        max_score += 0.25
        if above_2g_ratio > 0.80:
            leak_score += 0.25  # Most readings above 2G = severe
        elif above_2g_ratio > 0.50:
            leak_score += 0.20  # Majority above 2G = significant leak
        elif above_2g_ratio > 0.30:
            leak_score += 0.15
        elif above_2g_ratio > 0.15:
            leak_score += 0.10

        # High activity above 5G (weight: 0.15)
        max_score += 0.15
        if above_5g_ratio > 0.30:
            leak_score += 0.15  # Significant high activity
        elif above_5g_ratio > 0.15:
            leak_score += 0.10
        elif above_5g_ratio > 0.05:
            leak_score += 0.05

        # Calculate final probability
        leak_probability = (leak_score / max_score) * 100.0 if max_score > 0 else 0.0

        # Determine if leak detected (threshold: 50%)
        is_leak = leak_probability >= 50.0

        # Confidence based on how far from threshold
        if is_leak:
            # Higher probability = higher confidence
            confidence = min(1.0, leak_probability / 100.0 + 0.2)
        else:
            # Lower probability = higher confidence in normal
            confidence = min(1.0, (100.0 - leak_probability) / 100.0 + 0.2)

        # Generate explanation
        explanation = self._generate_explanation(features, criteria, leak_probability, is_leak)

        return LeakDetectionResult(
            is_leak=is_leak,
            confidence=confidence,
            leak_probability=leak_probability,
            criteria_met=criteria,
            feature_values=features,
            explanation=explanation
        )

    def _generate_explanation(self, features: Dict[str, float],
                               criteria: Dict[str, bool],
                               leak_probability: float,
                               is_leak: bool) -> str:
        """Generate human-readable explanation of detection result."""

        mean_amp = features['mean_amplitude']
        median_amp = features['median_amplitude']
        max_amp = features['max_amplitude']
        above_1g = features.get('above_1g_ratio', 0) * 100
        above_2g = features.get('above_2g_ratio', 0) * 100

        if is_leak:
            if leak_probability >= 80:
                severity = "SEVERE LEAK"
                reason = f"Mean amplitude ({mean_amp:.1f}G) significantly exceeds normal range (<2G)"
            elif leak_probability >= 60:
                severity = "MODERATE LEAK"
                reason = f"Mean amplitude ({mean_amp:.1f}G) shows sustained elevation"
            else:
                severity = "POSSIBLE LEAK"
                reason = f"Mean amplitude ({mean_amp:.1f}G) is borderline high"

            explanation = f"{severity} DETECTED\n"
            explanation += f"Reason: {reason}\n\n"
            explanation += "Physics Analysis:\n"
            explanation += f"- Mean amplitude: {mean_amp:.2f}G (normal: <2G, leak: >3G)\n"
            explanation += f"- Median amplitude: {median_amp:.2f}G\n"
            explanation += f"- Max amplitude: {max_amp:.2f}G\n"
            explanation += f"- Above 1G: {above_1g:.1f}% (normal: <70%, leak: >85%)\n"
            explanation += f"- Above 2G: {above_2g:.1f}% (normal: <20%, leak: >50%)\n"
            explanation += f"\nInterpretation: HIGH sustained ultrasonic amplitude indicates gas "
            explanation += "escaping through valve seat, creating continuous acoustic noise (smear pattern)."
        else:
            explanation = "NORMAL VALVE OPERATION\n"
            explanation += f"Reason: Mean amplitude ({mean_amp:.1f}G) is within normal range\n\n"
            explanation += "Physics Analysis:\n"
            explanation += f"- Mean amplitude: {mean_amp:.2f}G (normal: <2G)\n"
            explanation += f"- Median amplitude: {median_amp:.2f}G\n"
            explanation += f"- Max amplitude: {max_amp:.2f}G\n"
            explanation += f"- Above 1G: {above_1g:.1f}% (normal: <70%)\n"
            explanation += f"- Above 2G: {above_2g:.1f}% (normal: <20%)\n"
            explanation += f"\nInterpretation: LOW baseline with brief spikes indicates proper valve "
            explanation += "closure with clean mechanical impacts (normal operation)."

        return explanation


def detect_leak_simple(amplitudes: np.ndarray) -> Tuple[bool, float, str]:
    """
    Simple function to detect leak from amplitude array.

    Args:
        amplitudes: Array of ultrasonic amplitude values

    Returns:
        Tuple of (is_leak, leak_probability, explanation)
    """
    detector = PhysicsBasedLeakDetector()
    result = detector.detect_leak(amplitudes)
    return result.is_leak, result.leak_probability, result.explanation


# Quick test if run directly
if __name__ == "__main__":
    # Test with known leak values (C402 Cyl 3 CD characteristics)
    print("Testing Physics-Based Leak Detector")
    print("=" * 50)

    # Simulate leak valve (low amplitude, like C402 Cyl 3 CD)
    np.random.seed(42)
    leak_valve = np.random.normal(loc=4.5, scale=2.0, size=355)
    leak_valve = np.clip(leak_valve, 0.2, 12.0)

    # Simulate normal valve (high amplitude)
    normal_valve = np.random.normal(loc=23.0, scale=3.0, size=355)
    normal_valve = np.clip(normal_valve, 18.0, 30.0)

    detector = PhysicsBasedLeakDetector()

    print("\n1. Testing LEAK valve (simulated C402 Cyl 3 CD):")
    result = detector.detect_leak(leak_valve)
    print(f"   Is Leak: {result.is_leak}")
    print(f"   Probability: {result.leak_probability:.1f}%")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Mean amplitude: {result.feature_values['mean_amplitude']:.2f}G")

    print("\n2. Testing NORMAL valve:")
    result = detector.detect_leak(normal_valve)
    print(f"   Is Leak: {result.is_leak}")
    print(f"   Probability: {result.leak_probability:.1f}%")
    print(f"   Confidence: {result.confidence:.1%}")
    print(f"   Mean amplitude: {result.feature_values['mean_amplitude']:.2f}G")

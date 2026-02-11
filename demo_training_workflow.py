"""
Demo: Complete AI Training Workflow

This script demonstrates the complete workflow from WRPM files to trained ML model.
Uses the sample WRPM files for demonstration purposes.
"""

import os
import pandas as pd
import subprocess
from pathlib import Path


def run_demo():
    """Run complete training workflow demo."""
    print("="*70)
    print("AI TRAINING WORKFLOW DEMONSTRATION")
    print("="*70)
    print("\nThis demo shows the complete pipeline:")
    print("  1. Extract features from WRPM files")
    print("  2. Combine training data")
    print("  3. Train ML models")
    print("  4. Test predictions")
    print("\n" + "="*70)

    # Step 1: Extract features from sample WRPM files
    print("\n[STEP 1] Extracting features from sample WRPM files...")
    print("-"*70)

    wrpm_dir = "../../assets/wrpm-samples"

    if not os.path.exists(wrpm_dir):
        print(f"Error: WRPM samples directory not found: {wrpm_dir}")
        return

    # Extract features
    cmd = [
        "python", "extract_training_data_from_wrpm.py",
        wrpm_dir,
        "--output", "demo_training_data.csv",
        "--label", "unknown"  # We don't know if these are leaks or normal
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    if not os.path.exists("demo_training_data.csv"):
        print("Error: Failed to extract features")
        return

    # Step 2: Review the extracted data
    print("\n[STEP 2] Reviewing extracted features...")
    print("-"*70)

    df = pd.read_csv("demo_training_data.csv")
    print(f"Total samples: {len(df)}")
    print(f"Features extracted: {len([c for c in df.columns if c not in ['file_name', 'machine_id', 'date', 'curve_name', 'data_points', 'detected_leak', 'leak_probability', 'confidence', 'manual_label']])}")

    print("\nFeature statistics:")
    print(f"  Mean amplitude: {df['mean_amplitude'].mean():.2f}G (±{df['mean_amplitude'].std():.2f})")
    print(f"  Max amplitude: {df['max_amplitude'].mean():.2f}G (±{df['max_amplitude'].std():.2f})")
    print(f"  Above 2G ratio: {df['above_2g_ratio'].mean():.2%}")

    # Step 3: Simulate labeled data for training
    print("\n[STEP 3] Simulating labeled training data...")
    print("-"*70)
    print("NOTE: In real use, you would label files based on field reports.")
    print("For demo, we'll create a synthetic dataset with known labels.\n")

    # Create synthetic training data with known labels
    # (In reality, you'd have separate leak and normal files)
    np = __import__('numpy')

    # Generate synthetic features for demonstration
    n_samples = 100
    n_leak = 50
    n_normal = 50

    # Leak samples (higher mean amplitude)
    leak_samples = {
        'mean_amplitude': np.random.normal(4.5, 0.8, n_leak),
        'median_amplitude': np.random.normal(4.3, 0.7, n_leak),
        'std_amplitude': np.random.normal(1.2, 0.3, n_leak),
        'max_amplitude': np.random.normal(8.0, 1.5, n_leak),
        'min_amplitude': np.random.normal(0.5, 0.2, n_leak),
        'percentile_25': np.random.normal(2.8, 0.6, n_leak),
        'percentile_75': np.random.normal(5.5, 0.9, n_leak),
        'percentile_90': np.random.normal(6.8, 1.2, n_leak),
        'percentile_95': np.random.normal(7.2, 1.3, n_leak),
        'percentile_99': np.random.normal(7.8, 1.4, n_leak),
        'above_1g_ratio': np.random.uniform(0.9, 1.0, n_leak),
        'above_2g_ratio': np.random.uniform(0.85, 0.98, n_leak),
        'above_3g_ratio': np.random.uniform(0.7, 0.9, n_leak),
        'above_5g_ratio': np.random.uniform(0.3, 0.6, n_leak),
        'peaks_above_5g': np.random.randint(50, 200, n_leak),
        'peaks_above_10g': np.random.randint(0, 20, n_leak),
        'amplitude_range': np.random.normal(8.0, 1.5, n_leak),
        'iqr': np.random.normal(3.0, 0.8, n_leak),
        'rms': np.random.normal(4.8, 0.9, n_leak),
        'crest_factor': np.random.normal(1.8, 0.3, n_leak),
        'manual_label': ['leak'] * n_leak
    }

    # Normal samples (lower mean amplitude)
    normal_samples = {
        'mean_amplitude': np.random.normal(1.5, 0.5, n_normal),
        'median_amplitude': np.random.normal(1.3, 0.4, n_normal),
        'std_amplitude': np.random.normal(0.6, 0.2, n_normal),
        'max_amplitude': np.random.normal(4.0, 1.0, n_normal),
        'min_amplitude': np.random.normal(0.2, 0.1, n_normal),
        'percentile_25': np.random.normal(0.9, 0.3, n_normal),
        'percentile_75': np.random.normal(2.0, 0.5, n_normal),
        'percentile_90': np.random.normal(2.8, 0.7, n_normal),
        'percentile_95': np.random.normal(3.2, 0.8, n_normal),
        'percentile_99': np.random.normal(3.8, 0.9, n_normal),
        'above_1g_ratio': np.random.uniform(0.3, 0.7, n_normal),
        'above_2g_ratio': np.random.uniform(0.1, 0.4, n_normal),
        'above_3g_ratio': np.random.uniform(0.0, 0.2, n_normal),
        'above_5g_ratio': np.random.uniform(0.0, 0.05, n_normal),
        'peaks_above_5g': np.random.randint(0, 30, n_normal),
        'peaks_above_10g': np.random.randint(0, 5, n_normal),
        'amplitude_range': np.random.normal(4.0, 1.0, n_normal),
        'iqr': np.random.normal(1.2, 0.4, n_normal),
        'rms': np.random.normal(1.8, 0.5, n_normal),
        'crest_factor': np.random.normal(2.2, 0.4, n_normal),
        'manual_label': ['normal'] * n_normal
    }

    # Combine
    leak_df = pd.DataFrame(leak_samples)
    normal_df = pd.DataFrame(normal_samples)
    training_df = pd.concat([leak_df, normal_df], ignore_index=True)
    training_df = training_df.sample(frac=1, random_state=42).reset_index(drop=True)

    training_df.to_csv("demo_synthetic_training.csv", index=False)
    print(f"Created synthetic training dataset: {len(training_df)} samples")
    print(f"  Leak: {(training_df['manual_label'] == 'leak').sum()}")
    print(f"  Normal: {(training_df['manual_label'] == 'normal').sum()}")

    # Step 4: Train ML model
    print("\n[STEP 4] Training machine learning models...")
    print("-"*70)

    cmd = [
        "python", "train_ml_model.py",
        "demo_synthetic_training.csv",
        "--output-dir", "."
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    # Step 5: Test the trained model
    if os.path.exists("leak_detection_model_latest.pkl"):
        print("\n[STEP 5] Testing trained model...")
        print("-"*70)

        from ml_leak_detector import MLLeakDetector
        import numpy as np

        detector = MLLeakDetector()

        # Test with synthetic leak waveform
        leak_waveform = np.random.normal(4.5, 1.2, 355)
        result_leak = detector.detect_leak(leak_waveform)

        print("\nTest 1: Synthetic LEAK waveform (mean ~4.5G)")
        print(f"  Leak Probability: {result_leak.leak_probability:.1f}%")
        print(f"  Confidence: {result_leak.confidence:.2f}")
        print(f"  XGBoost: {result_leak.xgb_probability:.1f}%")
        print(f"  Random Forest: {result_leak.rf_probability:.1f}%")
        print(f"  Model Agreement: {result_leak.model_agreement:.2f}")

        # Test with synthetic normal waveform
        normal_waveform = np.random.normal(1.5, 0.6, 355)
        result_normal = detector.detect_leak(normal_waveform)

        print("\nTest 2: Synthetic NORMAL waveform (mean ~1.5G)")
        print(f"  Leak Probability: {result_normal.leak_probability:.1f}%")
        print(f"  Confidence: {result_normal.confidence:.2f}")
        print(f"  XGBoost: {result_normal.xgb_probability:.1f}%")
        print(f"  Random Forest: {result_normal.rf_probability:.1f}%")
        print(f"  Model Agreement: {result_normal.model_agreement:.2f}")

    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print("\nFiles created:")
    print("  - demo_training_data.csv (features from sample WRPM files)")
    print("  - demo_synthetic_training.csv (synthetic labeled data)")
    print("  - leak_detection_model_latest.pkl (trained ML model)")
    print("  - training_report_*.txt (performance metrics)")
    print("\nNext steps:")
    print("  1. Replace synthetic data with real labeled WRPM files")
    print("  2. Extract features from your actual leak/normal files")
    print("  3. Retrain model with real data")
    print("  4. Integrate with dashboard using HybridLeakDetector")
    print("\nSee AI_TRAINING_COMPLETE_GUIDE.md for detailed instructions.")


if __name__ == "__main__":
    run_demo()

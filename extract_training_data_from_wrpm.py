"""
Training Data Extraction from WRPM Files

This script processes WRPM files and extracts features for AI model training.
Outputs a CSV file with features that can be used for supervised machine learning.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from wrpm_parser_ae import WrpmParserAE
from leak_detector import PhysicsBasedLeakDetector


def extract_features_from_waveform(amplitudes: np.ndarray) -> dict:
    """
    Extract statistical and physics-based features from waveform.

    Args:
        amplitudes: Waveform amplitude array

    Returns:
        Dict of features
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

    # Threshold-based ratios (key for leak detection)
    above_1g = np.sum(np.abs(amplitudes) > 1.0) / len(amplitudes)
    above_2g = np.sum(np.abs(amplitudes) > 2.0) / len(amplitudes)
    above_3g = np.sum(np.abs(amplitudes) > 3.0) / len(amplitudes)
    above_5g = np.sum(np.abs(amplitudes) > 5.0) / len(amplitudes)

    # Peak characteristics
    peaks_above_5g = np.sum(np.abs(amplitudes) > 5.0)
    peaks_above_10g = np.sum(np.abs(amplitudes) > 10.0)

    # Range and spread
    amplitude_range = max_amp - min_amp
    iqr = p75 - p25

    # RMS (Root Mean Square)
    rms = np.sqrt(np.mean(amplitudes ** 2))

    # Crest factor (peak to RMS ratio)
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
        'peaks_above_5g': peaks_above_5g,
        'peaks_above_10g': peaks_above_10g,
        'amplitude_range': amplitude_range,
        'iqr': iqr,
        'rms': rms,
        'crest_factor': crest_factor
    }


def process_wrpm_file(wrpm_path: Path) -> list:
    """
    Process a single WRPM file and extract features from all curves.

    Args:
        wrpm_path: Path to WRPM file

    Returns:
        List of dicts, one per curve
    """
    print(f"Processing: {wrpm_path.name}")

    try:
        # Parse WRPM file
        parser = WrpmParserAE(wrpm_path)
        df_curves = parser.parse_to_dataframe()
        info = parser.get_curve_info()

        # Get ultrasonic curves
        ultrasonic_cols = [col for col in df_curves.columns
                          if 'ULTRASONIC' in col and col != 'Crank Angle']

        if not ultrasonic_cols:
            print(f"  Warning: No ultrasonic curves found")
            return []

        # Initialize leak detector for classification
        detector = PhysicsBasedLeakDetector()

        results = []

        for col_name in ultrasonic_cols:
            amplitudes = df_curves[col_name].values

            # Extract features
            features = extract_features_from_waveform(amplitudes)

            # Get leak detection result
            detection_result = detector.detect_leak(amplitudes)

            # Combine metadata and features
            record = {
                'file_name': wrpm_path.name,
                'machine_id': info.get('machine_id', 'Unknown'),
                'date': str(info.get('date', 'Unknown')),
                'curve_name': col_name,
                'data_points': len(amplitudes),

                # Leak detection results
                'detected_leak': detection_result.is_leak,
                'leak_probability': detection_result.leak_probability,
                'confidence': detection_result.confidence,

                # Features
                **features
            }

            results.append(record)

            print(f"  {col_name[:50]}... - Leak: {detection_result.leak_probability:.0f}%")

        return results

    except Exception as e:
        print(f"  Error: {e}")
        return []


def process_wrpm_directory(directory_path: str, output_csv: str = None):
    """
    Process all WRPM files in a directory and create training dataset.

    Args:
        directory_path: Path to directory containing WRPM files
        output_csv: Output CSV file path (default: training_data_from_wrpm.csv)
    """
    if output_csv is None:
        output_csv = "training_data_from_wrpm.csv"

    directory = Path(directory_path)

    if not directory.exists():
        print(f"Error: Directory not found: {directory_path}")
        return

    # Find all WRPM files
    wrpm_files = list(directory.glob("*.wrpm"))

    if not wrpm_files:
        print(f"No WRPM files found in: {directory_path}")
        return

    print(f"\nFound {len(wrpm_files)} WRPM files")
    print("=" * 70)

    all_results = []

    for wrpm_file in wrpm_files:
        results = process_wrpm_file(wrpm_file)
        all_results.extend(results)

    if not all_results:
        print("\nNo data extracted!")
        return

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Save to CSV
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 70)
    print(f"Training data saved to: {output_csv}")
    print(f"Total records: {len(df)}")
    print(f"Total curves: {len(df)}")
    print(f"Files processed: {df['file_name'].nunique()}")
    print(f"Machines: {df['machine_id'].nunique()}")

    # Summary statistics
    if 'detected_leak' in df.columns:
        leak_count = df['detected_leak'].sum()
        normal_count = len(df) - leak_count
        print(f"\nDetected Leaks: {leak_count} ({leak_count/len(df)*100:.1f}%)")
        print(f"Normal Valves: {normal_count} ({normal_count/len(df)*100:.1f}%)")

    # Feature statistics
    print(f"\nFeature Statistics:")
    print(f"  Mean Amplitude: {df['mean_amplitude'].mean():.2f}G (±{df['mean_amplitude'].std():.2f})")
    print(f"  Max Amplitude: {df['max_amplitude'].mean():.2f}G (±{df['max_amplitude'].std():.2f})")
    print(f"  Above 2G Ratio: {df['above_2g_ratio'].mean():.2%} (±{df['above_2g_ratio'].std():.2%})")

    return df


def main():
    """Main entry point for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(description='Extract training data from WRPM files')
    parser.add_argument('directory', help='Directory containing WRPM files')
    parser.add_argument('--output', '-o', default='training_data_from_wrpm.csv',
                       help='Output CSV file (default: training_data_from_wrpm.csv)')
    parser.add_argument('--label', '-l', help='Manual label for all files (leak/normal)')

    args = parser.parse_args()

    # Process files
    df = process_wrpm_directory(args.directory, args.output)

    # Add manual label if specified
    if args.label and df is not None:
        df['manual_label'] = args.label.lower()
        df.to_csv(args.output, index=False)
        print(f"\nAdded manual label '{args.label}' to all records")


if __name__ == "__main__":
    main()

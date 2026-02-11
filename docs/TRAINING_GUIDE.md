# Complete AI Training Workflow Guide

## Overview

This guide shows you **exactly how to train an AI model** using your WRPM files to create a supervised machine learning leak detector.

## The Complete Pipeline

```
WRPM Files → Extract Features → Label Data → Train ML Model → Make Predictions
```

### What You'll Build

1. **Extract training data** from WRPM files (28 features per valve)
2. **Label the data** (leak vs normal)
3. **Train ML models** (XGBoost + Random Forest ensemble)
4. **Evaluate performance** (accuracy, precision, recall)
5. **Use trained model** for predictions (standalone or hybrid with physics-based)

## Step-by-Step Workflow

### Step 1: Organize Your WRPM Files

First, organize your WRPM files based on what you know about them:

```
data/
├── known_leaks/
│   ├── compressor_A_leak.wrpm
│   ├── compressor_B_leak.wrpm
│   └── ... (more leak files)
│
├── known_normal/
│   ├── compressor_C_normal.wrpm
│   ├── compressor_D_normal.wrpm
│   └── ... (more normal files)
│
└── unknown/
    └── ... (files you're not sure about)
```

**How to know if it's a leak?**
- Field reports indicating valve leaks
- Maintenance records showing valve replacements
- High vibration or acoustic readings noted
- Historical XML files you've already analyzed

### Step 2: Extract Features from WRPM Files

Extract features from each group:

```bash
cd Leak_detector/physics_based

# Extract features from leak files
python extract_training_data_from_wrpm.py "data/known_leaks" \
    --output leak_data.csv \
    --label leak

# Extract features from normal files
python extract_training_data_from_wrpm.py "data/known_normal" \
    --output normal_data.csv \
    --label normal
```

**Output:** Two CSV files with 28 features per valve.

### Step 3: Combine Training Data

Create a single training dataset:

```python
# combine_data.py
import pandas as pd

# Load both datasets
leak_df = pd.read_csv('leak_data.csv')
normal_df = pd.read_csv('normal_data.csv')

# Combine
combined_df = pd.concat([leak_df, normal_df], ignore_index=True)

# Shuffle
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
combined_df.to_csv('training_dataset.csv', index=False)

print(f"Total samples: {len(combined_df)}")
print(f"Leak samples: {(combined_df['manual_label'] == 'leak').sum()}")
print(f"Normal samples: {(combined_df['manual_label'] == 'normal').sum()}")
```

Run it:
```bash
python combine_data.py
```

### Step 4: Train the ML Model

Now train your machine learning models:

```bash
python train_ml_model.py training_dataset.csv
```

**What happens:**
1. Loads your training data
2. Splits into train/test sets (80/20)
3. Trains XGBoost model
4. Trains Random Forest model
5. Evaluates both models
6. Runs 5-fold cross-validation
7. Shows feature importance
8. Saves trained models

**Expected Output:**
```
============================================================
TRAINING MACHINE LEARNING MODELS
============================================================

Data split:
  Training: 80 samples
  Testing: 20 samples

[1/2] Training XGBoost...
[2/2] Training Random Forest...

============================================================
MODEL EVALUATION
============================================================

XGBoost Performance:
  Training Accuracy: 0.975
  Test Accuracy: 0.950
  Precision: 0.947
  Recall: 0.947
  F1 Score: 0.947
  ROC AUC: 0.985

  Confusion Matrix:
    TN: 10  FP: 1
    FN: 0   TP: 9

Random Forest Performance:
  Training Accuracy: 0.988
  Test Accuracy: 0.950
  Precision: 0.950
  Recall: 0.950
  F1 Score: 0.950
  ROC AUC: 0.990

============================================================
CROSS-VALIDATION (5-fold)
============================================================

XGBoost:
  CV Accuracy: 0.930 (+/- 0.045)

Random Forest:
  CV Accuracy: 0.940 (+/- 0.040)

============================================================
FEATURE IMPORTANCE (Top 10)
============================================================

Feature                  Importance
----------------------------------------
mean_amplitude           0.2450
above_2g_ratio           0.1820
above_3g_ratio           0.1340
median_amplitude         0.0980
std_amplitude            0.0750
max_amplitude            0.0680
above_1g_ratio           0.0520
rms                      0.0490
percentile_90            0.0410
above_5g_ratio           0.0380

============================================================
MODELS SAVED
============================================================
  Timestamped: leak_detection_model_20250101_143022.pkl
  Latest: leak_detection_model_latest.pkl
  Report: training_report_20250101_143022.txt
```

### Step 5: Use the Trained Model

#### Option A: Standalone ML Prediction

```python
from ml_leak_detector import MLLeakDetector
from wrpm_parser_ae import WrpmParserAE

# Load trained model
detector = MLLeakDetector()  # Automatically loads latest model

# Parse WRPM file
parser = WrpmParserAE('test_file.wrpm')
df = parser.parse_to_dataframe()

# Get first ultrasonic curve
ultrasonic_cols = [c for c in df.columns if 'ULTRASONIC' in c]
amplitudes = df[ultrasonic_cols[0]].values

# Detect leak
result = detector.detect_leak(amplitudes)

print(f"ML Leak Probability: {result.leak_probability:.1f}%")
print(f"Confidence: {result.confidence:.2f}")
print(f"Model Agreement: {result.model_agreement:.2f}")
print(f"  XGBoost: {result.xgb_probability:.1f}%")
print(f"  Random Forest: {result.rf_probability:.1f}%")
```

#### Option B: Hybrid Prediction (Physics + ML)

```python
from ml_leak_detector import HybridLeakDetector
from wrpm_parser_ae import WrpmParserAE

# Load hybrid detector
detector = HybridLeakDetector()  # Uses both physics and ML

# Parse file and detect
parser = WrpmParserAE('test_file.wrpm')
df = parser.parse_to_dataframe()
ultrasonic_cols = [c for c in df.columns if 'ULTRASONIC' in c]
amplitudes = df[ultrasonic_cols[0]].values

result = detector.detect_leak(amplitudes)

# Physics result
print("Physics-Based:")
print(f"  Leak Probability: {result['physics']['leak_probability']:.1f}%")
print(f"  Confidence: {result['physics']['confidence']:.2f}")

# ML result
print("\nMachine Learning:")
print(f"  Leak Probability: {result['ml']['leak_probability']:.1f}%")
print(f"  Confidence: {result['ml']['confidence']:.2f}")
print(f"  Model Agreement: {result['ml']['model_agreement']:.2f}")

# Ensemble result
print("\nEnsemble (Combined):")
print(f"  Leak Probability: {result['ensemble']['leak_probability']:.1f}%")
print(f"  Confidence: {result['ensemble']['confidence']:.2f}")
print(f"  Recommendation: {result['ensemble']['recommendation']}")
```

### Step 6: Integrate with Dashboard

Update `app.py` to use the hybrid detector:

```python
# In app.py, replace:
from leak_detector import PhysicsBasedLeakDetector

# With:
from ml_leak_detector import HybridLeakDetector

# Then use:
detector = HybridLeakDetector()
result = detector.detect_leak(amplitudes)

# Use ensemble result:
leak_prob = result['ensemble']['leak_probability']
is_leak = result['ensemble']['is_leak']
recommendation = result['ensemble']['recommendation']
```

## Data Requirements for Good ML Performance

### Minimum Requirements

| Metric | Minimum | Recommended | Ideal |
|--------|---------|-------------|-------|
| **Total samples** | 50 | 200 | 500+ |
| **Unique leak valves** | 10 | 50 | 100+ |
| **Unique normal valves** | 10 | 50 | 100+ |
| **Class balance** | 30/70 | 40/60 | 50/50 |
| **Diverse compressors** | 2 | 5 | 10+ |

### Why These Numbers?

- **50+ unique valves per class:** Prevents overfitting to specific equipment
- **200+ total samples:** Enables reliable cross-validation
- **Balanced classes:** Prevents bias toward majority class
- **Diverse compressors:** Generalizes across different machines

### What If You Don't Have Enough Data Yet?

**Option 1: Start with Physics-Based Only**
- Use the physics-based detector (already 93% on known leaks)
- Collect more WRPM files over time
- Train ML model when you have sufficient data

**Option 2: Use Hybrid Approach**
- Train ML on whatever data you have
- Use hybrid detector (combines physics + ML)
- ML will improve as you add more data
- Retrain periodically

**Option 3: Active Learning**
- Use physics detector to label uncertain cases
- Manually review borderline predictions
- Add verified examples to training set
- Iteratively improve model

## Expected Performance

### With Limited Data (50-100 samples)

```
Accuracy: 75-85%
Precision: 70-80%
Recall: 75-85%
F1 Score: 72-82%
```

**Use case:** Hybrid system where ML confirms physics-based detection

### With Good Data (200-500 samples, 50+ unique valves each class)

```
Accuracy: 90-95%
Precision: 88-93%
Recall: 90-95%
F1 Score: 89-94%
ROC AUC: 0.95-0.98
```

**Use case:** Primary ML detector with physics as backup

### With Excellent Data (500+ samples, 100+ unique valves)

```
Accuracy: 95-98%
Precision: 94-97%
Recall: 95-98%
F1 Score: 95-97%
ROC AUC: 0.98-0.99
```

**Use case:** Production ML system, physics for explainability

## Retraining Workflow

As you collect more data, retrain your model:

```bash
# 1. Extract features from new WRPM files
python extract_training_data_from_wrpm.py "new_data_directory" \
    --output new_data.csv --label [leak/normal]

# 2. Combine with existing training data
python -c "
import pandas as pd
old = pd.read_csv('training_dataset.csv')
new = pd.read_csv('new_data.csv')
combined = pd.concat([old, new], ignore_index=True)
combined.to_csv('training_dataset_v2.csv', index=False)
print(f'Total samples: {len(combined)}')
"

# 3. Retrain model
python train_ml_model.py training_dataset_v2.csv

# 4. Compare performance
# - Check training_report_*.txt files
# - Compare test accuracy, precision, recall
# - If new model is better, it's automatically saved as latest
```

## Troubleshooting

### Low Accuracy (<70%)

**Causes:**
- Insufficient training data
- Unbalanced classes (95% normal, 5% leak)
- Inconsistent labeling
- Too many features for small dataset

**Solutions:**
- Collect more data (aim for 50+ each class)
- Balance classes (downsample majority or upsample minority)
- Review and fix mislabeled samples
- Use hybrid approach until more data available

### Model Overfitting (High train accuracy, low test accuracy)

**Example:** Train: 98%, Test: 65%

**Solutions:**
- Increase training data
- Reduce model complexity (fewer trees, shallower depth)
- Add more diverse compressor types
- Use cross-validation to tune hyperparameters

### Models Disagree Frequently

**Example:** XGBoost says leak (80%), Random Forest says normal (20%)

**Causes:**
- Dataset has high variance
- Features are noisy
- Some classes are very similar

**Solutions:**
- Use ensemble probability (average of both)
- Trust the hybrid system's recommendation
- Add more training examples for borderline cases
- Review feature importance and focus on top features

## Summary

### Complete Workflow Recap

1. **Organize WRPM files** by leak status
2. **Extract features** → CSV files with 28 features
3. **Combine and label** → Single training dataset
4. **Train models** → XGBoost + Random Forest
5. **Evaluate** → Check metrics, feature importance
6. **Deploy** → Use standalone ML or hybrid detector
7. **Collect more data** → Retrain periodically

### Files You'll Create

```
training_data/
├── leak_data.csv                      # Features from leak files
├── normal_data.csv                    # Features from normal files
├── training_dataset.csv               # Combined training data
├── leak_detection_model_latest.pkl    # Trained model
└── training_report_*.txt              # Performance metrics
```

### Scripts Available

| Script | Purpose |
|--------|---------|
| `extract_training_data_from_wrpm.py` | Extract features from WRPM files |
| `train_ml_model.py` | Train XGBoost + Random Forest models |
| `ml_leak_detector.py` | ML inference and hybrid detection |
| `app.py` | Streamlit dashboard (can use hybrid detector) |

### Next Steps

1. **Start small:** Even 20 leak + 20 normal samples can train a useful model
2. **Iterate:** Train → Evaluate → Collect more data → Retrain
3. **Use hybrid:** Combine ML with physics-based for best results
4. **Track performance:** Keep training reports to see improvement over time

You now have everything you need to train AI models on your WRPM data! 🚀

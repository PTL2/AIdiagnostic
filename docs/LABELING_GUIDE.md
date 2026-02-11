# Interactive Data Labeling Guide

## Overview

The **Interactive Labeling App** makes it easy to build your training dataset right in the browser!

## How It Works

```
Upload File → See AI Prediction → Label (Leak/Normal) → Export for Training
```

## Quick Start

### 1. Launch the Labeling App

```bash
cd Leak_detector/physics_based
streamlit run labeling_app.py
```

The app opens at `http://localhost:8501`

### 2. Label Your Files

**For each WRPM or XML file:**

1. **Upload** the file
2. **Review** the AI's analysis:
   - Waveform visualization
   - Mean amplitude
   - Leak probability
   - Key features
3. **Decide** based on:
   - AI's suggestion
   - Field reports
   - Maintenance records
   - Visual pattern (smear vs spike)
4. **Click** the appropriate button:
   - 🔴 **LEAK** - Confirmed valve leak
   - 🟢 **NORMAL** - Healthy valve
   - ⏭️ **SKIP** - Uncertain (don't label)

### 3. Build Your Dataset

The app automatically:
- ✅ Saves labels to `training_labels.json`
- ✅ Tracks statistics (total, leaks, normal)
- ✅ Shows AI agreement rate
- ✅ Prevents duplicate labeling

### 4. Export for Training

When you have enough labels (recommended: 20+ samples):

1. Click **"Export to CSV"** in sidebar
2. Download the CSV file
3. Train your model:
   ```bash
   python train_ml_model.py your_labeled_data.csv
   ```

## Labeling Best Practices

### How to Decide: Leak vs Normal

**Label as LEAK if:**
- ✅ Field report indicates leak
- ✅ Valve was replaced due to leakage
- ✅ High sustained amplitude (>3G mean)
- ✅ "Smear" pattern (continuous elevation)
- ✅ Maintenance records show leak

**Label as NORMAL if:**
- ✅ Recent valve inspection showed healthy
- ✅ Low mean amplitude (<2G)
- ✅ "Spike" pattern (brief peaks only)
- ✅ No historical leak issues
- ✅ Compressor running well

**SKIP if:**
- ⚠️ No field data available
- ⚠️ Unclear waveform pattern
- ⚠️ Borderline amplitude (2-3G)
- ⚠️ Uncertain about valve history

### Quality Over Quantity

**Better to have:**
- 20 high-quality, verified labels
- Than 100 uncertain/guessed labels

**Trust your sources:**
1. Field inspection reports (best)
2. Maintenance records
3. Operator notes
4. AI suggestion (use as guide, not truth)

## Understanding the AI Prediction

The labeling app shows you what the **physics-based AI** thinks (before you train ML):

```
AI Analysis:
  ⚠️ LEAK DETECTED
  Probability: 93%
  Confidence: 0.85
  Mean Amplitude: 4.59G
```

**Use this to:**
- Validate your own assessment
- Catch patterns you might miss
- Learn what "typical" leaks look like

**But remember:**
- AI can be wrong (especially on edge cases)
- **Your label is the ground truth** for training
- Field reports > AI prediction

## Example Workflow

### Scenario: You have 30 WRPM files

**Step 1: Organize**
```
my_wrpm_files/
├── compressor_A_2024_01.wrpm  (leak confirmed by field team)
├── compressor_A_2024_02.wrpm  (normal - pre-maintenance)
├── compressor_B_2024_01.wrpm  (leak - valve replaced)
└── ... (27 more files)
```

**Step 2: Label interactively**
```bash
streamlit run labeling_app.py
```

For each file:
- Upload → Review → Label
- Takes ~2 minutes per file
- Total time: ~1 hour for 30 files

**Step 3: Monitor progress**
```
Sidebar shows:
  Total Labeled: 30
  Leaks: 15
  Normal: 15
  Balance: 50% / 50% ✅
```

**Step 4: Export**
```
Click "Export to CSV"
Downloads: training_data_20250101_143000.csv
```

**Step 5: Train**
```bash
python train_ml_model.py training_data_20250101_143000.csv
```

## Features of the Labeling App

### Visual Feedback

- **Waveform plot** with mean amplitude line
- **Color-coded** prediction (red=leak, green=normal)
- **Key metrics** displayed prominently
- **Real-time statistics** in sidebar

### Data Safety

- **Auto-save** after each label
- **Persistent storage** in `training_labels.json`
- **Duplicate detection** prevents relabeling same file
- **Export anytime** to CSV

### Quality Control

- **AI agreement rate** shows how often AI matches your labels
  - High agreement (>80%): Good quality labels
  - Low agreement (<50%): Review your labeling criteria
- **Class balance** tracker ensures balanced dataset
- **Summary table** shows all labels at a glance

### Efficiency

- **One file at a time** - focused labeling
- **Quick decision** - clear buttons
- **Skip option** - don't force uncertain labels
- **Batch export** - download all at once

## File Outputs

### training_labels.json

**Format:**
```json
[
  {
    "file_name": "compressor_A.wrpm",
    "curve_name": "Machine - C.1CD.ULTRASONIC...",
    "manual_label": "leak",
    "ai_prediction": "leak",
    "ai_probability": 93.0,
    "features": {
      "mean_amplitude": 4.59,
      "above_2g_ratio": 0.92,
      ...
    },
    "labeled_at": "2025-01-01T14:30:00",
    "labeled_by": "User"
  },
  ...
]
```

**Purpose:** Persistent storage of all your labels

### Exported CSV

**Columns:**
- `file_name`, `machine_id`, `date`, `curve_name`
- `manual_label` (your label - used for training)
- `ai_suggested` (what physics AI predicted)
- `labeled_by`, `labeled_at`
- `mean_amplitude`, `above_2g_ratio`, ... (28 features)

**Purpose:** Ready for `train_ml_model.py`

## Tips for Efficient Labeling

### Batch by Source

Label files from the same source together:
1. Known leaks (from maintenance reports)
2. Known normal (from healthy compressors)
3. Uncertain (review AI predictions)

### Use Field Context

Keep field reports handy:
- Valve replacement dates
- Inspection notes
- Operator observations

### Take Breaks

- Label in sessions (10-15 files at a time)
- Review your labels periodically
- Check AI agreement rate for consistency

### Start with Clear Cases

1. Label obvious leaks first (mean >4G)
2. Label obvious normal next (mean <2G)
3. Save borderline cases for last
4. Skip truly uncertain ones

## Integration with ML Training

### Minimum Dataset

```
Recommended minimum:
  - 10 leak samples
  - 10 normal samples
  - Total: 20 labeled files

Better:
  - 25 leak samples
  - 25 normal samples
  - Total: 50 labeled files

Ideal:
  - 50+ leak samples
  - 50+ normal samples
  - Total: 100+ labeled files
```

### Training Command

```bash
# After exporting from labeling app:
python train_ml_model.py exported_data.csv
```

Expected output:
```
Loaded 50 samples
  Leak samples: 25 (50.0%)
  Normal samples: 25 (50.0%)

Training XGBoost...
Training Random Forest...

Test Accuracy: 0.90
Precision: 0.88
Recall: 0.92
```

## Troubleshooting

### "File already labeled" Warning

**Cause:** File was previously labeled

**Solution:**
- View existing label
- Or delete from `training_labels.json` and relabel

### Labels Not Saving

**Cause:** Permission error or disk full

**Solution:**
- Check write permissions
- Manually save: Click "Save Labels" button
- Backup `training_labels.json` regularly

### Export Shows No Data

**Cause:** No labels created yet

**Solution:**
- Label at least one file first
- Check sidebar shows > 0 labeled samples

### AI Prediction Seems Wrong

**Cause:** Physics-based AI has limitations

**Solution:**
- Trust field reports over AI
- Label with ground truth
- ML will learn from your corrections!

## Summary

The **Interactive Labeling App** is your tool for building high-quality training datasets:

✅ **User-friendly** - Browser-based, no command-line needed
✅ **Visual** - See waveforms and predictions
✅ **Efficient** - Label 30 files in ~1 hour
✅ **Safe** - Auto-save, no data loss
✅ **Quality-focused** - Skip uncertain cases
✅ **ML-ready** - Export directly to CSV for training

**Workflow:**
```
Launch App → Upload Files → Review & Label → Export → Train ML
```

**Result:**
Custom-trained ML model that learns from YOUR labeled examples!

---

**Questions?**
- Check `AI_TRAINING_COMPLETE_GUIDE.md` for ML training details
- See `WRPM_SUPPORT_GUIDE.md` for WRPM file handling

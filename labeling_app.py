"""
Interactive Data Labeling App for Training Data Collection

This app allows you to:
1. Upload WRPM or XML files
2. See the AI's prediction
3. Manually label as LEAK or NORMAL
4. Build your training dataset
5. Export labeled data for ML training
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
from unified_data_loader import load_valve_data, get_ultrasonic_curves
from leak_detector import PhysicsBasedLeakDetector
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="Training Data Labeler",
    page_icon="🏷️",
    layout="wide"
)

# Initialize session state
if 'labeled_data' not in st.session_state:
    st.session_state.labeled_data = []
if 'current_file_processed' not in st.session_state:
    st.session_state.current_file_processed = False

# Load existing labels if available
LABELS_FILE = "training_labels.json"

def load_existing_labels():
    """Load previously saved labels."""
    if Path(LABELS_FILE).exists():
        with open(LABELS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_labels():
    """Save labels to file."""
    with open(LABELS_FILE, 'w') as f:
        json.dump(st.session_state.labeled_data, f, indent=2)
    st.success(f"✅ Saved {len(st.session_state.labeled_data)} labeled samples to {LABELS_FILE}")

def export_to_csv():
    """Export labeled data to CSV format for training."""
    if not st.session_state.labeled_data:
        st.warning("No labeled data to export!")
        return None

    # Convert to DataFrame
    records = []
    for item in st.session_state.labeled_data:
        record = {
            'file_name': item['file_name'],
            'machine_id': item.get('machine_id', 'Unknown'),
            'date': item.get('date', 'Unknown'),
            'curve_name': item['curve_name'],
            'manual_label': item['manual_label'],
            'ai_suggested': item.get('ai_prediction', 'Unknown'),
            'labeled_by': item.get('labeled_by', 'User'),
            'labeled_at': item.get('labeled_at', datetime.now().isoformat()),
            **item['features']
        }
        records.append(record)

    df = pd.DataFrame(records)
    return df

# Title
st.markdown('<h1 style="color: #1976d2;">🏷️ Training Data Labeling Tool</h1>', unsafe_allow_html=True)
st.markdown("""
Upload WRPM or XML files, review the AI's prediction, and provide the correct label.
Build your training dataset for machine learning!
""")

# Sidebar - Statistics
st.sidebar.header("📊 Labeling Statistics")

# Load existing labels on startup
if not st.session_state.labeled_data:
    st.session_state.labeled_data = load_existing_labels()

total_labeled = len(st.session_state.labeled_data)
leak_count = sum(1 for item in st.session_state.labeled_data if item['manual_label'] == 'leak')
normal_count = total_labeled - leak_count

st.sidebar.metric("Total Labeled", total_labeled)
col1, col2 = st.sidebar.columns(2)
col1.metric("Leaks", leak_count)
col2.metric("Normal", normal_count)

if total_labeled > 0:
    st.sidebar.progress(leak_count / total_labeled if total_labeled > 0 else 0)
    st.sidebar.caption(f"Class balance: {leak_count/total_labeled*100:.1f}% leak / {normal_count/total_labeled*100:.1f}% normal")

st.sidebar.markdown("---")

# Export buttons
if st.sidebar.button("💾 Save Labels", type="primary", width='stretch'):
    save_labels()

if st.sidebar.button("📤 Export to CSV", width='stretch'):
    df = export_to_csv()
    if df is not None:
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width='stretch'
        )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Labeling Tips")
st.sidebar.markdown("""
- Review the waveform pattern
- Check mean amplitude
- Consider AI suggestion
- Trust field reports
- When uncertain, skip it
""")

# Main content
st.markdown("---")

# File uploader
st.subheader("📁 Upload File to Label")
uploaded_file = st.file_uploader(
    "Choose a WRPM or XML file",
    type=['xml', 'wrpm'],
    help="Upload one file at a time to label"
)

if uploaded_file is not None:
    file_name = uploaded_file.name

    # Check if already labeled
    already_labeled = any(item['file_name'] == file_name for item in st.session_state.labeled_data)

    if already_labeled:
        st.warning(f"⚠️ File '{file_name}' has already been labeled!")
        if st.button("View existing label"):
            existing = next(item for item in st.session_state.labeled_data if item['file_name'] == file_name)
            st.info(f"Current label: **{existing['manual_label'].upper()}**")

    try:
        # Load and analyze file
        with st.spinner("Analyzing file..."):
            df_curves, metadata, file_type = load_valve_data(uploaded_file)

        st.success(f"✅ Loaded {file_type} file: {file_name}")

        # Display metadata
        col1, col2, col3 = st.columns(3)
        col1.metric("File Type", file_type)
        col2.metric("Machine", metadata.get('machine_id', 'Unknown')[:30])
        col3.metric("Date", str(metadata.get('date', 'Unknown'))[:10])

        # Get ultrasonic curves
        ultrasonic_cols = get_ultrasonic_curves(df_curves)

        if not ultrasonic_cols:
            st.error("No ultrasonic curves found in file!")
        else:
            st.info(f"Found {len(ultrasonic_cols)} ultrasonic curve(s)")

            # Analyze each curve
            detector = PhysicsBasedLeakDetector()

            for idx, col_name in enumerate(ultrasonic_cols):
                st.markdown("---")
                st.markdown(f"### 📊 Curve {idx + 1}: {col_name[:80]}...")

                amplitudes = df_curves[col_name].values
                result = detector.detect_leak(amplitudes)

                # Display analysis
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Plot waveform
                    fig = go.Figure()
                    crank_angles = df_curves['Crank Angle'].values

                    fig.add_trace(go.Scatter(
                        x=crank_angles,
                        y=amplitudes,
                        mode='lines',
                        name='Amplitude',
                        line=dict(color='#1976d2', width=2)
                    ))

                    # Add mean line
                    mean_amp = result.feature_values['mean_amplitude']
                    fig.add_hline(
                        y=mean_amp,
                        line_dash="dash",
                        line_color="red" if result.is_leak else "green",
                        annotation_text=f"Mean: {mean_amp:.2f}G"
                    )

                    # Add threshold line
                    fig.add_hline(
                        y=2.0,
                        line_dash="dot",
                        line_color="orange",
                        annotation_text="Threshold: 2.0G"
                    )

                    fig.update_layout(
                        title="Ultrasonic Waveform",
                        xaxis_title="Crank Angle (degrees)",
                        yaxis_title="Amplitude (G)",
                        height=400,
                        showlegend=False
                    )

                    st.plotly_chart(fig, width='stretch')

                with col2:
                    # AI Prediction
                    st.markdown("#### 🤖 AI Analysis")

                    if result.is_leak:
                        st.markdown(f"""
                        <div style='background-color: #ffebee; padding: 1rem; border-radius: 8px; border-left: 4px solid #c62828;'>
                            <h3 style='color: #c62828; margin: 0;'>⚠️ LEAK DETECTED</h3>
                            <p style='margin: 0.5rem 0 0 0;'>Probability: <b>{result.leak_probability:.0f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background-color: #e8f5e9; padding: 1rem; border-radius: 8px; border-left: 4px solid #2e7d32;'>
                            <h3 style='color: #2e7d32; margin: 0;'>✓ NORMAL</h3>
                            <p style='margin: 0.5rem 0 0 0;'>Leak Probability: <b>{result.leak_probability:.0f}%</b></p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.metric("Confidence", f"{result.confidence:.2f}")
                    st.metric("Mean Amplitude", f"{mean_amp:.2f}G")

                    # Key features
                    st.markdown("**Key Features:**")
                    st.caption(f"Above 2G: {result.feature_values.get('above_2g_ratio', 0)*100:.1f}%")
                    st.caption(f"Above 3G: {result.feature_values.get('above_3g_ratio', 0)*100:.1f}%")
                    st.caption(f"Max: {result.feature_values.get('max_amplitude', 0):.2f}G")

                # Labeling interface
                st.markdown("---")
                st.markdown("### 🏷️ Your Label")

                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("🔴 LEAK", key=f"leak_{idx}", type="primary", width='stretch'):
                        # Save label
                        label_data = {
                            'file_name': file_name,
                            'curve_name': col_name,
                            'machine_id': metadata.get('machine_id', 'Unknown'),
                            'date': str(metadata.get('date', 'Unknown')),
                            'manual_label': 'leak',
                            'ai_prediction': 'leak' if result.is_leak else 'normal',
                            'ai_probability': result.leak_probability,
                            'features': result.feature_values,
                            'labeled_at': datetime.now().isoformat(),
                            'labeled_by': 'User'
                        }
                        st.session_state.labeled_data.append(label_data)
                        save_labels()
                        st.success("✅ Labeled as LEAK!")
                        st.balloons()

                with col2:
                    if st.button("🟢 NORMAL", key=f"normal_{idx}", type="secondary", width='stretch'):
                        # Save label
                        label_data = {
                            'file_name': file_name,
                            'curve_name': col_name,
                            'machine_id': metadata.get('machine_id', 'Unknown'),
                            'date': str(metadata.get('date', 'Unknown')),
                            'manual_label': 'normal',
                            'ai_prediction': 'leak' if result.is_leak else 'normal',
                            'ai_probability': result.leak_probability,
                            'features': result.feature_values,
                            'labeled_at': datetime.now().isoformat(),
                            'labeled_by': 'User'
                        }
                        st.session_state.labeled_data.append(label_data)
                        save_labels()
                        st.success("✅ Labeled as NORMAL!")

                with col3:
                    if st.button("⏭️ SKIP", key=f"skip_{idx}", width='stretch'):
                        st.info("Skipped - no label saved")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

else:
    st.info("👆 Upload a file to start labeling")

# Display labeled data
st.markdown("---")
st.subheader("📋 Labeled Data Summary")

if st.session_state.labeled_data:
    # Create summary DataFrame
    summary_data = []
    for item in st.session_state.labeled_data:
        summary_data.append({
            'File': item['file_name'][:40],
            'Curve': item['curve_name'][:50],
            'Label': item['manual_label'].upper(),
            'AI Predicted': item.get('ai_prediction', 'N/A').upper(),
            'Agreement': '✓' if item['manual_label'] == item.get('ai_prediction') else '✗',
            'Mean Amp': f"{item['features']['mean_amplitude']:.2f}G",
            'Labeled': item.get('labeled_at', 'Unknown')[:10]
        })

    df_summary = pd.DataFrame(summary_data)

    # Style the dataframe
    def highlight_agreement(row):
        if row['Agreement'] == '✓':
            return ['background-color: #e8f5e9'] * len(row)
        else:
            return ['background-color: #fff3e0'] * len(row)

    st.dataframe(
        df_summary.style.apply(highlight_agreement, axis=1),
        width='stretch',
        height=400
    )

    # Statistics
    agreement_rate = sum(1 for item in st.session_state.labeled_data
                        if item['manual_label'] == item.get('ai_prediction')) / len(st.session_state.labeled_data)

    st.info(f"📊 AI Agreement Rate: {agreement_rate*100:.1f}% - The AI agrees with your labels {agreement_rate*100:.1f}% of the time")

    # Ready to train?
    if total_labeled >= 20:
        st.success(f"""
        ✅ You have {total_labeled} labeled samples!

        **Next steps:**
        1. Click "Export to CSV" in the sidebar
        2. Download the CSV file
        3. Train your ML model: `python train_ml_model.py your_data.csv`
        """)
    else:
        st.warning(f"You have {total_labeled} labeled samples. Recommend at least 20 (10 leak + 10 normal) to start training.")

else:
    st.info("No labeled data yet. Start labeling files above!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <small>Valve Leak Detection Training Data Labeler | Save your progress frequently!</small>
</div>
""", unsafe_allow_html=True)

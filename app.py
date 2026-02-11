"""
Valve Leak Detection - Physics-Based System
Streamlit app using ultrasonic sensor physics for leak detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xml_parser import parse_curves_xml, get_curve_info
from leak_detector import PhysicsBasedLeakDetector
from unified_data_loader import load_valve_data, get_ultrasonic_curves, load_wrpm_pressure_data
import re

# Page configuration
st.set_page_config(
    page_title="Valve Leak Detection - AI Pattern Recognition",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    /* Main headers */
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Result boxes */
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .leak-detected {
        background-color: #ffebee;
        color: #c62828;
        border: 3px solid #c62828;
    }
    .normal-detected {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }

    /* Explanation boxes */
    .physics-explanation {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        font-family: monospace;
        white-space: pre-wrap;
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1976d2;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }

    /* Info cards */
    .info-card {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1976d2;
        margin: 1rem 0;
    }

    /* Metric improvements */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }

    /* Improve spacing */
    .stButton button {
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def calculate_envelope(amplitudes, window_size=15):
    """Calculate upper and lower envelopes using rolling max/min"""
    # Convert to pandas Series for rolling operations
    signal = pd.Series(amplitudes)

    # Calculate upper envelope (rolling max)
    upper = signal.rolling(window=window_size, center=True).max()

    # Calculate lower envelope (rolling min)
    lower = signal.rolling(window=window_size, center=True).min()

    # Fill NaN values at edges
    upper = upper.fillna(signal)
    lower = lower.fillna(signal)

    return upper.values, lower.values

def create_waveform_plot(amplitudes, crank_angles, is_leak, mean_amp, valve_name):
    """Create interactive waveform visualization with mirrored signal and envelope"""

    # Create figure
    fig = go.Figure()

    # Determine color based on leak status
    waveform_color = 'rgba(66, 165, 245, 0.4)'  # Blue, semi-transparent for filled area
    envelope_color = '#c62828' if is_leak else '#2e7d32'  # Red for leak, green for normal

    # Calculate upper envelope only
    upper_env, _ = calculate_envelope(amplitudes)

    # Add filled mirrored waveform using fill='tonexty'
    # First add the negative (mirrored) trace
    fig.add_trace(go.Scatter(
        x=crank_angles,
        y=-amplitudes,  # Negative values for mirror effect
        mode='lines',
        name='Waveform',
        line=dict(color=waveform_color, width=0),
        fill=None,
        showlegend=False,
        hoverinfo='skip'
    ))

    # Then add the positive trace with fill to create the envelope band
    fig.add_trace(go.Scatter(
        x=crank_angles,
        y=amplitudes,
        mode='lines',
        name='Waveform',
        line=dict(color=waveform_color, width=0),
        fill='tonexty',  # Fill to previous trace (creates the envelope band)
        fillcolor=waveform_color,
        showlegend=False,
        hovertemplate='<b>Crank Angle:</b> %{x}°<br><b>Amplitude:</b> %{y:.2f}G<extra></extra>'
    ))

    # Add upper envelope line (red/green on top)
    fig.add_trace(go.Scatter(
        x=crank_angles,
        y=upper_env,
        mode='lines',
        name='Amplitude Envelope',
        line=dict(color=envelope_color, width=2.5),
        hovertemplate='<b>Peak:</b> %{y:.2f}G<extra></extra>'
    ))

    # Add mean amplitude reference line
    fig.add_hline(
        y=mean_amp,
        line_dash="dash",
        line_color=envelope_color,
        annotation_text=f"Mean: {mean_amp:.2f}G",
        annotation_position="right"
    )

    # Add 2G threshold reference (leak indicator)
    fig.add_hline(
        y=2.0,
        line_dash="dot",
        line_color="orange",
        annotation_text="2G Threshold",
        annotation_position="left"
    )

    # Update layout
    pattern_type = "SMEAR PATTERN (Leak)" if is_leak else "SPIKE PATTERN (Normal)"
    title_color = '#c62828' if is_leak else '#2e7d32'

    fig.update_layout(
        title={
            'text': f"<b>{valve_name}</b><br><span style='font-size:14px; color:{title_color}'>{pattern_type}</span>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title="<b>Crank Angle (degrees)</b>",
        yaxis_title="<b>Amplitude (G)</b>",
        height=450,
        hovermode='x unified',
        showlegend=False,
        template='plotly_white',
        font=dict(size=13),
        margin=dict(l=60, r=40, t=80, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='#666'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='#666'
        )
    )

    return fig


def create_pressure_plot(pressure_data, crank_angles, channel_name):
    """
    Create pressure curve visualization with TDC/BDC markers.

    Args:
        pressure_data: Array of pressure values (PSI)
        crank_angles: Array of crank angle values (degrees)
        channel_name: Name of the pressure channel

    Returns:
        Plotly figure object
    """
    fig = go.Figure()

    # Add pressure curve
    fig.add_trace(go.Scatter(
        x=crank_angles,
        y=pressure_data,
        mode='lines',
        name='Pressure',
        line=dict(color='#1976d2', width=2),
        hovertemplate='<b>Crank Angle:</b> %{x}°<br><b>Pressure:</b> %{y:.1f} PSI<extra></extra>'
    ))

    # Add TDC marker at 0° (and 360° if in range)
    max_angle = max(crank_angles)
    fig.add_vline(
        x=0,
        line_dash="dash",
        line_color="#666666",
        line_width=1.5,
        annotation_text="TDC",
        annotation_position="top",
        annotation_font_color="#666666"
    )

    if max_angle >= 360:
        fig.add_vline(
            x=360,
            line_dash="dash",
            line_color="#666666",
            line_width=1.5,
            annotation_text="TDC",
            annotation_position="top",
            annotation_font_color="#666666"
        )

    # Add BDC marker at 180°
    fig.add_vline(
        x=180,
        line_dash="dash",
        line_color="#999999",
        line_width=1.5,
        annotation_text="BDC",
        annotation_position="top",
        annotation_font_color="#999999"
    )

    # Extract cylinder info from channel name for title
    # Format: "Machine - C.1P.PVPT (PRESSURE).1P"
    import re
    match = re.search(r'C\.(\d+)P', channel_name)
    if match:
        cyl_num = match.group(1)
        title = f"Cylinder {cyl_num} - Pressure (PVPT)"
    else:
        title = channel_name

    # Update layout
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title="<b>Crank Angle (degrees)</b>",
        yaxis_title="<b>Pressure (PSI)</b>",
        height=400,
        hovermode='x unified',
        showlegend=False,
        template='plotly_white',
        font=dict(size=12),
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            zeroline=False,
            range=[0, max_angle]
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e0e0e0',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='#ccc'
        )
    )

    return fig


# Header
st.markdown('<div class="main-header">🤖 AI-Powered Valve Leak Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Intelligent Pattern Recognition for Compressors</div>', unsafe_allow_html=True)

# Introduction
with st.expander("About This System", expanded=False):
    st.markdown("""
    ### AI-Powered Leak Detection

    This system uses **AI pattern recognition** to detect valve leaks from ultrasonic acoustic emission (AE) sensor data.

    **The Science:**

    Ultrasonic sensors (36-44 KHz narrow band) detect acoustic emissions from valve events:
    - **NORMAL valve**: Clean closure creates brief acoustic spikes → **LOW mean amplitude** (~1-2G)
    - **LEAKING valve**: Gas escaping through gaps creates continuous noise → **HIGH mean amplitude** (~3-5G)

    **How It Works:**
    1. Upload a Windrock **XML** or **WRPM** file containing ultrasonic sensor data
    2. AI extracts and analyzes AE waveform data for each valve
    3. Pattern recognition identifies "smear" patterns (sustained elevation = leak) vs "spike" patterns (brief peaks = normal)
    4. Results show leak probability, confidence score, and detailed explanation

    **AI Detection Thresholds:**
    | Mean Amplitude | Classification |
    |----------------|----------------|
    | < 2G | Normal operation |
    | 2-3G | Possible concern |
    | 3-4G | Likely leak |
    | > 4G | Probable leak |
    | > 5G | Severe leak |

    **Key Features:**
    - Analyzes all cylinders and valves in a single file
    - Color-coded results for quick identification
    - Explainable AI results with physics explanations
    - Interactive waveform visualizations
    """)

st.markdown("---")

# File uploader
st.subheader("Upload Valve Data (XML or WRPM File)")
uploaded_file = st.file_uploader(
    "Choose a Windrock data file",
    type=['xml', 'wrpm'],
    help="Upload a Windrock Curves XML file OR WRPM file containing ultrasonic sensor readings"
)

if uploaded_file is not None:
    try:
        # Display file info
        file_type_display = "WRPM" if uploaded_file.name.lower().endswith('.wrpm') else "XML"
        st.success(f"File uploaded: **{uploaded_file.name}** ({file_type_display} format)")

        # Load data using unified loader
        with st.spinner(f"Analyzing {file_type_display} file..."):
            df_curves, metadata, file_type = load_valve_data(uploaded_file)

        # Load pressure data if WRPM file
        df_pressure = None
        if file_type == 'WRPM' and metadata.get('has_pressure_data', False):
            uploaded_file.seek(0)  # Reset file pointer
            df_pressure = load_wrpm_pressure_data(uploaded_file)

        # Display metadata
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Curves", metadata.get('total_curves', 0))
        col2.metric("AE Curves Found", len(metadata.get('ae_curves', [])))
        col3.metric("Data Points", metadata.get('data_points', 0))
        col4.metric("Crank Angle Range", metadata.get('crank_angle_range', '0-360°'))

        # Show file type badge with machine type
        if file_type == 'WRPM':
            machine_type = metadata.get('machine_type', 'compressor')
            is_engine = metadata.get('is_engine', False)
            machine_type_display = "Engine" if is_engine else "Compressor"

            st.info(f"📊 **WRPM File** - Machine: {metadata.get('machine_id', 'Unknown')} | Type: {machine_type_display} | Date: {metadata.get('date', 'Unknown')}")

            # Warning for engine files
            if is_engine:
                st.warning("⚠️ **Engine File Detected** - This file is from an engine (720° crank angle), not a compressor. Valve leak detection is designed for compressors only. Analysis results may not be applicable.")

        st.markdown("---")

        # Check if file is from engine
        is_engine_file = metadata.get('is_engine', False)

        # Analyze button
        if st.button("Analyze All Cylinders", type="primary", width='stretch', disabled=is_engine_file):
            with st.spinner("Analyzing ultrasonic patterns..."):
                if df_curves is None or len(df_curves) == 0:
                    st.error("Failed to parse file. Please check file format.")
                else:
                    # Find all ULTRASONIC curves
                    ultrasonic_cols = get_ultrasonic_curves(df_curves)

                if not ultrasonic_cols:
                    st.error("No ULTRASONIC/AE curves found in file.")
                else:
                    st.info(f"Found {len(ultrasonic_cols)} ultrasonic curves to analyze")

                    # Initialize detector
                    detector = PhysicsBasedLeakDetector()

                    # Analyze each valve
                    all_results = []

                    for col in ultrasonic_cols:
                        amplitudes = df_curves[col].values
                        result = detector.detect_leak(amplitudes)

                        # Parse valve info from column name
                        # Format: "C402 - C.3CD1.ULTRASONIC G 36KHZ - 44KHZ (NARROW BAND).3CD1"
                        parts = col.split('.')
                        if len(parts) >= 2:
                            valve_id = parts[1] if len(parts) > 1 else col
                            # Extract cylinder number
                            cyl_match = re.search(r'(\d+)', valve_id)
                            cyl_num = int(cyl_match.group(1)) if cyl_match else 0
                            # Determine valve position
                            if 'CS' in valve_id:
                                valve_pos = 'Crank Suction'
                            elif 'CD' in valve_id:
                                valve_pos = 'Crank Discharge'
                            elif 'HS' in valve_id:
                                valve_pos = 'Head Suction'
                            elif 'HD' in valve_id:
                                valve_pos = 'Head Discharge'
                            else:
                                valve_pos = valve_id
                        else:
                            valve_id = col
                            cyl_num = 0
                            valve_pos = col

                        all_results.append({
                            'column': col,
                            'valve_id': valve_id,
                            'cylinder_num': cyl_num,
                            'valve_position': valve_pos,
                            'result': result
                        })

                    # Group by cylinder
                    cylinders = {}
                    for item in all_results:
                        cyl_num = item['cylinder_num']
                        if cyl_num not in cylinders:
                            cylinders[cyl_num] = []
                        cylinders[cyl_num].append(item)

                    # Calculate cylinder-level status
                    cylinder_status = {}
                    for cyl_num, valves in cylinders.items():
                        max_prob = max(v['result'].leak_probability for v in valves)
                        has_leak = any(v['result'].is_leak for v in valves)
                        leak_count = sum(1 for v in valves if v['result'].is_leak)
                        cylinder_status[cyl_num] = {
                            'has_leak': has_leak,
                            'max_leak_prob': max_prob,
                            'leak_count': leak_count
                        }

                    # Display overall summary
                    st.markdown('<div class="section-header">📋 Analysis Results Summary</div>', unsafe_allow_html=True)

                    leaking_cylinders = [cyl for cyl, status in cylinder_status.items() if status['has_leak']]
                    total_cylinders = len([c for c in cylinder_status.keys() if c != 0])
                    total_valves = len(all_results)

                    if leaking_cylinders:
                        st.markdown(
                            f'<div class="result-box leak-detected">⚠️ LEAKS DETECTED IN {len(leaking_cylinders)} OF {total_cylinders} CYLINDER(S)</div>',
                            unsafe_allow_html=True
                        )
                        st.error(f"**Affected Cylinders:** {', '.join([f'Cylinder {c}' for c in sorted(leaking_cylinders)])}")
                        st.warning("**⚡ Recommendation:** Schedule immediate maintenance inspection for affected cylinders to prevent efficiency loss and potential damage.")

                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Cylinders", total_cylinders)
                        with col2:
                            st.metric("Leaking Cylinders", len(leaking_cylinders), delta=f"{len(leaking_cylinders)}/{total_cylinders}")
                        with col3:
                            st.metric("Total Valves Analyzed", total_valves)
                    else:
                        st.markdown(
                            f'<div class="result-box normal-detected">✅ ALL {total_cylinders} CYLINDERS OPERATING NORMALLY</div>',
                            unsafe_allow_html=True
                        )
                        st.success("**✓ Status:** All valves operating within normal parameters. No maintenance required at this time.")

                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Cylinders", total_cylinders)
                        with col2:
                            st.metric("Normal Cylinders", total_cylinders, delta="100%")
                        with col3:
                            st.metric("Total Valves Analyzed", total_valves)

                    st.markdown("---")

                    # Display per-cylinder results
                    st.markdown('<div class="section-header">🔍 Detailed Cylinder-by-Cylinder Analysis</div>', unsafe_allow_html=True)

                    for cyl_num in sorted(cylinders.keys()):
                        if cyl_num == 0:
                            continue  # Skip if no cylinder number

                        valves = cylinders[cyl_num]
                        status = cylinder_status[cyl_num]

                        # Cylinder header with better visual separation
                        st.markdown("---")
                        if status['has_leak']:
                            st.markdown(f"""
                            <div style='background-color: #ffebee; padding: 1rem; border-radius: 8px; border-left: 5px solid #c62828;'>
                                <h3 style='color: #c62828; margin: 0;'>
                                    ⚠️ Cylinder {cyl_num} - LEAK DETECTED ({status['leak_count']} valve(s))
                                </h3>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style='background-color: #e8f5e9; padding: 1rem; border-radius: 8px; border-left: 5px solid #2e7d32;'>
                                <h3 style='color: #2e7d32; margin: 0;'>
                                    ✓ Cylinder {cyl_num} - Normal Operation
                                </h3>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("")  # Add spacing

                        # Create table for this cylinder's valves
                        valve_results = []
                        for valve in valves:
                            r = valve['result']
                            status_display = "LEAK" if r.is_leak else "Normal"

                            valve_results.append({
                                "Valve": valve['valve_id'],
                                "Position": valve['valve_position'],
                                "Status": status_display,
                                "Leak Probability": f"{r.leak_probability:.1f}%",
                                "Mean Amp": f"{r.feature_values['mean_amplitude']:.2f}G",
                                "Max Amp": f"{r.feature_values['max_amplitude']:.2f}G",
                                "Confidence": f"{r.confidence:.1%}"
                            })

                        df_results = pd.DataFrame(valve_results)

                        # Color code the dataframe
                        def highlight_leaks(row):
                            if "LEAK" in row['Status']:
                                return ['background-color: #ffebee'] * len(row)
                            return [''] * len(row)

                        def style_leak_probability(val):
                            """Color code leak probability based on risk level"""
                            try:
                                # Extract numeric value (remove %)
                                prob = float(val.replace('%', ''))

                                if prob <= 30:
                                    return 'background-color: #e8f5e9; color: #2e7d32; font-weight: bold'  # Green
                                elif prob <= 50:
                                    return 'background-color: #fff3e0; color: #f57c00; font-weight: bold'  # Orange
                                else:
                                    return 'background-color: #ffebee; color: #c62828; font-weight: bold'  # Red
                            except:
                                return ''

                        # Apply styling
                        styled_df = df_results.style.apply(highlight_leaks, axis=1).map(
                            style_leak_probability,
                            subset=['Leak Probability']
                        )

                        st.dataframe(
                            styled_df,
                            hide_index=True,
                            width='stretch'
                        )

                        # Detailed view in expander
                        with st.expander(f"📊 Detailed Physics Analysis - Cylinder {cyl_num}", expanded=False):
                            for valve in valves:
                                r = valve['result']

                                # Valve header
                                valve_color = '#c62828' if r.is_leak else '#2e7d32'
                                st.markdown(f"""
                                <div style='background-color: {"#ffebee" if r.is_leak else "#e8f5e9"};
                                            padding: 0.75rem; border-radius: 6px; margin-bottom: 1rem;
                                            border-left: 4px solid {valve_color};'>
                                    <h4 style='color: {valve_color}; margin: 0;'>
                                        {valve['valve_id']} ({valve['valve_position']})
                                    </h4>
                                </div>
                                """, unsafe_allow_html=True)

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.markdown("#### 🎯 Detection Result")
                                    st.metric("Leak Probability", f"{r.leak_probability:.1f}%",
                                             delta="⚠️ LEAK" if r.is_leak else "✓ Normal",
                                             delta_color="inverse" if r.is_leak else "normal")
                                    st.metric("Confidence Score", f"{r.confidence:.1%}")

                                with col2:
                                    st.markdown("#### 📈 Amplitude Statistics")
                                    st.metric("Mean Amplitude", f"{r.feature_values['mean_amplitude']:.2f} G")
                                    st.metric("Median Amplitude", f"{r.feature_values['median_amplitude']:.2f} G")
                                    st.metric("Max Amplitude", f"{r.feature_values['max_amplitude']:.2f} G")

                                # Physics explanation
                                st.markdown("#### 🔬 Physics Explanation")
                                st.markdown(f'<div class="physics-explanation">{r.explanation}</div>',
                                           unsafe_allow_html=True)

                                # Waveform visualization
                                st.markdown("#### 📉 Waveform Pattern Analysis")
                                fig = create_waveform_plot(
                                    amplitudes=df_curves[valve['column']].values,
                                    crank_angles=df_curves['Crank Angle'].values,
                                    is_leak=r.is_leak,
                                    mean_amp=r.feature_values['mean_amplitude'],
                                    valve_name=valve['valve_id']
                                )
                                st.plotly_chart(fig, width='stretch')

                                st.markdown("---")

                        st.markdown("")  # Spacing

                    st.markdown("---")

                    # Summary Visualization
                    st.subheader("Leak Probability Summary")

                    cyl_summary = []
                    for cyl_num in sorted(cylinders.keys()):
                        if cyl_num == 0:
                            continue
                        status = cylinder_status[cyl_num]
                        cyl_summary.append({
                            'Cylinder': f"Cyl {cyl_num}",
                            'Max Leak Probability': status['max_leak_prob'],
                            'Status': 'LEAK' if status['has_leak'] else 'Normal'
                        })

                    if cyl_summary:
                        df_summary = pd.DataFrame(cyl_summary)

                        fig = go.Figure()

                        colors = ['#c62828' if row['Status'] == 'LEAK' else '#2e7d32'
                                 for _, row in df_summary.iterrows()]

                        fig.add_trace(go.Bar(
                            x=df_summary['Cylinder'],
                            y=df_summary['Max Leak Probability'],
                            marker_color=colors,
                            text=df_summary['Max Leak Probability'].apply(lambda x: f"{x:.1f}%"),
                            textposition='outside'
                        ))

                        fig.add_hline(
                            y=50,
                            line_dash="dash",
                            line_color="orange",
                            annotation_text="50% Threshold",
                            annotation_position="right"
                        )

                        fig.update_layout(
                            title="Maximum Leak Probability by Cylinder",
                            xaxis_title="Cylinder",
                            yaxis_title="Leak Probability (%)",
                            yaxis_range=[0, 105],
                            height=400,
                            showlegend=False
                        )

                        st.plotly_chart(fig, width='stretch')

                    st.info("Each cylinder bar shows the HIGHEST leak probability among its valves.")

                    # Technical details
                    with st.expander("Technical Details - AI Pattern Recognition"):
                        st.markdown("""
                        **Detection Method: AI Waveform Analysis**

                        The AI analyzes ultrasonic acoustic emission (AE) sensor data to detect valve leaks.

                        **Why This Works:**

                        Ultrasonic AE sensors (36-44 KHz) detect high-frequency acoustic emissions:
                        - **Normal valve closure**: Creates brief mechanical impact → discrete amplitude spikes
                        - **Leaking valve**: Gas escaping through seat creates continuous turbulent flow → sustained "smear" pattern

                        **AI Detection Thresholds:**
                        | Metric | Normal | Leak Indicator |
                        |--------|--------|----------------|
                        | Mean amplitude | < 2G | > 3G |
                        | % above 1G | < 70% | > 85% |
                        | % above 2G | < 20% | > 50% |

                        **AI Scoring Algorithm:**

                        Leak probability is calculated using weighted criteria:
                        - Mean amplitude level (35% weight)
                        - Sustained elevation above 1G (25% weight)
                        - Sustained elevation above 2G (25% weight)
                        - High activity above 5G (15% weight)

                        **Classification:**
                        - Leak probability ≥ 50% → LEAK detected
                        - Leak probability < 50% → Normal operation

                        **Confidence Score:**

                        Indicates certainty of the AI classification based on how far the measurements are from the 50% threshold.
                        """)

                    # Pressure Curve Analysis Section (WRPM files only)
                    if df_pressure is not None and len(df_pressure) > 0:
                        st.markdown("---")
                        st.markdown('<div class="section-header">📊 Pressure Curve Analysis (PVPT)</div>', unsafe_allow_html=True)
                        st.markdown("""
                        **Valve Timing Reference:**
                        - **TDC (Top Dead Center):** 0° and 360° - Piston at top of stroke
                        - **BDC (Bottom Dead Center):** 180° - Piston at bottom of stroke

                        Pressure curves help verify valve timing and can indicate valve problems when events occur at unexpected crank angles.
                        """)

                        # Get pressure curve columns (all columns except 'Crank Angle')
                        pressure_cols = [col for col in df_pressure.columns if col != 'Crank Angle']

                        if pressure_cols:
                            # Create expandable section for each pressure curve
                            with st.expander(f"View Pressure Curves ({len(pressure_cols)} channels)", expanded=True):
                                for col in pressure_cols:
                                    fig = create_pressure_plot(
                                        pressure_data=df_pressure[col].values,
                                        crank_angles=df_pressure['Crank Angle'].values,
                                        channel_name=col
                                    )
                                    st.plotly_chart(fig, width='stretch')
                        else:
                            st.warning("Pressure data found but no channels could be extracted.")

    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        st.info("Please check that your file is a valid Windrock XML or WRPM file.")

else:
    # No file uploaded yet
    st.info("Upload a Windrock XML or WRPM file to begin analysis")

    st.markdown("---")
    st.subheader("Supported File Formats")
    st.markdown("""
    **XML Files (Curves.xml)**
    - Exported from Windrock diagnostic software
    - Contains ultrasonic waveform data for all cylinders

    **WRPM Files**
    - Native Windrock portable recording format
    - Contains AE sensor data and machine metadata
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p><strong>By Autoflow Solutions</strong> | v2.2.1</p>
    <p>AI-Powered Valve Leak Detection | Intelligent Pattern Recognition</p>
</div>
""", unsafe_allow_html=True)

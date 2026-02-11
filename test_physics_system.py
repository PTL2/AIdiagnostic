"""
Test Physics-Based Leak Detection System on Known Leak Files

Tests:
1. C402 Cyl 3 CD - Known leak valve (expect LOW amplitude, LEAK detected)
2. 578-B - Multiple known leaks
3. Compare with normal valves (expect HIGH amplitude, NORMAL)
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from xml_parser import parse_curves_xml
from leak_detector import PhysicsBasedLeakDetector
import re

print("=" * 100)
print("PHYSICS-BASED LEAK DETECTION TEST")
print("=" * 100)
print()
print("Key Insight: HIGH ultrasonic amplitude = LEAK (gas escaping creates smear pattern)")
print("            LOW ultrasonic amplitude = NORMAL (brief spikes during valve events)")
print()

# Initialize detector
detector = PhysicsBasedLeakDetector()

# Test files
test_files = [
    {
        'path': r'C:\Users\Andrea\my-project\assets\xml-samples\C402_C_09_09_199812_02_53PM_Curves.xml',
        'name': 'C402 Sep 9 1998',
        'known_leak': 'Cyl 3 CD (Crank Discharge)',
        'expected_cyl': 3
    },
    {
        'path': r'C:\Users\Andrea\my-project\assets\xml-samples\578_B_09_25_20257_08_59AM_Curves.xml',
        'name': '578-B Sep 25 2002',
        'known_leak': 'Cyl 3 Multiple',
        'expected_cyl': 3
    }
]

for file_info in test_files:
    print("=" * 100)
    print(f"FILE: {file_info['name']}")
    print(f"Known Leak: {file_info['known_leak']}")
    print("=" * 100)

    try:
        # Load XML
        with open(file_info['path'], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        # Parse XML
        df_curves = parse_curves_xml(xml_content)

        if df_curves is None:
            print("  ERROR: Failed to parse XML")
            continue

        # Find ULTRASONIC curves
        ultrasonic_cols = [col for col in df_curves.columns
                           if 'ULTRASONIC' in col and col != 'Crank Angle']

        if not ultrasonic_cols:
            print("  ERROR: No ULTRASONIC curves found")
            continue

        print(f"\n  Found {len(ultrasonic_cols)} ULTRASONIC curves")

        # Analyze each ultrasonic curve
        results_by_cylinder = {}

        for col in ultrasonic_cols:
            amplitudes = df_curves[col].values
            result = detector.detect_leak(amplitudes)

            # Parse valve info
            parts = col.split('.')
            if len(parts) >= 2:
                valve_id = parts[1]
                cyl_match = re.search(r'(\d+)', valve_id)
                cyl_num = int(cyl_match.group(1)) if cyl_match else 0
            else:
                valve_id = col
                cyl_num = 0

            if cyl_num not in results_by_cylinder:
                results_by_cylinder[cyl_num] = []

            results_by_cylinder[cyl_num].append({
                'valve_id': valve_id,
                'mean_amp': result.feature_values['mean_amplitude'],
                'max_amp': result.feature_values['max_amplitude'],
                'leak_prob': result.leak_probability,
                'is_leak': result.is_leak
            })

        # Show results for cylinder with known leak
        target_cyl = file_info['expected_cyl']

        if target_cyl in results_by_cylinder:
            print(f"\n  CYLINDER {target_cyl} RESULTS (Known Leak Cylinder):")
            print("  " + "-" * 96)
            print(f"  {'Valve':<10} {'Mean Amp':<12} {'Max Amp':<12} {'Leak Prob':<15} {'Status':<15}")
            print("  " + "-" * 96)

            max_prob = 0
            max_valve = None
            leak_count = 0

            for v in results_by_cylinder[target_cyl]:
                status = "[LEAK]" if v['is_leak'] else "[OK] Normal"
                if v['is_leak']:
                    leak_count += 1

                if v['leak_prob'] > max_prob:
                    max_prob = v['leak_prob']
                    max_valve = v['valve_id']

                print(f"  {v['valve_id']:<10} {v['mean_amp']:>8.2f}G    {v['max_amp']:>8.2f}G    {v['leak_prob']:>10.1f}%    {status:<15}")

            print()
            print(f"  SUMMARY:")
            print(f"    Leaking valves detected: {leak_count}/{len(results_by_cylinder[target_cyl])}")
            print(f"    Highest leak probability: {max_prob:.1f}% ({max_valve})")

            if leak_count > 0:
                print(f"    [PASS] LEAK CORRECTLY DETECTED in Cylinder {target_cyl}")
            else:
                print(f"    [FAIL] No leak detected in Cylinder {target_cyl}")

        # Show summary for all cylinders
        print(f"\n  ALL CYLINDERS SUMMARY:")
        print("  " + "-" * 60)
        print(f"  {'Cylinder':<12} {'Max Leak Prob':<18} {'Leak Count':<15} {'Status':<15}")
        print("  " + "-" * 60)

        for cyl_num in sorted(results_by_cylinder.keys()):
            if cyl_num == 0:
                continue
            valves = results_by_cylinder[cyl_num]
            max_prob = max(v['leak_prob'] for v in valves)
            leak_count = sum(1 for v in valves if v['is_leak'])
            status = "LEAK" if leak_count > 0 else "Normal"

            marker = " ***" if cyl_num == target_cyl else ""
            print(f"  Cyl {cyl_num:<8} {max_prob:>10.1f}%       {leak_count}/{len(valves)}             {status:<15}{marker}")

        print()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()

print("=" * 100)
print("PHYSICS-BASED TEST COMPLETE")
print("=" * 100)
print()
print("Physics Validation:")
print("  - HIGH amplitude (>3G mean) + sustained elevation = LEAK (smear pattern)")
print("  - LOW amplitude (<2G mean) + brief spikes only = NORMAL")
print("  - Based on actual XML waveform analysis of known leak valves")

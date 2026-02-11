# key_files_analysis.py
"""
Analyze the structure of key files for refactoring planning
"""

def analyze_file_structure(filename):
    """Analyze a Python file's structure."""
    print(f"\n🔍 Analyzing: {filename}")
    print("-" * 40)
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        
        # Count functions and classes
        functions = 0
        classes = 0
        imports = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def '):
                functions += 1
                if functions <= 3:  # Show first 3 functions
                    print(f"  Function: {stripped[4:].split('(')[0]}")
            elif stripped.startswith('class '):
                classes += 1
                if classes <= 3:  # Show first 3 classes
                    print(f"  Class: {stripped[6:].split('(')[0]}")
            elif stripped.startswith('import ') or stripped.startswith('from '):
                imports += 1
        
        print(f"\nSummary: {classes} classes, {functions} functions, {imports} imports")
        
        # Check for hard-coded values
        hardcoded_indicators = ['path = "', "path = '", 'file = "', "file = '", '.xml"', ".xml'"]
        hardcoded_count = 0
        
        for i, line in enumerate(lines[:100], 1):  # Check first 100 lines
            for indicator in hardcoded_indicators:
                if indicator in line:
                    print(f"  Line {i}: Possible hard-coded value")
                    hardcoded_count += 1
                    break
        
        if hardcoded_count > 0:
            print(f"⚠️  Found {hardcoded_count} possible hard-coded values")

# Analyze key files
key_files = ['app.py', 'leak_detector.py', 'xml_parser.py', 'wrpm_parser_ae.py']

for file in key_files:
    try:
        analyze_file_structure(file)
    except FileNotFoundError:
        print(f"❌ File not found: {file}")
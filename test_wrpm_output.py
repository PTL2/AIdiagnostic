# test_wrpm_output.py
"""
Test WRPM parser output to understand what format it produces
"""

import wrpm_parser_ae
import pandas as pd

def analyze_wrpm_output(wrpm_file):
    """Analyze what the WRPM parser produces."""
    print(f"\n📊 Analyzing WRPM Parser Output: {wrpm_file}")
    print("=" * 50)
    
    try:
        # Use existing WRPM parser
        parser = wrpm_parser_ae.WrpmParserAE(wrpm_file)
        df = parser.parse_to_dataframe()
        
        print(f"✅ Successfully parsed")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Show column names
        print("\nColumn names:")
        for col in df.columns:
            print(f"  - {col}")
        
        # Show data sample
        print(f"\nFirst 5 rows of first column:")
        first_col = df.columns[0]
        print(df[first_col].head())
        
        # Check data types
        print(f"\nData types:")
        print(df.dtypes)
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# Test with available WRPM files
wrpm_files = ['Dwale - Unit 3C.wrpm', 'Station H - Unit 2 C.wrpm']

for file in wrpm_files:
    analyze_wrpm_output(file)
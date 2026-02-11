"""
Unified Data Loader for Both XML and WRPM Files

Provides a single interface to load valve data regardless of file format.
Automatically detects file type and returns data in consistent format.

Machine Type Detection:
- Compressors (unit names ending in C, like 2C, 3C): 360° crank angle
- Engines (unit names ending in E, like 2E): 720° crank angle, excluded from leak detection
"""

import pandas as pd
from typing import Dict, Tuple, Optional
from io import BytesIO, StringIO


def load_valve_data(uploaded_file) -> Tuple[pd.DataFrame, Dict, str]:
    """
    Load valve data from either XML or WRPM file.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Tuple of (curves_dataframe, metadata_dict, file_type)
        - curves_dataframe: DataFrame with 'Crank Angle' and curve columns
        - metadata_dict: Dict with machine info and curve metadata
        - file_type: 'XML' or 'WRPM'

    Raises:
        ValueError: If file type is not recognized or parsing fails
    """
    filename = uploaded_file.name.lower()

    if filename.endswith('.wrpm'):
        return load_wrpm_file(uploaded_file)
    elif filename.endswith('.xml'):
        return load_xml_file(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}. Please upload .xml or .wrpm file.")


def load_wrpm_file(uploaded_file) -> Tuple[pd.DataFrame, Dict, str]:
    """
    Load WRPM file and extract AE sensor data.

    Args:
        uploaded_file: Streamlit UploadedFile object with .wrpm file

    Returns:
        Tuple of (curves_dataframe, metadata_dict, 'WRPM')

    Note:
        Engine files (e.g., Unit 2E) will have 'is_engine': True in metadata
        and should be excluded from leak detection analysis.
    """
    from wrpm_parser_ae import WrpmParserAE

    try:
        # Create BytesIO object from uploaded file
        file_bytes = BytesIO(uploaded_file.read())

        # Parse WRPM file
        parser = WrpmParserAE(file_bytes)

        # Get metadata
        info = parser.get_curve_info()

        # Get DataFrame
        file_bytes.seek(0)  # Reset file pointer
        parser = WrpmParserAE(file_bytes)
        df_curves = parser.parse_to_dataframe()

        # Create metadata dict with machine type info
        metadata = {
            'machine_id': info.get('machine_id', 'Unknown'),
            'date': info.get('date'),
            'total_curves': info.get('total_curves', 0),
            'ae_curves': info.get('ae_curves', []),
            'data_points': info.get('data_points', 0),
            'crank_angle_range': info.get('crank_angle_range', '0-360°'),
            'file_type': 'WRPM',
            'machine_type': info.get('machine_type', 'compressor'),
            'is_engine': info.get('is_engine', False),
            'has_pressure_data': info.get('has_pressure_data', False)
        }

        return df_curves, metadata, 'WRPM'

    except Exception as e:
        raise ValueError(f"Failed to parse WRPM file: {str(e)}")


def load_xml_file(uploaded_file) -> Tuple[pd.DataFrame, Dict, str]:
    """
    Load XML file (Curves.xml format).

    Args:
        uploaded_file: Streamlit UploadedFile object with .xml file

    Returns:
        Tuple of (curves_dataframe, metadata_dict, 'XML')
    """
    from xml_parser import parse_curves_xml, get_curve_info

    try:
        # Read XML content
        xml_content = uploaded_file.read().decode('utf-8')

        # Get metadata
        info = get_curve_info(xml_content)

        # Parse curves
        df_curves = parse_curves_xml(xml_content)

        if df_curves is None or len(df_curves) == 0:
            raise ValueError("Failed to parse XML file - no curves found")

        # Create metadata dict
        metadata = {
            'machine_id': info.get('machine_id', 'Unknown'),
            'total_curves': info.get('total_curves', 0),
            'ae_curves': info.get('ae_curves', []),
            'data_points': info.get('data_points', 0),
            'crank_angle_range': info.get('crank_angle_range', '0-720°'),
            'file_type': 'XML'
        }

        return df_curves, metadata, 'XML'

    except Exception as e:
        raise ValueError(f"Failed to parse XML file: {str(e)}")


def get_ultrasonic_curves(df_curves: pd.DataFrame) -> list:
    """
    Extract ultrasonic/AE curve column names from DataFrame.

    Args:
        df_curves: DataFrame with curve data

    Returns:
        List of column names containing ultrasonic sensor data
    """
    ultrasonic_cols = [
        col for col in df_curves.columns
        if 'ULTRASONIC' in col and col != 'Crank Angle'
    ]
    return ultrasonic_cols


def parse_valve_id_from_column(column_name: str) -> str:
    """
    Extract valve ID from column name.

    Examples:
        'C402 - C.3CD1.ULTRASONIC G...' -> 'Cylinder 3 CD'
        'Dwale - Unit 3C - C.1AE.ULTRASONIC...' -> 'Cylinder 1 AE'

    Args:
        column_name: Full column name from DataFrame

    Returns:
        Simplified valve ID
    """
    import re

    # Try to find cylinder number and position
    # Pattern: C.{number}{position}
    match = re.search(r'C\.(\d+)([A-Z]{2,3})', column_name)
    if match:
        cyl_num = match.group(1)
        position = match.group(2)
        return f"Cylinder {cyl_num} {position}"

    # Pattern: C.{number}AE
    match = re.search(r'C\.(\d+)AE', column_name)
    if match:
        cyl_num = match.group(1)
        return f"Cylinder {cyl_num} AE"

    # Fallback: return column name
    return column_name.split('-')[-1].strip() if '-' in column_name else column_name


def load_wrpm_pressure_data(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load pressure (PVPT) curves from a WRPM file.

    Args:
        uploaded_file: Streamlit UploadedFile object with .wrpm file

    Returns:
        DataFrame with pressure curves, or None if no pressure data available
        Columns: 'Crank Angle' + pressure curve columns
    """
    from wrpm_parser_ae import WrpmParserAE

    try:
        # Create BytesIO object from uploaded file
        file_bytes = BytesIO(uploaded_file.read())

        # Parse WRPM file for pressure data
        parser = WrpmParserAE(file_bytes)
        df_pressure = parser.parse_pressure_to_dataframe()

        return df_pressure

    except Exception as e:
        return None

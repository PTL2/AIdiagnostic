"""
WRPM File Parser with AE Sensor Support for Leak Detection

Parses Windrock .wrpm files (ZIP archives) and extracts:
- Ultrasonic AE sensor data (.SDD files) - PRIMARY for leak detection
- Pressure data (.S$ files) - PVPT curves
- Vibration data (.V$ files) - Secondary
- Machine metadata and calibration

Machine Type Detection:
- Compressors (unit names ending in C, like 2C, 3C): 360° crank angle
- Engines (unit names ending in E, like 2E): 720° crank angle

Designed to work with the Physics-Based Leak Detector.
"""

import zipfile
import struct
import re
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, BinaryIO
from datetime import datetime
from collections import Counter
from io import BytesIO


class WrpmParserAE:
    """
    Enhanced WRPM Parser with Acoustic Emission (AE) sensor support.

    Usage:
        parser = WrpmParserAE(wrpm_file_or_path)
        curves_df = parser.parse_to_dataframe()
        # Returns DataFrame compatible with XML parser format

    Machine Type Detection:
        - Compressors (2C, 3C, etc.): 360° crank angle
        - Engines (2E, etc.): 720° crank angle
    """

    def __init__(self, wrpm_source):
        """
        Initialize parser.

        Args:
            wrpm_source: Path to .wrpm file (str/Path) or file-like object (BytesIO)
        """
        if isinstance(wrpm_source, (str, Path)):
            self.wrpm_path = Path(wrpm_source)
            self.file_obj = None
        else:
            # File-like object (e.g., from Streamlit uploader)
            self.file_obj = wrpm_source
            self.wrpm_path = None

        # Parsed data
        self.machine_id = None
        self.date = None
        self.session_name = None
        self.full_scale_psi = 2000.0  # Default
        self.full_scale_g = 10.0  # Default for AE sensors (G units)
        self.calibration_channels = []

        # Machine type detection
        self.machine_type = None  # 'compressor' or 'engine'
        self.crank_angle_range = 360  # Default for compressors
        self.is_engine = False  # Flag for excluding from leak detection

    def _get_zipfile(self):
        """Get zipfile object from source."""
        if self.file_obj:
            return zipfile.ZipFile(self.file_obj, 'r')
        else:
            return zipfile.ZipFile(self.wrpm_path, 'r')

    def parse_machine_id(self, z: zipfile.ZipFile) -> str:
        """Parse machine identifier from D6NAME3.DAT."""
        try:
            if 'D6NAME3.DAT' not in z.namelist():
                return "Unknown Machine"

            raw = z.read('D6NAME3.DAT')
            machine_id = raw.decode('ascii', errors='ignore').strip()
            machine_id = machine_id.replace('\r\n', ' - ').replace('\n', ' - ')

            self.machine_id = machine_id

            # Detect machine type from ID
            self._detect_machine_type()

            return machine_id

        except Exception as e:
            self.machine_id = "Unknown Machine"
            return self.machine_id

    def _detect_machine_type(self):
        """
        Detect if machine is a compressor or engine based on unit name.

        Naming convention:
        - Unit names ending in 'C' (e.g., '2C', '3C') = Compressor = 360° crank angle
        - Unit names ending in 'E' (e.g., '2E') = Engine = 720° crank angle
        """
        if not self.machine_id:
            self.machine_type = 'compressor'
            self.crank_angle_range = 360
            self.is_engine = False
            return

        machine_id_upper = self.machine_id.upper()

        # Check for engine indicators (unit ending in E)
        # Patterns: "Unit 2E", "Unit 2 E", "2E", etc.
        engine_patterns = [
            r'UNIT\s*\d+\s*E\b',  # "Unit 2E" or "Unit 2 E"
            r'\b\d+\s*E\b',       # "2E" or "2 E"
            r'[-\s]E\b',          # " - E" or " E" at end
        ]

        for pattern in engine_patterns:
            if re.search(pattern, machine_id_upper):
                self.machine_type = 'engine'
                self.crank_angle_range = 720
                self.is_engine = True
                return

        # Check for compressor indicators (unit ending in C)
        # Patterns: "Unit 2C", "Unit 3C", "2C", "3C", etc.
        compressor_patterns = [
            r'UNIT\s*\d+\s*C\b',  # "Unit 2C" or "Unit 2 C"
            r'\b\d+\s*C\b',       # "2C" or "3C"
            r'[-\s]C\b',          # " - C" or " C" at end
        ]

        for pattern in compressor_patterns:
            if re.search(pattern, machine_id_upper):
                self.machine_type = 'compressor'
                self.crank_angle_range = 360
                self.is_engine = False
                return

        # Default to compressor if no pattern matched
        self.machine_type = 'compressor'
        self.crank_angle_range = 360
        self.is_engine = False

    def parse_date_from_filename(self, z: zipfile.ZipFile) -> Optional[datetime]:
        """
        Extract date from waveform filename pattern: SYYMMDDD
        Example: S25S0903 → September 3, 2025
        """
        try:
            pattern = r'S(\d{2})S(\d{2})(\d{2})'

            for filename in z.namelist():
                match = re.search(pattern, filename)
                if match:
                    year = 2000 + int(match.group(1))
                    month = int(match.group(2))
                    day = int(match.group(3))

                    try:
                        self.date = datetime(year, month, day)
                        self.session_name = filename.split('.')[0]
                        return self.date
                    except ValueError:
                        continue

            return None

        except Exception as e:
            return None

    def extract_calibration(self, z: zipfile.ZipFile) -> Dict:
        """Extract full-scale calibration values from D6CALFAC.DAT."""
        calibration = {
            'full_scale_psi': 2000.0,
            'full_scale_g': 10.0,
            'channels': []
        }

        try:
            if 'D6CALFAC.DAT' not in z.namelist():
                return calibration

            raw = z.read('D6CALFAC.DAT')
            text = ''.join(chr(b) if 32 <= b < 127 else ' ' for b in raw)

            # Find all floating point numbers
            float_matches = re.findall(r'(\d+\.\d+)', text)

            full_scale_values = []
            for match in float_matches:
                value = float(match)
                if 100 <= value <= 10000:  # Typical PSI range
                    full_scale_values.append(value)
                elif 1 <= value <= 100:  # Typical G range for AE
                    self.full_scale_g = value

            if full_scale_values:
                value_counts = Counter(full_scale_values)
                most_common = value_counts.most_common(1)[0][0]
                calibration['full_scale_psi'] = most_common
                calibration['channels'] = full_scale_values
                self.full_scale_psi = most_common
                self.calibration_channels = full_scale_values

        except Exception as e:
            pass

        return calibration

    def find_waveform_files(self, z: zipfile.ZipFile) -> Dict[str, str]:
        """
        Find all waveform data files including AE sensors.

        Returns:
            Dict with 'pressure', 'vibration', 'ae_primary', 'ae_secondary' file paths
        """
        if not self.session_name:
            session_files = [f for f in z.namelist() if re.match(r'S\d{2}S\d{4}', f)]
            if session_files:
                self.session_name = session_files[0].split('.')[0]

        if not self.session_name:
            raise FileNotFoundError("No session files found (pattern: SYYMMDDD)")

        base_name = self.session_name
        files = z.namelist()

        result = {
            'pressure': None,
            'vibration': None,
            'ae_primary': None,  # .SDD - PRIMARY AE waveform data (has real varied values)
            'ae_secondary': None  # .S&& - Secondary (often contains trigger/timing data, not waveforms)
        }

        # Find AE sensor files (PRIORITY for leak detection)
        # NOTE: SDD files contain actual waveform data, S&& often has timing/trigger data
        for candidate in [f'{base_name}.SDD', f'{base_name}.SD']:
            if candidate in files:
                result['ae_primary'] = candidate
                break

        for candidate in [f'{base_name}.S&&', f'{base_name}.S&']:
            if candidate in files:
                result['ae_secondary'] = candidate
                break

        # Find pressure waveform
        for candidate in [f'{base_name}.S$$', f'{base_name}.S$', f'{base_name}.S##']:
            if candidate in files:
                result['pressure'] = candidate
                break

        # Find vibration waveform
        for candidate in [f'{base_name}.V$$', f'{base_name}.V$', f'{base_name}.V##']:
            if candidate in files:
                result['vibration'] = candidate
                break

        return result

    def parse_waveform_int16(self, z: zipfile.ZipFile, filename: str) -> np.ndarray:
        """Parse waveform file as signed int16 array."""
        raw = z.read(filename)
        # Handle odd byte counts by truncating to even
        if len(raw) % 2 == 1:
            raw = raw[:-1]
        num_samples = len(raw) // 2
        ints = struct.unpack("<" + "h" * num_samples, raw[:num_samples*2])
        return np.array(ints, dtype=np.int16)

    def apply_calibration_g(self, raw_counts: np.ndarray, full_scale_g: float) -> np.ndarray:
        """
        Apply calibration to convert ADC counts to G units (for AE sensors).

        Formula: g = (raw_count / 32768.0) * full_scale_g
        """
        return (raw_counts.astype(np.float32) / 32768.0) * full_scale_g

    def apply_calibration_psi(self, raw_counts: np.ndarray, full_scale_psi: float) -> np.ndarray:
        """Apply calibration to convert ADC counts to PSI."""
        return (raw_counts.astype(np.float32) / 32768.0) * full_scale_psi

    def segment_multichannel_data(self, waveform: np.ndarray, num_channels: int = None,
                                     points_per_rev: int = None) -> Dict[int, np.ndarray]:
        """
        Segment multi-channel waveform data into individual channels.

        Args:
            waveform: Combined waveform array
            num_channels: Number of channels (auto-detect if None)
            points_per_rev: Points per revolution (auto-detect based on machine type if None)

        Returns:
            Dict mapping channel number to waveform array
        """
        total_samples = len(waveform)

        # Determine points per revolution based on machine type
        # Compressor (360°): 355 points
        # Engine (720°): 710 points (or auto-detect)
        if points_per_rev is None:
            if self.is_engine:
                points_per_rev = 710  # 720° engine
            else:
                points_per_rev = 355  # 360° compressor

        # Auto-detect number of channels
        if num_channels is None:
            # Try common channel counts
            for n_channels in [8, 9, 10, 6, 12, 4]:
                if total_samples % (n_channels * points_per_rev) == 0:
                    num_channels = n_channels
                    break

            # If no exact match, try with slightly different points_per_rev
            if num_channels is None:
                for ppr in [355, 356, 354, 710, 720]:
                    for n_channels in [8, 9, 10, 6, 12, 4]:
                        if total_samples % (n_channels * ppr) == 0:
                            num_channels = n_channels
                            points_per_rev = ppr
                            break
                    if num_channels:
                        break

            if num_channels is None:
                # Fallback: assume single channel
                return {1: waveform}

        # Segment data
        samples_per_channel = total_samples // num_channels
        channels = {}

        for i in range(num_channels):
            start_idx = i * samples_per_channel
            end_idx = (i + 1) * samples_per_channel
            channels[i + 1] = waveform[start_idx:end_idx]

        return channels

    def parse_to_dataframe(self) -> pd.DataFrame:
        """
        Parse WRPM file and return DataFrame compatible with XML parser format.

        Returns:
            DataFrame with columns:
            - 'Crank Angle': 0-360 degrees (compressor) or 0-720 degrees (engine)
            - Multiple curve columns named like 'Machine - Location.ULTRASONIC G 36KHZ - 44KHZ.ID'
        """
        with self._get_zipfile() as z:
            # Parse metadata (this also detects machine type)
            self.parse_machine_id(z)
            self.parse_date_from_filename(z)
            self.extract_calibration(z)

            # Find waveform files
            waveform_files = self.find_waveform_files(z)

            # Priority: Extract AE sensor data (best for leak detection)
            ae_data = None
            ae_file_used = None

            if waveform_files['ae_primary']:
                ae_raw = self.parse_waveform_int16(z, waveform_files['ae_primary'])
                ae_data = self.apply_calibration_g(ae_raw, self.full_scale_g)
                ae_file_used = waveform_files['ae_primary']
            elif waveform_files['ae_secondary']:
                ae_raw = self.parse_waveform_int16(z, waveform_files['ae_secondary'])
                ae_data = self.apply_calibration_g(ae_raw, self.full_scale_g)
                ae_file_used = waveform_files['ae_secondary']

            # Fallback: Use pressure data if no AE data
            if ae_data is None and waveform_files['pressure']:
                pressure_raw = self.parse_waveform_int16(z, waveform_files['pressure'])
                ae_data = self.apply_calibration_psi(pressure_raw, self.full_scale_psi)
                ae_file_used = waveform_files['pressure']

            if ae_data is None:
                raise ValueError("No usable waveform data found in WRPM file")

            # Segment multi-channel data
            channels = self.segment_multichannel_data(ae_data)

            # Create DataFrame with correct crank angle range based on machine type
            # Compressor: 360°, Engine: 720°
            if self.is_engine:
                points_per_rev = 710
                max_angle = 720
            else:
                points_per_rev = 355
                max_angle = 360  # Compressor uses 360°

            # Create crank angle column
            crank_angles = np.linspace(0, max_angle, points_per_rev)

            df_data = {'Crank Angle': crank_angles}

            # Add each channel as a column
            for ch_num, ch_data in channels.items():
                # Take first revolution
                if len(ch_data) >= points_per_rev:
                    ch_waveform = ch_data[:points_per_rev]
                else:
                    # Interpolate if needed
                    ch_waveform = np.interp(
                        crank_angles,
                        np.linspace(0, max_angle, len(ch_data)),
                        ch_data
                    )

                # Create column name similar to XML format
                machine_id = self.machine_id or "Unknown"
                col_name = f"{machine_id} - C.{ch_num}AE.ULTRASONIC G 36KHZ - 44KHZ (NARROW BAND).{ch_num}AE"
                df_data[col_name] = ch_waveform

            df = pd.DataFrame(df_data)
            return df

    def parse_pressure_to_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Parse WRPM file and extract PVPT (pressure) curves.

        Returns:
            DataFrame with columns:
            - 'Crank Angle': 0-360 degrees (compressor) or 0-720 degrees (engine)
            - Multiple pressure curve columns named like 'Machine - C.{X}P.PVPT.{X}P'
            Returns None if no pressure data available.
        """
        with self._get_zipfile() as z:
            # Parse metadata (this also detects machine type)
            self.parse_machine_id(z)
            self.parse_date_from_filename(z)
            self.extract_calibration(z)

            # Find waveform files
            waveform_files = self.find_waveform_files(z)

            # Check for pressure data
            if not waveform_files['pressure']:
                return None

            # Extract pressure data
            pressure_raw = self.parse_waveform_int16(z, waveform_files['pressure'])
            pressure_data = self.apply_calibration_psi(pressure_raw, self.full_scale_psi)

            # Segment multi-channel data
            channels = self.segment_multichannel_data(pressure_data)

            # Create DataFrame with correct crank angle range based on machine type
            if self.is_engine:
                points_per_rev = 710
                max_angle = 720
            else:
                points_per_rev = 355
                max_angle = 360

            # Create crank angle column
            crank_angles = np.linspace(0, max_angle, points_per_rev)

            df_data = {'Crank Angle': crank_angles}

            # Add each channel as a pressure column
            for ch_num, ch_data in channels.items():
                # Take first revolution
                if len(ch_data) >= points_per_rev:
                    ch_waveform = ch_data[:points_per_rev]
                else:
                    # Interpolate if needed
                    ch_waveform = np.interp(
                        crank_angles,
                        np.linspace(0, max_angle, len(ch_data)),
                        ch_data
                    )

                # Create column name for pressure curves (PVPT format)
                machine_id = self.machine_id or "Unknown"
                col_name = f"{machine_id} - C.{ch_num}P.PVPT (PRESSURE).{ch_num}P"
                df_data[col_name] = ch_waveform

            df = pd.DataFrame(df_data)
            return df

    def get_curve_info(self) -> Dict:
        """
        Get curve metadata (similar to XML parser's get_curve_info).

        Returns:
            Dict with file information including machine type and crank angle range
        """
        with self._get_zipfile() as z:
            self.parse_machine_id(z)
            self.parse_date_from_filename(z)
            waveform_files = self.find_waveform_files(z)

            # Count AE curves
            ae_curves = []
            if waveform_files['ae_primary']:
                ae_raw = self.parse_waveform_int16(z, waveform_files['ae_primary'])
                channels = self.segment_multichannel_data(ae_raw)
                ae_curves = list(channels.keys())

            # Determine points and angle range based on machine type
            if self.is_engine:
                data_points = 710
                angle_range = '0-720°'
            else:
                data_points = 355
                angle_range = '0-360°'

            return {
                'total_curves': len(ae_curves),
                'ae_curves': ae_curves,
                'data_points': data_points,
                'crank_angle_range': angle_range,
                'machine_id': self.machine_id,
                'date': self.date,
                'file_type': 'WRPM',
                'machine_type': self.machine_type,
                'is_engine': self.is_engine,
                'has_pressure_data': waveform_files['pressure'] is not None
            }


def parse_wrpm_to_dataframe(wrpm_source) -> pd.DataFrame:
    """
    Convenience function to parse WRPM file to DataFrame.

    Args:
        wrpm_source: Path to .wrpm file or file-like object

    Returns:
        DataFrame with crank angles and curve data
    """
    parser = WrpmParserAE(wrpm_source)
    return parser.parse_to_dataframe()


def get_wrpm_curve_info(wrpm_source) -> Dict:
    """
    Convenience function to get WRPM file metadata.

    Args:
        wrpm_source: Path to .wrpm file or file-like object

    Returns:
        Dict with file information
    """
    parser = WrpmParserAE(wrpm_source)
    return parser.get_curve_info()


def parse_wrpm_pressure_to_dataframe(wrpm_source) -> Optional[pd.DataFrame]:
    """
    Convenience function to parse WRPM file and extract PVPT pressure curves.

    Args:
        wrpm_source: Path to .wrpm file or file-like object

    Returns:
        DataFrame with crank angles and pressure curve data, or None if no pressure data
    """
    parser = WrpmParserAE(wrpm_source)
    return parser.parse_pressure_to_dataframe()


def is_engine_file(wrpm_source) -> bool:
    """
    Check if a WRPM file is from an engine (vs compressor).

    Engine files should be excluded from valve leak detection analysis.

    Args:
        wrpm_source: Path to .wrpm file or file-like object

    Returns:
        True if file is from an engine (e.g., Unit 2E), False for compressors
    """
    parser = WrpmParserAE(wrpm_source)
    info = parser.get_curve_info()
    return info.get('is_engine', False)

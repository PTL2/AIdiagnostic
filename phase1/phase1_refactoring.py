"""
Phase 1 Refactoring - Final Corrected Version

FIXES:
1. File adapter integration (for CLI compatibility) ✅
2. Correct attribute names for PhysicsBasedLeakDetector ✅
3. Handle LeakDetectionResult as object (not dict) ✅
"""

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Import the file adapter for CLI compatibility
from file_adapter import adapt_file_path

# Import existing system components
sys.path.insert(0, str(Path(__file__).parent.parent))
from unified_data_loader import load_valve_data
from leak_detector import PhysicsBasedLeakDetector


@dataclass
class AnalysisResult:
    """
    Structured result from valve analysis.
    
    Attributes:
        success: Whether analysis completed successfully
        file_name: Name of the analyzed file
        file_type: Type of file (WRPM or XML)
        leak_detected: Overall leak detection result
        leak_probability: Overall leak probability (0-100%)
        mean_amplitude: Mean amplitude across all cylinders
        cylinder_results: List of per-cylinder results
        error_message: Error message if analysis failed
    """
    success: bool
    file_name: str
    file_type: str = ""
    leak_detected: bool = False
    leak_probability: float = 0.0
    mean_amplitude: float = 0.0
    cylinder_results: List[Dict[str, Any]] = None
    error_message: str = ""
    
    def __post_init__(self):
        if self.cylinder_results is None:
            self.cylinder_results = []


class ValveAnalysisEngine:
    """
    Business logic layer for valve leak detection.
    
    Separates core analysis logic from UI concerns, enabling:
    - Reuse in CLI tools
    - Reuse in API endpoints
    - Easier testing
    - UI independence
    
    Usage:
        # From CLI:
        engine = ValveAnalysisEngine()
        result = engine.analyze_file("path/to/file.wrpm")
        
        # From UI (Streamlit):
        engine = ValveAnalysisEngine()
        result = engine.analyze_uploaded_file(uploaded_file)
    """
    
    def __init__(self):
        """Initialize the analysis engine with default detector."""
        # Use existing physics-based detector (unchanged from original system)
        self.detector = PhysicsBasedLeakDetector()
        print("✅ ValveAnalysisEngine initialized - Ready for analysis")
        print(f"   Using default thresholds")
    
    def analyze_file(self, file_path: str) -> AnalysisResult:
        """
        Analyze a valve data file from disk (for CLI/batch processing).
        
        This method handles file paths (strings) and adapts them to work
        with unified_data_loader which expects UploadedFile objects.
        
        Args:
            file_path: Path to XML or WRPM file
            
        Returns:
            AnalysisResult: Structured analysis results
            
        Example:
            >>> engine = ValveAnalysisEngine()
            >>> result = engine.analyze_file("data/sample.wrpm")
            >>> if result.success:
            >>>     print(f"Leak detected: {result.leak_detected}")
        """
        try:
            print(f"📥 Loading data from: {Path(file_path).name}")
            
            # KEY FIX: Adapt file path to UploadedFile interface
            adapted_file = adapt_file_path(file_path)
            
            # Now unified_data_loader works because adapted_file has .name property
            df_curves, metadata, file_type = load_valve_data(adapted_file)
            
            if df_curves is None or df_curves.empty:
                return AnalysisResult(
                    success=False,
                    file_name=Path(file_path).name,
                    error_message="No valid data found in file"
                )
            
            # Run analysis using existing detector (unchanged)
            cylinder_results = []
            all_leak_probabilities = []
            all_amplitudes = []
            
            # Process each curve (cylinder)
            for col in df_curves.columns:
                if col == 'Crank Angle':
                    continue
                
                amplitudes = df_curves[col].values
                
                # Use existing physics-based detector
                # FIX: result is a LeakDetectionResult object, not a dict
                result = self.detector.detect_leak(amplitudes)
                
                # Access as object attributes, not dict keys
                cylinder_results.append({
                    'cylinder': col,
                    'is_leak': result.is_leak,  # ✅ Object attribute
                    'leak_probability': result.leak_probability,  # ✅ Object attribute
                    'confidence': result.confidence,  # ✅ Object attribute
                    'mean_amplitude': result.mean_amplitude  # ✅ Object attribute
                })
                
                all_leak_probabilities.append(result.leak_probability)
                all_amplitudes.append(result.mean_amplitude)
            
            # Calculate overall results
            import numpy as np
            overall_leak_probability = np.mean(all_leak_probabilities)
            overall_mean_amplitude = np.mean(all_amplitudes)
            overall_leak_detected = any(r['is_leak'] for r in cylinder_results)
            
            return AnalysisResult(
                success=True,
                file_name=Path(file_path).name,
                file_type=file_type,
                leak_detected=overall_leak_detected,
                leak_probability=overall_leak_probability,
                mean_amplitude=overall_mean_amplitude,
                cylinder_results=cylinder_results
            )
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"❌ Analysis failed: {str(e)}")
            print(error_details)
            
            return AnalysisResult(
                success=False,
                file_name=Path(file_path).name if file_path else "unknown",
                error_message=f"Analysis failed: {str(e)}"
            )
    
    def analyze_uploaded_file(self, uploaded_file) -> AnalysisResult:
        """
        Analyze an uploaded file from Streamlit UI.
        
        This method works directly with Streamlit UploadedFile objects,
        so no adapter is needed.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            AnalysisResult: Structured analysis results
            
        Example (in Streamlit app):
            >>> uploaded_file = st.file_uploader("Upload file")
            >>> if uploaded_file:
            >>>     engine = ValveAnalysisEngine()
            >>>     result = engine.analyze_uploaded_file(uploaded_file)
        """
        try:
            # For UI uploads, pass directly (already has .name attribute)
            df_curves, metadata, file_type = load_valve_data(uploaded_file)
            
            if df_curves is None or df_curves.empty:
                return AnalysisResult(
                    success=False,
                    file_name=uploaded_file.name,
                    error_message="No valid data found in file"
                )
            
            # Same analysis logic as analyze_file()
            cylinder_results = []
            all_leak_probabilities = []
            all_amplitudes = []
            
            for col in df_curves.columns:
                if col == 'Crank Angle':
                    continue
                
                amplitudes = df_curves[col].values
                result = self.detector.detect_leak(amplitudes)
                
                # Access as object attributes, not dict keys
                cylinder_results.append({
                    'cylinder': col,
                    'is_leak': result.is_leak,
                    'leak_probability': result.leak_probability,
                    'confidence': result.confidence,
                    'mean_amplitude': result.mean_amplitude
                })
                
                all_leak_probabilities.append(result.leak_probability)
                all_amplitudes.append(result.mean_amplitude)
            
            import numpy as np
            return AnalysisResult(
                success=True,
                file_name=uploaded_file.name,
                file_type=file_type,
                leak_detected=any(r['is_leak'] for r in cylinder_results),
                leak_probability=np.mean(all_leak_probabilities),
                mean_amplitude=np.mean(all_amplitudes),
                cylinder_results=cylinder_results
            )
            
        except Exception as e:
            return AnalysisResult(
                success=False,
                file_name=uploaded_file.name,
                error_message=f"Analysis failed: {str(e)}"
            )
    
    def get_waveform_data(self, file_path_or_uploaded: Any) -> Optional[Any]:
        """
        Get waveform data for visualization.
        
        Args:
            file_path_or_uploaded: Either file path string or UploadedFile
            
        Returns:
            DataFrame with waveform data or None
        """
        try:
            # Detect if it's a string path or UploadedFile
            if isinstance(file_path_or_uploaded, str):
                adapted = adapt_file_path(file_path_or_uploaded)
                df_curves, _, _ = load_valve_data(adapted)
            else:
                df_curves, _, _ = load_valve_data(file_path_or_uploaded)
            
            return df_curves
            
        except Exception as e:
            print(f"❌ Failed to get waveform data: {e}")
            return None


# Quick test when run directly
if __name__ == "__main__":
    print("🧪 Testing ValveAnalysisEngine with file_adapter...")
    print("=" * 60)
    
    # Test that we can import everything
    print("\n✅ Imports successful:")
    print(f"   - FilePathAdapter from file_adapter")
    print(f"   - load_valve_data from unified_data_loader")
    print(f"   - PhysicsBasedLeakDetector from leak_detector")
    
    # Create engine
    print("\n🔧 Creating ValveAnalysisEngine...")
    engine = ValveAnalysisEngine()
    
    # Look for test files
    print("\n🔍 Looking for test files...")
    data_dir = Path(__file__).parent.parent / "data"
    
    test_files = []
    if (data_dir / "wrpm-samples").exists():
        test_files.extend(list((data_dir / "wrpm-samples").glob("*.wrpm")))
    if (data_dir / "xml-samples").exists():
        test_files.extend(list((data_dir / "xml-samples").glob("*.xml")))
    
    if not test_files:
        print("   ⚠️  No test files found")
        print(f"   Create data directory with sample files: {data_dir}")
    else:
        print(f"   Found {len(test_files)} test files")
        
        # Test with first file
        test_file = test_files[0]
        print(f"\n📝 Testing with: {test_file.name}")
        
        result = engine.analyze_file(str(test_file))
        
        if result.success:
            print(f"\n✅ Analysis successful!")
            print(f"   File type: {result.file_type}")
            print(f"   Leak detected: {result.leak_detected}")
            print(f"   Leak probability: {result.leak_probability:.1f}%")
            print(f"   Mean amplitude: {result.mean_amplitude:.2f}G")
            print(f"   Cylinders analyzed: {len(result.cylinder_results)}")
            
            # Show details for each cylinder
            print(f"\n   Cylinder Details:")
            for cyl in result.cylinder_results:
                status = "🔴 LEAK" if cyl['is_leak'] else "🟢 NORMAL"
                print(f"      {status} - {cyl['cylinder'][:40]}...")
                print(f"         Probability: {cyl['leak_probability']:.1f}%")
                print(f"         Mean Amplitude: {cyl['mean_amplitude']:.2f}G")
        else:
            print(f"\n❌ Analysis failed: {result.error_message}")
    
    print("\n" + "=" * 60)
    print("🎉 ValveAnalysisEngine ready for use!")
    print("=" * 60)
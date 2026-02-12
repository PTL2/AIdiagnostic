# run_analysis.py - COMPLETE CLI IMPLEMENTATION
"""
CLI for batch processing XML/WRPM files
Phase 1: Complete implementation with actual file processing
"""

import argparse
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(__file__)) if __file__ else os.getcwd()
sys.path.insert(0, parent_dir)

# Import our engine
try:
    from phase1.phase1_refactoring import ValveAnalysisEngine, AnalysisResult
    print("✅ Successfully imported ValveAnalysisEngine")
except ImportError as e:
    print(f"⚠️  Import from phase1 subdirectory failed: {e}")
    print("💡 Trying direct import...")
    sys.path.insert(0, os.path.dirname(__file__))
    from phase1_refactoring import ValveAnalysisEngine, AnalysisResult

def discover_files(input_path: str, file_format: str = 'auto') -> List[Path]:
    """
    Discover XML and WRPM files in input path.
    """
    path = Path(input_path)
    files = []
    
    if not path.exists():
        print(f"❌ Error: Path does not exist: {input_path}")
        return files
    
    if path.is_file():
        # Single file
        suffix = path.suffix.lower()[1:]  # Remove dot
        if file_format == 'auto' or suffix == file_format:
            files.append(path)
            print(f"📄 Single file: {path.name}")
    elif path.is_dir():
        # Directory - find matching files
        print(f"📁 Searching directory: {input_path}")
        
        if file_format in ['xml', 'auto']:
            xml_files = list(path.glob('**/*.xml'))
            files.extend(xml_files)
            if xml_files:
                print(f"   Found {len(xml_files)} XML files")
        
        if file_format in ['wrpm', 'auto']:
            wrpm_files = list(path.glob('**/*.wrpm'))
            files.extend(wrpm_files)
            if wrpm_files:
                print(f"   Found {len(wrpm_files)} WRPM files")
    
    # Remove duplicates and sort
    files = sorted(list(set(files)))
    return files

def process_single_file(file_path: Path, engine: ValveAnalysisEngine, verbose: bool = False) -> Dict:
    """
    Process a single file using our business logic engine.
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n  🔍 Processing: {file_path.name}")
    
    try:
        # Analyze the file
        result = engine.analyze_file(str(file_path))
        
        processing_time = time.time() - start_time
        
        if result.success:
            output = {
                'success': True,
                'file': file_path.name,
                'file_path': str(file_path),
                'file_type': result.file_type,
                'leak_detected': result.leak_detected,
                'leak_probability': round(result.leak_probability, 2),
                'mean_amplitude': round(result.mean_amplitude, 3),
                'cylinders_analyzed': len(result.cylinder_results),
                'cylinders_with_leak': sum(1 for c in result.cylinder_results if c['is_leak']),
                'processing_time_seconds': round(processing_time, 2),
                'analysis_time': datetime.now().isoformat(),
                'error': None
            }
            
            if verbose:
                status = "🔴 LEAK" if result.leak_detected else "🟢 NORMAL"
                print(f"    {status} - {result.leak_probability:.1f}% probability")
                print(f"    📊 {len(result.cylinder_results)} cylinders, {output['cylinders_with_leak']} with leaks")
                print(f"    ⏱️  Processed in {processing_time:.2f}s")
            
            return output
            
        else:
            output = {
                'success': False,
                'file': file_path.name,
                'file_path': str(file_path),
                'error': result.error_message,
                'processing_time_seconds': round(processing_time, 2)
            }
            
            if verbose:
                print(f"    ❌ Failed: {result.error_message}")
            
            return output
            
    except Exception as e:
        processing_time = time.time() - start_time
        error_msg = str(e)
        
        output = {
            'success': False,
            'file': file_path.name,
            'file_path': str(file_path),
            'error': error_msg,
            'processing_time_seconds': round(processing_time, 2)
        }
        
        if verbose:
            print(f"    💥 Exception: {error_msg[:100]}...")
        
        return output

def save_results(results: List[Dict], output_dir: Path, output_format: str):
    """
    Save results in CSV and/or JSON format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print(f"\n💾 Saving results to: {output_dir}")
    
    if output_format in ['csv', 'both'] and successful_results:
        try:
            import pandas as pd
            csv_path = output_dir / f'analysis_results_{timestamp}.csv'
            
            # Create DataFrame from successful results
            df = pd.DataFrame(successful_results)
            
            # Reorder columns for readability
            preferred_order = ['file', 'file_type', 'leak_detected', 'leak_probability', 
                             'mean_amplitude', 'cylinders_analyzed', 'cylinders_with_leak',
                             'processing_time_seconds', 'analysis_time', 'file_path']
            
            # Keep only columns that exist
            existing_cols = [col for col in preferred_order if col in df.columns]
            other_cols = [col for col in df.columns if col not in preferred_order]
            df = df[existing_cols + other_cols]
            
            df.to_csv(csv_path, index=False)
            print(f"  ✅ CSV saved: {csv_path.name} ({len(successful_results)} records)")
            
        except Exception as e:
            print(f"  ❌ Failed to save CSV: {e}")
    
    if output_format in ['json', 'both']:
        try:
            json_path = output_dir / f'analysis_results_{timestamp}.json'
            
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'total_files_processed': len(results),
                'successful_analysis': len(successful_results),
                'failed_analysis': len(failed_results),
                'files_with_leaks': sum(1 for r in successful_results if r['leak_detected']),
                'processing_summary': {
                    'total_processing_time': sum(r.get('processing_time_seconds', 0) for r in results),
                    'average_processing_time': sum(r.get('processing_time_seconds', 0) for r in results) / len(results) if results else 0
                },
                'results': results
            }
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
            
            print(f"  ✅ JSON saved: {json_path.name}")
            
        except Exception as e:
            print(f"  ❌ Failed to save JSON: {e}")
    
    # Save failed files list if any
    if failed_results:
        failed_path = output_dir / f'failed_files_{timestamp}.txt'
        with open(failed_path, 'w') as f:
            f.write("Failed Files Report\n")
            f.write("=" * 50 + "\n\n")
            for r in failed_results:
                f.write(f"File: {r['file']}\n")
                f.write(f"Error: {r['error']}\n")
                f.write("-" * 30 + "\n")
        print(f"  📝 Failed files list: {failed_path.name}")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='🔧 Batch valve leak detection analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py data/wrpm-samples/ output/              # Auto-detect format
  python run_analysis.py single_file.wrpm output/ --format wrpm  # Single file
  python run_analysis.py folder/ output/ --output-format json    # JSON only
  python run_analysis.py input/ output/ --verbose                # Detailed output

File Support:
  • XML files (.xml) - Windrock Curves.xml format
  • WRPM files (.wrpm) - Windrock binary format
        """
    )
    
    parser.add_argument('input_path', help='Input file or folder path')
    parser.add_argument('output_path', help='Output folder path')
    parser.add_argument('--format', choices=['xml', 'wrpm', 'auto'], 
                       default='auto', help='Input file format (default: auto-detect)')
    parser.add_argument('--output-format', choices=['csv', 'json', 'both'],
                       default='both', help='Output format (default: both)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed processing information')
    
    args = parser.parse_args()
    
    print("🔧 Valve Leak Analysis CLI - Phase 1")
    print("=" * 60)
    
    # Initialize engine
    print("🚀 Initializing analysis engine...")
    engine = ValveAnalysisEngine()
    
    # Discover files
    print(f"\n📁 Discovering files in: {args.input_path}")
    files = discover_files(args.input_path, args.format)
    
    if not files:
        print(f"❌ No {args.format} files found in: {args.input_path}")
        print("💡 Check that:")
        print("   1. The path exists")
        print("   2. Files have .xml or .wrpm extension")
        print("   3. You have read permissions")
        return 1
    
    print(f"🎯 Found {len(files)} files to process")
    
    # Process each file
    print(f"\n⚙️  Starting analysis...")
    start_time = time.time()
    
    results = []
    for i, file_path in enumerate(files, 1):
        if args.verbose:
            print(f"\n[{i}/{len(files)}] ", end='')
        else:
            print(f"  {i:3d}/{len(files)}: {file_path.name[:40]:40s}", end='')
        
        result = process_single_file(file_path, engine, args.verbose)
        results.append(result)
        
        if not args.verbose:
            if result['success']:
                prob = result.get('leak_probability', 0)
                status = "🔴" if result.get('leak_detected', False) else "🟢"
                print(f" {status} {prob:5.1f}%")
            else:
                print(f" ❌ FAILED")
    
    total_time = time.time() - start_time
    
    # Save results
    output_dir = Path(args.output_path)
    save_results(results, output_dir, args.output_format)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    leaks_found = sum(1 for r in results if r.get('leak_detected', False))
    
    print(f"\n📊 Analysis Complete")
    print("=" * 30)
    print(f"  Total files processed: {len(files)}")
    print(f"  Successfully analyzed: {successful}")
    print(f"  Failed analysis: {len(files) - successful}")
    print(f"  Files with leaks detected: {leaks_found}")
    print(f"  Total processing time: {total_time:.1f}s")
    print(f"  Average per file: {total_time/len(files):.1f}s" if files else "N/A")
    print(f"  Output saved to: {output_dir.absolute()}")
    
    if successful == 0:
        print(f"\n⚠️  Warning: No files were successfully analyzed")
        return 1
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
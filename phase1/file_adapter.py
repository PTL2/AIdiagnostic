"""
File Path Adapter for CLI/Batch Processing Compatibility

Problem:
- unified_data_loader.py expects Streamlit UploadedFile objects (has .name attribute)
- CLI/batch processing uses file path strings (no .name attribute)

Solution:
- FilePathAdapter wraps file paths to mimic UploadedFile interface
- Enables unified_data_loader to work with both UI and CLI

Usage:
    from file_adapter import adapt_file_path
    from unified_data_loader import load_valve_data
    
    # Instead of:
    # df = load_valve_data("path/to/file.wrpm")  # ❌ Fails
    
    # Do this:
    adapted = adapt_file_path("path/to/file.wrpm")
    df, metadata, file_type = load_valve_data(adapted)  # ✅ Works
"""

from pathlib import Path
from typing import Union
import io


class FilePathAdapter:
    """
    Adapter to make file paths compatible with unified_data_loader.
    
    The unified_data_loader expects Streamlit UploadedFile objects with:
    - .name property (filename)
    - .read() method (file contents)
    - .getvalue() method (file contents)
    
    This adapter provides those interfaces for regular file paths.
    """
    
    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize adapter with a file path.
        
        Args:
            file_path: Path to the file (string or Path object)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If path is a directory
        """
        self.path = Path(file_path)
        
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if self.path.is_dir():
            raise ValueError(f"Path is a directory, not a file: {file_path}")
        
        # Cache file contents for .read() and .getvalue()
        self._contents = None
    
    @property
    def name(self) -> str:
        """
        Return filename (compatible with UploadedFile.name).
        
        This is what unified_data_loader uses to detect file type:
        - filename.endswith('.wrpm') → WRPM parser
        - filename.endswith('.xml') → XML parser
        """
        return self.path.name
    
    def read(self) -> bytes:
        """
        Read file contents (compatible with UploadedFile.read()).
        
        Returns:
            bytes: Raw file contents
        """
        if self._contents is None:
            with open(self.path, 'rb') as f:
                self._contents = f.read()
        return self._contents
    
    def getvalue(self) -> bytes:
        """
        Get file contents (compatible with UploadedFile.getvalue()).
        
        Some parsers use .getvalue() instead of .read()
        
        Returns:
            bytes: Raw file contents
        """
        return self.read()
    
    def seek(self, offset: int, whence: int = 0) -> int:
        """
        Seek to position in file (compatible with file-like objects).
        
        For file paths, we just return 0 (no-op) since we cache contents.
        
        Args:
            offset: Position to seek to
            whence: Reference point (0=start, 1=current, 2=end)
            
        Returns:
            int: New position (always 0 for cached adapter)
        """
        return 0
    
    def __str__(self) -> str:
        """String representation."""
        return f"FilePathAdapter({self.path})"
    
    def __repr__(self) -> str:
        """Developer representation."""
        return f"FilePathAdapter(path='{self.path}', name='{self.name}')"


def adapt_file_path(file_path: Union[str, Path]) -> FilePathAdapter:
    """
    Convert a file path to an object compatible with unified_data_loader.
    
    This is the main function you'll use in your code.
    
    Args:
        file_path: Path to XML or WRPM file (string or Path)
        
    Returns:
        FilePathAdapter: Object compatible with unified_data_loader
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If path is a directory
        
    Example:
        >>> from file_adapter import adapt_file_path
        >>> from unified_data_loader import load_valve_data
        >>> 
        >>> # Adapt the file path
        >>> adapted = adapt_file_path("data/sample.wrpm")
        >>> 
        >>> # Now it works with unified_data_loader
        >>> df, metadata, file_type = load_valve_data(adapted)
        >>> print(f"Loaded {len(df)} data points")
    """
    return FilePathAdapter(file_path)


# Quick test when run directly
if __name__ == "__main__":
    import sys
    
    print("🧪 Testing FilePathAdapter...")
    print("=" * 60)
    
    # Test with a dummy file path (won't actually exist in test)
    print("\n📝 Creating test file...")
    test_file = Path("test_sample.wrpm")
    
    # Create a dummy file for testing
    test_file.write_bytes(b"dummy WRPM content for testing")
    
    try:
        # Test the adapter
        print(f"\n✅ Test file created: {test_file}")
        print(f"   File size: {test_file.stat().st_size} bytes")
        
        # Create adapter
        print(f"\n🔧 Creating FilePathAdapter...")
        adapter = adapt_file_path(test_file)
        
        # Test .name property
        print(f"\n📋 Testing .name property:")
        print(f"   adapter.name = '{adapter.name}'")
        assert adapter.name == "test_sample.wrpm", "Name mismatch!"
        print(f"   ✅ .name works correctly")
        
        # Test .read() method
        print(f"\n📖 Testing .read() method:")
        content = adapter.read()
        print(f"   Read {len(content)} bytes")
        assert content == b"dummy WRPM content for testing", "Content mismatch!"
        print(f"   ✅ .read() works correctly")
        
        # Test .getvalue() method
        print(f"\n📖 Testing .getvalue() method:")
        content2 = adapter.getvalue()
        print(f"   Got {len(content2)} bytes")
        assert content2 == content, "getvalue() != read()!"
        print(f"   ✅ .getvalue() works correctly")
        
        # Test string representation
        print(f"\n🔍 Testing string representations:")
        print(f"   str(adapter) = {str(adapter)}")
        print(f"   repr(adapter) = {repr(adapter)}")
        print(f"   ✅ String representations work")
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\n💡 Usage in your code:")
        print("""
    from file_adapter import adapt_file_path
    from unified_data_loader import load_valve_data
    
    # Adapt file path for unified_data_loader
    adapted = adapt_file_path("path/to/file.wrpm")
    df, metadata, file_type = load_valve_data(adapted)
        """)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        # Clean up test file
        if test_file.exists():
            test_file.unlink()
            print(f"\n🧹 Cleaned up test file")
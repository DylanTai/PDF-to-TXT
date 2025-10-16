"""
Windows-specific configuration for PDF to Text converter
"""

def configure_paths():
    """
    Configure Tesseract and Poppler paths for Windows
    
    Returns:
        tuple: (tesseract_cmd, poppler_path)
    """
    
    # Default Tesseract path for Windows
    tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    # Default Poppler path for Windows
    poppler_path = r'C:\poppler\Library\bin'
    
    print("\nWindows Configuration:")
    print(f"  Tesseract: {tesseract_cmd}")
    print(f"  Poppler: {poppler_path}")
    print("\nIf these paths are incorrect, edit windows.py to update them.")
    
    return tesseract_cmd, poppler_path
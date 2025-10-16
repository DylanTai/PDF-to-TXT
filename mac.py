"""
macOS-specific configuration for PDF to Text converter
"""

def configure_paths():
    """
    Configure Tesseract and Poppler paths for macOS
    
    Returns:
        tuple: (tesseract_cmd, poppler_path)
    """
    
    # Try to detect Tesseract path automatically
    import subprocess
    
    tesseract_cmd = None
    try:
        # Try to find Tesseract using 'which'
        result = subprocess.run(['which', 'tesseract'], 
                              capture_output=True, 
                              text=True, 
                              check=True)
        tesseract_cmd = result.stdout.strip()
    except:
        # If 'which' fails, try common installation paths
        import os
        common_paths = [
            '/opt/homebrew/bin/tesseract',  # Apple Silicon Macs
            '/usr/local/bin/tesseract',      # Intel Macs
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                tesseract_cmd = path
                break
    
    # Poppler is usually in PATH on Mac, so we don't need to specify it
    poppler_path = None
    
    print("\nmacOS Configuration:")
    if tesseract_cmd:
        print(f"  Tesseract: {tesseract_cmd}")
    else:
        print("  Tesseract: Using system PATH")
        print("  ⚠ Warning: Tesseract not found in common locations")
        print("  Make sure it's installed: brew install tesseract")
    
    print(f"  Poppler: Using system PATH")
    print("\nIf Tesseract path is incorrect, edit mac.py to update it.")
    
    return tesseract_cmd, poppler_path
import sys
from pathlib import Path
import time
from PIL import Image

# Import functions from the main pdftotext.py file
try:
    from pdftotext import (
        configure_paths,
        preprocess_image_for_ocr,
        get_dpi_for_enhancement_level,
        process_for_excel,
        extract_item_from_buffer,
        parse_data_fields,
        format_item_line,
        TESSERACT_CMD,
        POPPLER_PATH
    )
except ImportError as e:
    print("Error: Could not import from pdftotext.py")
    print("Make sure pdftotext.py is in the same directory as pdftotext_test.py")
    print(f"Import error: {e}")
    sys.exit(1)

import pytesseract
from pdf2image import convert_from_path

# Use the configured paths from pdftotext.py
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ============================================================================
# TEST PROCESSING FUNCTION
# ============================================================================

def test_pdf_processing(pdf_path, ocr_enhancement='medium', num_pages=10):
    """
    Test PDF processing on first N pages only
    
    Args:
        pdf_path: Path to the PDF file
        ocr_enhancement: 'low', 'medium', or 'high'
        num_pages: Number of pages to process (default: 10)
    
    Returns:
        Path to the test_files directory
    """
    
    # Create test_files directory
    test_dir = Path('test_files')
    test_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    images_dir = test_dir / 'preprocessed_images'
    images_dir.mkdir(exist_ok=True)
    
    dpi = get_dpi_for_enhancement_level(ocr_enhancement)
    
    print(f"\n{'='*60}")
    print(f"TEST MODE - Processing first {num_pages} pages only")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path}")
    print(f"Enhancement: {ocr_enhancement}")
    print(f"DPI: {dpi}")
    print(f"Output directory: {test_dir}")
    print(f"{'='*60}\n")
    
    # Convert first N pages
    print(f"Converting first {num_pages} pages to images...")
    start_time = time.time()
    
    if POPPLER_PATH:
        images = convert_from_path(
            pdf_path, 
            dpi=dpi,
            first_page=1,
            last_page=num_pages,
            poppler_path=POPPLER_PATH
        )
    else:
        images = convert_from_path(
            pdf_path, 
            dpi=dpi,
            first_page=1,
            last_page=num_pages
        )
    
    conversion_time = time.time() - start_time
    print(f"✓ Converted {len(images)} pages in {conversion_time:.2f} seconds\n")
    
    # Process each page
    raw_text_all = []
    
    for i, image in enumerate(images, 1):
        print(f"Processing page {i}/{len(images)}...")
        
        # Save original image
        original_path = images_dir / f'page_{i:02d}_original.png'
        image.save(original_path)
        print(f"  ✓ Saved original: {original_path}")
        
        # Preprocess image using function from pdftotext.py
        processed_image = preprocess_image_for_ocr(image, ocr_enhancement)
        
        # Save preprocessed image
        processed_path = images_dir / f'page_{i:02d}_preprocessed.png'
        processed_image.save(processed_path)
        print(f"  ✓ Saved preprocessed: {processed_path}")
        
        # Perform OCR
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(processed_image, lang='eng', config=custom_config)
        raw_text_all.append(f"\n{'='*60}\n")
        raw_text_all.append(f"PAGE {i}\n")
        raw_text_all.append(f"{'='*60}\n")
        raw_text_all.append(text)
        
        print(f"  ✓ OCR completed")
        print(f"  Preview: {text[:100].replace(chr(10), ' ')}...\n")
        
        # Free memory
        del image
        del processed_image
    
    # Free memory
    del images
    
    # Save raw OCR text
    raw_text_path = test_dir / 'raw_ocr_text.txt'
    with open(raw_text_path, 'w', encoding='utf-8') as f:
        f.writelines(raw_text_all)
    print(f"✓ Saved raw OCR text: {raw_text_path}\n")
    
    # Process and format for Excel using function from pdftotext.py
    print("Formatting text for Excel...")
    combined_text = ''.join(raw_text_all)
    formatted_text = process_for_excel(combined_text)
    
    # Save formatted text
    formatted_path = test_dir / 'formatted_output.txt'
    with open(formatted_path, 'w', encoding='utf-8') as f:
        f.write(formatted_text)
    print(f"✓ Saved formatted output: {formatted_path}\n")
    
    print(f"{'='*60}")
    print("TEST COMPLETE!")
    print(f"{'='*60}")
    print(f"\nAll files saved to: {test_dir.absolute()}")
    print(f"  - Preprocessed images: {images_dir.absolute()}")
    print(f"  - Raw OCR text: {raw_text_path.absolute()}")
    print(f"  - Formatted output: {formatted_path.absolute()}")
    
    return test_dir


# ============================================================================
# MAIN TEST PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PDF OCR TEST MODE - First 10 Pages Only")
    print("=" * 60)
    
    # Get input filename
    pdf_file = input("\nEnter the PDF filename: ").strip()
    
    if not pdf_file.lower().endswith('.pdf'):
        pdf_file = pdf_file + '.pdf'
    
    # Check if file exists
    if not Path(pdf_file).exists():
        print(f"\n✗ Error: Could not find file '{pdf_file}'")
        print("Make sure the PDF file exists in the same directory as this script.")
        sys.exit(1)
    
    # Ask for OCR enhancement level
    print("\nOCR Enhancement Levels:")
    print("  'low'    - Basic enhancement, DPI: 200 (fastest)")
    print("  'medium' - Balanced enhancement, DPI: 300 (recommended)")
    print("  'high'   - Aggressive enhancement, DPI: 400 (best quality)")
    
    enhancement = input("\nSelect enhancement level (low/medium/high) [default: medium]: ").strip().lower()
    if enhancement not in ['low', 'medium', 'high']:
        enhancement = 'medium'
        print(f"Using default: {enhancement}")
    
    # Run the test
    try:
        total_start = time.time()
        result = test_pdf_processing(pdf_file, ocr_enhancement=enhancement, num_pages=10)
        total_time = time.time() - total_start
        
        print(f"\n✓ Total test time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
import sys
from pathlib import Path
import time
import shutil

# Import functions from the main pdftotext.py file
try:
    from pdftotext import (
        preprocess_image_for_ocr,
        get_dpi_for_enhancement_level,
        process_for_excel,
        extract_item_from_buffer,
        parse_data_fields,
        format_item_line,
        get_textract_client,
        textract_detect_text,
        POPPLER_PATH,
    )
except ImportError as e:
    print("Error: Could not import from pdftotext.py")
    print("Make sure pdftotext.py is in the same directory as pdftotext_test.py")
    print(f"Import error: {e}")
    sys.exit(1)

from pdf2image import convert_from_path, pdfinfo_from_path


# ============================================================================
# TEST PROCESSING FUNCTION
# ============================================================================

def test_pdf_processing(pdf_path, ocr_enhancement='medium', textract_client=None):
    """
    Test PDF processing on up to the first 10 and last 10 pages.
    Uses Amazon Textract for OCR so AWS credentials must be configured.
    
    Args:
        pdf_path: Path to the PDF file
        ocr_enhancement: 'low', 'medium', or 'high'
        textract_client: Optional Textract client (pass stub for unit tests)
    
    Returns:
        Path to the test_files directory
    """
    
    # Create test_files directory
    test_dir = Path('test_files')
    if not test_dir.exists():
        print(f"\n✗ Error: 'test_files' directory does not exist!")
        print("Please create it first: mkdir test_files")
        sys.exit(1)
    
    # Create subdirectories
    images_dir = test_dir / 'preprocessed_images'
    
    # Clear and recreate the preprocessed_images directory
    if images_dir.exists():
        print(f"\nCleaning up old images in {images_dir}...")
        shutil.rmtree(images_dir)
        print(f"  ✓ Removed old images")
    
    images_dir.mkdir(exist_ok=True)
    print(f"  ✓ Created fresh preprocessed_images directory\n")
    
    dpi = get_dpi_for_enhancement_level(ocr_enhancement)
    textract_client = textract_client or get_textract_client()
    
    # Get total number of pages
    try:
        if POPPLER_PATH:
            info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
        else:
            info = pdfinfo_from_path(pdf_path)
        total_pages = info['Pages']
    except Exception as e:
        print(f"\n✗ Error: Could not get PDF info: {e}")
        sys.exit(1)
    
    head_pages = list(range(1, min(total_pages, 10) + 1))
    tail_start = max(1, total_pages - 9)
    tail_pages = list(range(tail_start, total_pages + 1))
    pages_to_process = sorted(set(head_pages + tail_pages))

    print(f"{'='*60}")
    print(f"TEST MODE - Processing {len(pages_to_process)} page(s)")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path}")
    print(f"Total pages: {total_pages}")
    print(f"Processing pages: {pages_to_process}")
    print(f"Enhancement: {ocr_enhancement}")
    print(f"DPI: {dpi}")
    print(f"AWS Textract Region: {getattr(textract_client.meta, 'region_name', 'unknown')}")
    print(f"Output directory: {test_dir}")
    print(f"{'='*60}\n")
    
    images_to_process = []

    for page_num in pages_to_process:
        print(f"Converting page {page_num}...")
        start_time = time.time()

        if POPPLER_PATH:
            page_images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_num,
                last_page=page_num,
                poppler_path=POPPLER_PATH
            )
        else:
            page_images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_num,
                last_page=page_num
            )

        conversion_time = time.time() - start_time
        print(f"✓ Converted page {page_num} in {conversion_time:.2f} seconds\n")

        if page_images:
            images_to_process.append((page_num, page_images[0]))

        del page_images
    
    # Process each page
    raw_text_all = []
    
    for page_num, image in images_to_process:
        print(f"Processing page {page_num}...")
        
        # Save original image
        original_path = images_dir / f'page_{page_num:04d}_original.png'
        image.save(original_path)
        print(f"  ✓ Saved original: {original_path}")
        
        # Preprocess image using function from pdftotext.py
        processed_image = preprocess_image_for_ocr(image, ocr_enhancement)
        
        # Save preprocessed image
        processed_path = images_dir / f'page_{page_num:04d}_preprocessed.png'
        processed_image.save(processed_path)
        print(f"  ✓ Saved preprocessed: {processed_path}")
        
        # Perform OCR with Textract
        text = textract_detect_text(processed_image, textract_client=textract_client)
        raw_text_all.append(f"\n{'='*60}\n")
        raw_text_all.append(f"PAGE {page_num}\n")
        raw_text_all.append(f"{'='*60}\n")
        raw_text_all.append(text)
        
        print(f"  ✓ Textract OCR completed")
        print(f"  Preview: {text[:100].replace(chr(10), ' ')}...\n")
        
        # Free memory
        del image
        del processed_image
    
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
    print("PDF OCR TEST MODE - First & Last 10 Pages")
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
        result = test_pdf_processing(pdf_file, ocr_enhancement=enhancement)
        total_time = time.time() - total_start
        
        print(f"\n✓ Total test time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

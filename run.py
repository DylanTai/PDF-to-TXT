#!/usr/bin/env python3
"""
Main entry point for PDF processing.
Provides three options:
1. read_all - Process all pages of a PDF
2. read_some - Process first N pages of a PDF
3. read_raw - Show raw Textract output from first N pages
"""

import sys
from pathlib import Path
import time
import re

# Import necessary functions
try:
    from src.read_some import (
        pdf_to_trade_summary,
        get_dpi_for_enhancement_level,
        get_textract_client,
        textract_detect_text,
        preprocess_image_for_ocr,
        process_trade_summary,
        POPPLER_PATH,
    )
    from src.read_raw import show_raw_textract_output
except ImportError as e:
    print("Error: Could not import required modules")
    print(f"Import error: {e}")
    sys.exit(1)

from pdf2image import convert_from_path, pdfinfo_from_path


# Pattern to detect section headers (for name format creation)
SECTION_HEADER_PATTERN = re.compile(r'^([A-Z]{3})\s+---\s+(.+)$')


# ============================================================================
# FORMATTING HELPER FUNCTIONS
# ============================================================================

def create_name_format(section_file_path, delimiter='|'):
    """
    Create a name-formatted version (items only, no section headers).

    Args:
        section_file_path: Path to the _section.txt file
        delimiter: Column delimiter

    Returns:
        Path: Path to the created _name.txt file
    """
    # Determine output path
    if section_file_path.stem.endswith('_section'):
        base_name = section_file_path.stem[:-8]  # Remove '_section'
    else:
        base_name = section_file_path.stem

    name_path = section_file_path.parent / f"{base_name}_name.txt"

    # Read the section file
    with open(section_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Extract items (skip header, section headers, TOTALS section, and blank lines)
    header = None
    items = []
    in_totals = False

    for line in lines:
        line = line.rstrip('\n')

        # Skip blank lines
        if not line:
            continue

        # Capture header
        if not header and delimiter in line and line.startswith('Description'):
            header = line
            continue

        # Check if we're entering TOTALS section
        if line == 'TOTALS':
            in_totals = True
            continue

        # Skip section headers
        if SECTION_HEADER_PATTERN.match(line):
            continue

        # Skip lines in TOTALS section
        if in_totals:
            continue

        # This is an item line
        if delimiter in line:
            items.append(line)

    # Write name format file
    with open(name_path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header + '\n')
        for item in items:
            f.write(item + '\n')

    return name_path


# ============================================================================
# PROCESS SOME PAGES FUNCTION
# ============================================================================

def process_some_pages(pdf_path, num_pages, ocr_enhancement='medium', delimiter='|'):
    """
    Process only the first N pages of a PDF.

    Args:
        pdf_path: Path to the PDF file
        num_pages: Number of pages to process starting from page 1
        ocr_enhancement: 'low', 'medium', or 'high'
        delimiter: Column separator (default: '|')

    Returns:
        Path to the output text file
    """
    dpi = get_dpi_for_enhancement_level(ocr_enhancement)

    # Set output path
    pdf_file = Path(pdf_path)
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)
    output_txt_path = output_dir / f"{pdf_file.stem}_some_{num_pages}_section.txt"

    textract_client = get_textract_client()
    client_meta = getattr(textract_client, 'meta', None)
    region_name = getattr(client_meta, 'region_name', 'unknown')

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

    # Limit to available pages
    pages_to_process = min(num_pages, total_pages)

    print(f"{'='*60}")
    print(f"PROCESSING FIRST {pages_to_process} PAGE(S)")
    print(f"{'='*60}")
    print(f"PDF: {pdf_path}")
    print(f"Total pages in PDF: {total_pages}")
    print(f"Processing pages: 1 to {pages_to_process}")
    print(f"OCR Enhancement Level: {ocr_enhancement}")
    print(f"DPI Setting: {dpi}")
    print(f"AWS Textract Region: {region_name}")
    print(f"Delimiter: '{delimiter}'")
    print(f"{'='*60}\n")

    all_text = []

    # Process pages one by one
    for page_num in range(1, pages_to_process + 1):
        print(f"Processing page {page_num}/{pages_to_process}...")
        page_start = time.time()

        # Convert PDF page to image
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

        if not page_images:
            print(f"  ✗ Failed to convert page {page_num}")
            continue

        image = page_images[0]

        # Preprocess image for better OCR
        processed_image = preprocess_image_for_ocr(image, ocr_enhancement)

        # Perform OCR with Textract
        text = textract_detect_text(processed_image, textract_client=textract_client)

        page_time = time.time() - page_start
        print(f"  ✓ Completed in {page_time:.2f} seconds\n")

        all_text.append(text)

        # Free memory
        del image
        del processed_image
        del page_images

    print(f"{'='*60}")
    print("Processing and formatting text for Excel...")
    print(f"{'='*60}\n")

    # Combine all text
    combined_text = '\n'.join(all_text)

    # Process text to make it Excel-friendly
    processed_text = process_trade_summary(combined_text, delimiter=delimiter)

    # Save to TXT file
    print(f"Saving to file: {output_txt_path}")
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    print(f"✓ TXT file created!")

    print(f"{'='*60}")
    print(f"\n✓✓✓ SUCCESS! ✓✓✓")
    print(f"\nFile saved: {output_txt_path}")
    print(f"Total characters extracted: {len(processed_text):,}")
    print(f"\nYou can open this file and copy-paste into Excel.")
    print(f"Then use Data > Text to Columns > Delimited > Other: {delimiter}")
    print(f"{'='*60}")

    return output_txt_path


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Main program entry point with user menu."""
    print("\n" + "=" * 60)
    print("PDF PROCESSING TOOL")
    print("=" * 60)
    print("\nAvailable options:")
    print("  1. read_all    - Process all pages of a PDF")
    print("  2. read_some   - Process first N pages of a PDF")
    print("  3. read_raw    - Show raw Textract output from first N pages")
    print("=" * 60)

    # ========================================
    # Get user choice
    # ========================================

    choice = input("\nEnter your choice (1, 2, or 3): ").strip()

    if choice not in ['1', '2', '3']:
        print("\n✗ Error: Invalid choice. Please enter 1, 2, or 3.")
        sys.exit(1)

    # ========================================
    # Scan for PDF files
    # ========================================

    # Get project root directory
    project_root = Path(__file__).resolve().parent
    pdf_dir = project_root / 'pdf'

    # Check if pdf directory exists
    if not pdf_dir.exists():
        print(f"\n✗ Error: 'pdf/' directory does not exist!")
        print(f"Please create it: mkdir {pdf_dir.absolute()}")
        sys.exit(1)

    # Scan for PDF files in the pdf/ directory
    pdf_files = sorted([f for f in pdf_dir.iterdir() if f.is_file() and f.suffix.lower() == '.pdf'])

    # Filter out hidden files and .gitkeep
    pdf_files = [f for f in pdf_files if not f.name.startswith('.') and f.name != '.gitkeep']

    if not pdf_files:
        print(f"\n✗ Error: No PDF files found in the 'pdf/' directory!")
        print(f"Please place your PDF files in: {pdf_dir.absolute()}")
        sys.exit(1)

    # ========================================
    # Display available PDFs
    # ========================================

    print("\n" + "=" * 60)
    print("AVAILABLE PDF FILES:")
    print("=" * 60)
    for idx, pdf_file in enumerate(pdf_files, 1):
        file_size = pdf_file.stat().st_size
        size_mb = file_size / (1024 * 1024)
        print(f"  {idx}. {pdf_file.name} ({size_mb:.2f} MB)")
    print("=" * 60)

    # ========================================
    # Get PDF selection from user
    # ========================================

    pdf_choice = input(f"\nSelect a PDF (1-{len(pdf_files)}): ").strip()

    try:
        pdf_index = int(pdf_choice) - 1
        if pdf_index < 0 or pdf_index >= len(pdf_files):
            print(f"\n✗ Error: Please enter a number between 1 and {len(pdf_files)}.")
            sys.exit(1)
    except ValueError:
        print("\n✗ Error: Please enter a valid number.")
        sys.exit(1)

    # Get the selected PDF path
    pdf_path = pdf_files[pdf_index]

    # Verify it's still a valid PDF file
    if not pdf_path.exists() or not pdf_path.is_file():
        print(f"\n✗ Error: Selected file is not accessible: {pdf_path.name}")
        sys.exit(1)

    if pdf_path.suffix.lower() != '.pdf':
        print(f"\n✗ Error: Selected file is not a PDF: {pdf_path.name}")
        sys.exit(1)

    print(f"\n✓ Selected: {pdf_path.name}")

    # ========================================
    # Get number of pages for read_some and read_raw
    # ========================================

    num_pages = None
    if choice in ['2', '3']:
        num_pages_input = input("\nHow many pages to process? ").strip()
        try:
            num_pages = int(num_pages_input)
            if num_pages < 1:
                print("\n✗ Error: Number of pages must be at least 1.")
                sys.exit(1)
        except ValueError:
            print("\n✗ Error: Please enter a valid number.")
            sys.exit(1)

    # ========================================
    # Get OCR enhancement level
    # ========================================

    enhancement = 'medium'
    if choice in ['1', '2', '3']:
        print("\nOCR Enhancement Levels:")
        print("  'low'    - Basic enhancement, DPI: 200 (fastest)")
        print("  'medium' - Balanced enhancement, DPI: 300 (recommended)")
        print("  'high'   - Aggressive enhancement, DPI: 400 (best quality)")

        enhancement_input = input("\nSelect enhancement level (low/medium/high) [default: medium]: ").strip().lower()
        if enhancement_input in ['low', 'medium', 'high']:
            enhancement = enhancement_input
        else:
            print(f"Using default: {enhancement}")

    # ========================================
    # Get delimiter for read_all and read_some
    # ========================================

    delimiter = '|'
    if choice in ['1', '2']:
        print("\nColumn Delimiter:")
        print("  Press Enter for default pipe delimiter: |")
        print("  Or enter your preferred delimiter (e.g., comma, tab, etc.)")

        delimiter_input = input("\nDelimiter [default: |]: ").strip()
        delimiter = delimiter_input if delimiter_input else '|'

    # ========================================
    # Get output format for read_all and read_some
    # ========================================

    format_type = 'both'
    if choice in ['1', '2']:
        print("\nOutput Format:")
        print("  1. section - Items organized by sections (with section headers)")
        print("  2. name    - Items only (no section headers)")
        print("  3. both    - Create both formats")

        format_input = input("\nSelect format (1/2/3) [default: 3]: ").strip()
        format_map = {'1': 'section', '2': 'name', '3': 'both', '': 'both'}
        format_type = format_map.get(format_input, 'both')

    # ========================================
    # Execute the chosen option
    # ========================================

    try:
        total_start = time.time()

        if choice == '1':
            # Process all pages
            print(f"\nProcessing ALL pages from: {pdf_path.absolute()}")
            print(f"Enhancement level: {enhancement}")
            print(f"Delimiter: '{delimiter}'\n")

            result = pdf_to_trade_summary(
                str(pdf_path),
                None,  # Let function determine output path
                batch_size=20,
                ocr_enhancement=enhancement,
                delimiter=delimiter
            )

        elif choice == '2':
            # Process first N pages
            print(f"\nProcessing first {num_pages} page(s) from: {pdf_path.absolute()}")
            print(f"Enhancement level: {enhancement}")
            print(f"Delimiter: '{delimiter}'\n")

            result = process_some_pages(
                str(pdf_path),
                num_pages,
                ocr_enhancement=enhancement,
                delimiter=delimiter
            )

        elif choice == '3':
            # Show raw output from first N pages
            print(f"\nShowing raw output from first {num_pages} page(s): {pdf_path.absolute()}")
            print(f"Enhancement level: {enhancement}\n")

            result = show_raw_textract_output(
                str(pdf_path),
                num_pages=num_pages,
                ocr_enhancement=enhancement
            )

        # Create additional format if requested (for read_all and read_some)
        if choice in ['1', '2'] and format_type in ['name', 'both']:
            print(f"\n{'='*60}")
            print("Creating additional format(s)...")
            print(f"{'='*60}")

            result_path = Path(result)

            if format_type == 'name' or format_type == 'both':
                try:
                    name_path = create_name_format(result_path, delimiter)
                    print(f"✓ Created name format: {name_path.name}")
                except Exception as e:
                    print(f"✗ Error creating name format: {e}")

        total_time = time.time() - total_start
        print(f"\n✓ Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

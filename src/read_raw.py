import sys
from pathlib import Path
import time

# Import functions from the main pdftotext.py file
try:
    from src.read_some import (
        preprocess_image_for_ocr,
        get_dpi_for_enhancement_level,
        get_textract_client,
        textract_detect_text,
        POPPLER_PATH,
    )
except ImportError as e:
    print("Error: Could not import from pdftotext.py")
    print("Make sure pdftotext.py is in the same directory")
    print(f"Import error: {e}")
    sys.exit(1)

from pdf2image import convert_from_path, pdfinfo_from_path


def show_raw_textract_output(pdf_path, num_pages=3, ocr_enhancement='medium'):
    """
    Show raw Textract output from the first N pages of a PDF.
    This helps understand the formatting before writing a parser.

    Args:
        pdf_path: Path to the PDF file
        num_pages: Number of pages to process (default: 3)
        ocr_enhancement: 'low', 'medium', or 'high'
    """

    dpi = get_dpi_for_enhancement_level(ocr_enhancement)
    textract_client = get_textract_client()

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

    pages_to_process = min(num_pages, total_pages)

    print(f"{'='*80}")
    print(f"RAW TEXTRACT OUTPUT VIEWER")
    print(f"{'='*80}")
    print(f"PDF: {pdf_path}")
    print(f"Total pages in PDF: {total_pages}")
    print(f"Processing first {pages_to_process} page(s)")
    print(f"Enhancement: {ocr_enhancement}")
    print(f"DPI: {dpi}")
    print(f"AWS Textract Region: {getattr(textract_client.meta, 'region_name', 'unknown')}")
    print(f"{'='*80}\n")

    all_raw_text = []

    for page_num in range(1, pages_to_process + 1):
        print(f"Processing page {page_num}...")
        start_time = time.time()

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

        # Preprocess image
        processed_image = preprocess_image_for_ocr(image, ocr_enhancement)

        # Perform OCR with Textract
        text = textract_detect_text(processed_image, textract_client=textract_client)

        elapsed = time.time() - start_time
        print(f"  ✓ Completed in {elapsed:.2f} seconds")
        print(f"  ✓ Extracted {len(text)} characters\n")

        # Store the raw text
        all_raw_text.append(f"\n{'='*80}\n")
        all_raw_text.append(f"PAGE {page_num}\n")
        all_raw_text.append(f"{'='*80}\n")
        all_raw_text.append(text)
        all_raw_text.append("\n")

        # Free memory
        del image
        del processed_image
        del page_images

    # Combine all text
    combined_text = ''.join(all_raw_text)

    # Save to file in outputs folder (in project root)
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / 'outputs'
    output_dir.mkdir(exist_ok=True)

    # Get PDF filename for output naming
    pdf_filename = Path(pdf_path).stem
    output_path = output_dir / f'{pdf_filename}_raw_{num_pages}_section.txt'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    print(f"{'='*80}")
    print(f"COMPLETE!")
    print(f"{'='*80}")
    print(f"\nRaw Textract output saved to: {output_path.absolute()}")
    print(f"Total characters extracted: {len(combined_text):,}")
    print(f"\n{'='*80}")
    print("RAW TEXT PREVIEW (first 2000 characters):")
    print(f"{'='*80}\n")
    print(combined_text[:2000])
    print(f"\n{'='*80}")
    print(f"See full output in: {output_path.absolute()}")
    print(f"{'='*80}\n")

    return output_path


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RAW TEXTRACT OUTPUT VIEWER")
    print("Shows raw OCR text from first 3 pages to help understand formatting")
    print("=" * 80)

    # Get project root directory
    project_root = Path(__file__).resolve().parent.parent
    pdf_dir = project_root / 'pdf'

    # Get input filename
    pdf_file = input("\nEnter the PDF filename: ").strip()

    if not pdf_file.lower().endswith('.pdf'):
        pdf_file = pdf_file + '.pdf'

    # Construct full path to PDF in pdf/ directory
    pdf_path = pdf_dir / pdf_file

    # Check if file exists
    if not pdf_path.exists():
        print(f"\n✗ Error: Could not find file '{pdf_file}'")
        print(f"Please place your PDF files in the 'pdf/' directory: {pdf_dir.absolute()}")
        sys.exit(1)

    # Ask how many pages
    num_pages_input = input("\nHow many pages to process? [default: 3]: ").strip()
    try:
        num_pages = int(num_pages_input) if num_pages_input else 3
    except ValueError:
        num_pages = 3
        print(f"Invalid input, using default: {num_pages}")

    # Ask for OCR enhancement level
    print("\nOCR Enhancement Levels:")
    print("  'low'    - Basic enhancement, DPI: 200 (fastest)")
    print("  'medium' - Balanced enhancement, DPI: 300 (recommended)")
    print("  'high'   - Aggressive enhancement, DPI: 400 (best quality)")

    enhancement = input("\nSelect enhancement level (low/medium/high) [default: medium]: ").strip().lower()
    if enhancement not in ['low', 'medium', 'high']:
        enhancement = 'medium'
        print(f"Using default: {enhancement}")

    # Run the viewer
    try:
        total_start = time.time()
        result = show_raw_textract_output(str(pdf_path), num_pages=num_pages, ocr_enhancement=enhancement)
        total_time = time.time() - total_start

        print(f"\n✓ Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

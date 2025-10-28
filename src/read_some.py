import io
import os
import platform
import re
import sys
import time
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from pdf2image import convert_from_path

import numpy as np
import cv2
from PIL import Image, ImageOps

try:
    RESAMPLING_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9.1
    RESAMPLING_BICUBIC = Image.BICUBIC


# ============================================================================
# LOAD LOCAL ENVIRONMENT VARIABLES IF AVAILABLE
# ============================================================================

def _simple_env_loader(env_path):
    """
    Minimal .env parser used when python-dotenv isn't installed.
    Supports KEY=VALUE pairs, ignoring blank lines and comments.
    """
    with env_path.open("r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
if ENV_PATH.exists():
    try:
        from dotenv import load_dotenv
    except ImportError:
        _simple_env_loader(ENV_PATH)
    else:
        load_dotenv(dotenv_path=ENV_PATH)

# ============================================================================
# PLATFORM-SPECIFIC CONFIGURATION
# ============================================================================

def configure_environment():
    """
    Configure platform specific settings.

    Returns:
        str|None: Poppler path if it needs to be explicitly provided.
    """

    system = platform.system()

    if system == 'Windows':
        poppler_path = r'C:\poppler\Library\bin'

        print("\nWindows Configuration:")
        print(f"  Poppler: {poppler_path}")
        print("\nUpdate configure_environment() if Poppler is installed elsewhere.")

        return poppler_path

    if system == 'Darwin':  # macOS
        poppler_path = None

        print("\nmacOS Configuration:")
        print("  Poppler: Using system PATH")
        print("\nInstall via Homebrew if missing: brew install poppler")

        return poppler_path

    print(f"Unsupported operating system: {system}")
    print("This script supports Windows and macOS only.")
    sys.exit(1)


# Configure paths based on OS
POPPLER_PATH = configure_environment()


# ============================================================================
# IMAGE PREPROCESSING FOR BETTER OCR
# ============================================================================

def remove_watermark(image):
    """
    Light watermark suppression that preserves text edges.
    Falls back to the original image if no obvious watermark is detected.
    """
    img_array = np.array(image)

    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    blurred = cv2.medianBlur(gray, 3)
    background = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    residual = cv2.subtract(blurred, background)
    enhanced = cv2.normalize(residual, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    blended = cv2.addWeighted(gray, 0.7, enhanced, 0.3, 0)
    return Image.fromarray(blended)


def _correct_orientation(image):
    """
    Deskew and ensure the page is mostly upright using min-area rectangle.
    Only corrects small skews; avoids rotating landscape pages that are
    already correctly oriented.
    """
    upright = ImageOps.exif_transpose(image)
    img_array = np.array(upright)
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh < 255))

    if coords.size == 0:
        return upright

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5 or abs(angle) > 15:
        return upright

    return upright.rotate(angle, expand=True, fillcolor="white")


def _crop_document_region(image):
    """
    Remove large uniform borders so OCR focuses on document content.
    """
    img_array = np.array(image)
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(thresh)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(inverted, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    page_area = image.width * image.height
    if w * h < page_area * 0.5:
        return image

    return image.crop((x, y, x + w, y + h))


def _ensure_min_resolution(image, target_dpi):
    """
    Upscale very small renders to improve readability without shrinking images.
    """
    dpi = image.info.get('dpi')

    if isinstance(dpi, tuple) and dpi[0] > 0:
        scale_from_dpi = max(1.0, target_dpi / dpi[0])
    elif isinstance(dpi, (int, float)) and dpi > 0:
        scale_from_dpi = max(1.0, target_dpi / dpi)
    else:
        scale_from_dpi = 1.0

    min_dimension = min(image.size)
    scale_from_size = 1.0
    if min_dimension < 1500:
        scale_from_size = 1500 / float(min_dimension)

    scale = max(scale_from_dpi, scale_from_size)

    if scale <= 1.05:
        return image

    new_size = (int(image.width * scale), int(image.height * scale))
    return image.resize(new_size, RESAMPLING_BICUBIC)


def _adjust_brightness_contrast(image, enhancement_level):
    """
    Apply gentle brightness/contrast corrections when the page is too dark or light.
    """
    img_array = np.array(image)
    if img_array.ndim == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    mean_intensity = float(gray.mean())
    alpha = 1.0
    beta = 0

    if mean_intensity < 110:
        alpha = 1.15 if enhancement_level != 'low' else 1.1
        beta = 12
    elif mean_intensity > 190:
        alpha = 0.9
        beta = -12

    if alpha == 1.0 and beta == 0:
        return image

    adjusted = cv2.convertScaleAbs(img_array, alpha=alpha, beta=beta)
    return Image.fromarray(adjusted)


def _gentle_denoise(image, enhancement_level):
    """
    Light despeckle to remove sensor noise without blurring text strokes.
    """
    img_array = np.array(image)
    if img_array.ndim == 2:
        h_value = 5 if enhancement_level == 'high' else 3
        cleaned = cv2.fastNlMeansDenoising(img_array, None, h=h_value, templateWindowSize=7, searchWindowSize=21)
    else:
        h_value = 7 if enhancement_level == 'high' else 5
        cleaned = cv2.fastNlMeansDenoisingColored(img_array, None, h_value, h_value, 7, 21)
    return Image.fromarray(cleaned)


def _to_grayscale(image):
    """
    Convert to grayscale while preserving subtle intensity variations.
    """
    if image.mode == 'L':
        return image
    return ImageOps.grayscale(image)


def preprocess_image_for_ocr(image, enhancement_level='medium'):
    """
    Preprocess image to improve OCR accuracy using gentle, document-safe steps.

    Args:
        image: PIL Image object
        enhancement_level: 'low', 'medium', or 'high'

    Returns:
        PIL Image object (preprocessed)
    """
    enhancement_level = enhancement_level or 'medium'

    adjusted = remove_watermark(image)
    adjusted = _correct_orientation(adjusted)
    adjusted = _crop_document_region(adjusted)
    adjusted = _ensure_min_resolution(adjusted, get_dpi_for_enhancement_level(enhancement_level))

    adjusted = _adjust_brightness_contrast(adjusted, enhancement_level)
    adjusted = _gentle_denoise(adjusted, enhancement_level)

    grayscale = _to_grayscale(adjusted)
    return grayscale


def get_dpi_for_enhancement_level(enhancement_level):
    """
    Get the appropriate DPI based on enhancement level

    Args:
        enhancement_level: 'low', 'medium', or 'high'

    Returns:
        int: DPI value
    """
    dpi_settings = {
        'low': 200,      # Fast processing, good for high-quality PDFs
        'medium': 300,   # Balanced quality and speed
        'high': 400      # Best quality for difficult PDFs
    }

    return dpi_settings.get(enhancement_level, 300)


# ============================================================================
# AWS TEXTRACT OCR HELPERS
# ============================================================================

MAX_TEXTRACT_IMAGE_BYTES = 4_500_000  # Safety margin under the 5 MB Textract limit


def get_textract_client(region_name=None):
    """
    Create (or reuse) a Textract client using environment settings.

    Args:
        region_name: Optional AWS region override.

    Returns:
        botocore.client.Textract
    """
    session_kwargs = {}

    if region_name:
        session_kwargs['region_name'] = region_name
    else:
        env_region = os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION')
        if env_region:
            session_kwargs['region_name'] = env_region

    profile_name = os.getenv('AWS_PROFILE')
    if profile_name:
        session_kwargs['profile_name'] = profile_name

    if 'region_name' not in session_kwargs:
        session_kwargs['region_name'] = 'us-east-1'

    session = boto3.Session(**session_kwargs)
    return session.client('textract')


def _encode_image_for_textract(image):
    """
    Encode a PIL image into a byte payload acceptable by Textract.

    Textract enforces a maximum size of 5 MB for synchronous calls, so we try
    PNG first, then JPEG as a fallback with mild compression.
    """
    working_image = image
    if working_image.mode not in ('L', 'RGB'):
        working_image = working_image.convert('L')

    buffer = io.BytesIO()
    working_image.save(buffer, format='PNG', optimize=True)
    payload = buffer.getvalue()

    if len(payload) <= MAX_TEXTRACT_IMAGE_BYTES:
        return payload

    # Fallback to compressed JPEG in grayscale to reduce size
    jpeg_image = working_image.convert('L')
    buffer = io.BytesIO()
    jpeg_image.save(buffer, format='JPEG', quality=85, optimize=True)
    payload = buffer.getvalue()

    if len(payload) > MAX_TEXTRACT_IMAGE_BYTES:
        raise ValueError(
            "Preprocessed page exceeds Textract's 5 MB limit after compression. "
            "Try lowering the OCR enhancement level or reducing DPI."
        )

    return payload


def textract_detect_text(image, textract_client=None):
    """
    Run Textract OCR on an image and return the detected text.

    Args:
        image: PIL Image to process
        textract_client: Optional, pre-initialised Textract client

    Returns:
        str: Concatenated line text detected by Textract.
    """
    client = textract_client or get_textract_client()
    document_bytes = _encode_image_for_textract(image)

    try:
        response = client.detect_document_text(Document={'Bytes': document_bytes})
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"AWS Textract OCR failed: {exc}") from exc

    lines = [
        block['Text']
        for block in response.get('Blocks', [])
        if block.get('BlockType') == 'LINE' and 'Text' in block
    ]

    return '\n'.join(lines)


# ============================================================================
# TRADE SUMMARY PARSING FUNCTIONS
# ============================================================================

# Pattern to detect section headers (3 uppercase letters)
SECTION_CODE_PATTERN = re.compile(r'^([A-Z]{3})$')

# Pattern to detect quantity with unit
QUANTITY_PATTERN = re.compile(r'^(\d+\.?\d*)\s+(EA|ea|PC|pc|LB|lb|OZ|oz|PK|pk)\b')

# Pattern to detect currency
CURRENCY_PATTERN = re.compile(r'^-?\$[\d,]+(?:\.\d{2})?$')

# Pattern to detect total lines
TOTAL_PATTERN = re.compile(r'^TOTAL\s+(.+?)$', re.IGNORECASE)

# Skip these header lines
SKIP_LINES = (
    'trade summary',
    'includes all applicable',
    'description',
    'line item',
    'qty',
    'repl. cost',
    'total',
    'acv',
    'non-rec.',
    'deprec.',
    'max addl.',
    'amt avail.',
    'note: slight variances',
    'page:',
)


def _normalize_text(text):
    """Collapse excessive whitespace."""
    return ' '.join(text.split())


def _is_skip_line(line):
    """Check if this line should be skipped (headers, footers, etc.)"""
    lower = line.lower()
    return any(skip in lower for skip in SKIP_LINES)


def _consume_currency(lines, index):
    """
    Consume a currency value from lines[index].
    Returns the value and the next index.
    """
    while index < len(lines) and not lines[index].strip():
        index += 1

    if index < len(lines):
        candidate = lines[index].strip()
        if CURRENCY_PATTERN.match(candidate):
            return candidate, index + 1

    return '', index


def process_trade_summary(text, delimiter='|'):
    """
    Parse trade summary format from Textract OCR output.

    Expected format:
    - Section header: 3-letter code (e.g., "AMA")
    - Section description: Full name (e.g., "AUTOMOTIVE & MOTORCYCLE ACC.")
    - Items: Description + Qty + 4 currency columns
    - Totals: "TOTAL <section name>" + 4 currency columns

    Args:
        text: Raw OCR text from Textract
        delimiter: Column separator (default: '|')

    Returns:
        Formatted text ready for Excel import
    """
    lines = text.split('\n')

    # Build header
    header = delimiter.join([
        "Description",
        "Line Item Qty",
        "Repl. Cost Total",
        "ACV",
        "Non-Rec. Deprec.",
        "Max Addl. Amt Avail.",
        "Type"
    ])

    processed_lines = [header]
    totals = []  # Store totals for final summary

    current_section_code = None
    current_section_name = None
    index = 0
    total_lines = len(lines)

    while index < total_lines:
        line = lines[index].strip()

        # Skip empty lines
        if not line:
            index += 1
            continue

        # Skip header/footer lines
        if _is_skip_line(line):
            index += 1
            continue

        # Check for section code (3 uppercase letters)
        section_match = SECTION_CODE_PATTERN.match(line)
        if section_match:
            current_section_code = section_match.group(1)
            index += 1

            # Next line should be the section description
            if index < total_lines:
                next_line = lines[index].strip()
                if next_line and not _is_skip_line(next_line):
                    current_section_name = next_line

                    # Add section header to output
                    section_header = f"{current_section_code} --- {current_section_name}"
                    processed_lines.append(section_header)
                    index += 1
            continue

        # Check for total line
        total_match = TOTAL_PATTERN.match(line)
        if total_match:
            section_for_total = total_match.group(1)
            index += 1

            # Consume 4 currency values
            repl_cost, index = _consume_currency(lines, index)
            acv, index = _consume_currency(lines, index)
            non_rec_deprec, index = _consume_currency(lines, index)
            max_addl, index = _consume_currency(lines, index)

            # Store total for summary
            if current_section_code and repl_cost:
                totals.append({
                    'code': current_section_code,
                    'name': section_for_total,
                    'repl_cost': repl_cost,
                    'acv': acv,
                    'non_rec_deprec': non_rec_deprec,
                    'max_addl': max_addl
                })

            continue

        # Check if this might be an item description
        # Items typically start with text, not currency
        if not CURRENCY_PATTERN.match(line) and not QUANTITY_PATTERN.match(line):
            # Collect description lines
            description_parts = [line]
            index += 1

            # Keep collecting until we hit a quantity pattern
            while index < total_lines:
                next_line = lines[index].strip()

                if not next_line:
                    index += 1
                    continue

                # Stop if we hit a quantity line
                if QUANTITY_PATTERN.match(next_line):
                    break

                # Stop if we hit a new section or total
                if SECTION_CODE_PATTERN.match(next_line) or TOTAL_PATTERN.match(next_line):
                    break

                # Stop if we hit a skip line
                if _is_skip_line(next_line):
                    index += 1
                    continue

                # Add to description if it's not currency
                if not CURRENCY_PATTERN.match(next_line):
                    description_parts.append(next_line)
                    index += 1
                else:
                    break

            description = _normalize_text(' '.join(description_parts))

            # Now try to consume quantity
            qty = ''
            if index < total_lines:
                qty_candidate = lines[index].strip()
                qty_match = QUANTITY_PATTERN.match(qty_candidate)
                if qty_match:
                    qty = f"{qty_match.group(1)} {qty_match.group(2).upper()}"
                    index += 1

            # Consume 4 currency values
            repl_cost, index = _consume_currency(lines, index)
            acv, index = _consume_currency(lines, index)
            non_rec_deprec, index = _consume_currency(lines, index)
            max_addl, index = _consume_currency(lines, index)

            # Only add if we have at least description and one currency value
            if description and (repl_cost or acv or non_rec_deprec or max_addl):
                formatted_line = delimiter.join([
                    description,
                    qty,
                    repl_cost,
                    acv,
                    non_rec_deprec,
                    max_addl,
                    current_section_code or ''
                ])
                processed_lines.append(formatted_line)
            else:
                # No valid data, skip this line
                pass
        else:
            # Line starts with currency or quantity, skip it
            index += 1

    # Add totals summary at the end
    if totals:
        processed_lines.append('')  # Blank line
        processed_lines.append('TOTALS')

        for total in totals:
            total_line = delimiter.join([
                f"TOTAL {total['name']} ({total['code']})",
                '',  # No quantity for totals
                total['repl_cost'],
                total['acv'],
                total['non_rec_deprec'],
                total['max_addl'],
                ''  # No type for totals
            ])
            processed_lines.append(total_line)

    return '\n'.join(processed_lines)


# ============================================================================
# MAIN PDF TO EXCEL CONVERSION FUNCTION
# ============================================================================

def pdf_to_trade_summary(
    pdf_path,
    output_txt_path=None,
    batch_size=20,
    ocr_enhancement='medium',
    textract_client=None,
    delimiter='|'
):
    """
    Convert PDF to trade summary format using Amazon Textract OCR.

    Args:
        pdf_path: Path to the PDF file
        output_txt_path: Path for output text file (optional)
        batch_size: Number of pages to process at once
        ocr_enhancement: 'low', 'medium', or 'high'
        textract_client: Optional preconfigured Textract client
        delimiter: Column separator (default: '|')

    Returns:
        Path to the output text file
    """

    # Get DPI based on enhancement level
    dpi = get_dpi_for_enhancement_level(ocr_enhancement)

    # Set output path if not provided
    if output_txt_path is None:
        pdf_file = Path(pdf_path)
        # Get the project root directory (parent of src/)
        project_root = Path(__file__).resolve().parent.parent
        output_dir = project_root / 'outputs'
        output_dir.mkdir(exist_ok=True)
        output_txt_path = output_dir / f"{pdf_file.stem}_output.txt"

    textract_client = textract_client or get_textract_client()
    client_meta = getattr(textract_client, 'meta', None)
    region_name = getattr(client_meta, 'region_name', 'unknown')

    print(f"Starting conversion of: {pdf_path}")
    print(f"Operating System: {platform.system()}")
    print(f"OCR Enhancement Level: {ocr_enhancement}")
    print(f"DPI Setting: {dpi}")
    print(f"AWS Textract Region: {region_name}")
    print(f"Batch size: {batch_size} pages at a time")
    print(f"Delimiter: '{delimiter}'")
    print("=" * 60)

    all_text = []

    # Get total number of pages
    from pdf2image import pdfinfo_from_path
    try:
        if POPPLER_PATH:
            info = pdfinfo_from_path(pdf_path, poppler_path=POPPLER_PATH)
        else:
            info = pdfinfo_from_path(pdf_path)
        total_pages = info['Pages']
        print(f"\nTotal pages in PDF: {total_pages}")
    except:
        print("\nCouldn't determine page count, will process in batches...")
        total_pages = None

    print("\nProcessing PDF in batches...")
    print("=" * 60)

    # Process in batches
    page_num = 1
    batch_num = 1

    while True:
        try:
            print(f"\nBatch {batch_num}: Processing pages {page_num} to {page_num + batch_size - 1}...")
            batch_start = time.time()

            # Convert a batch of pages
            if POPPLER_PATH:
                images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=page_num,
                    last_page=page_num + batch_size - 1,
                    poppler_path=POPPLER_PATH
                )
            else:
                images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=page_num,
                    last_page=page_num + batch_size - 1
                )

            if not images:
                print("No more pages to process.")
                break

            batch_time = time.time() - batch_start
            print(f"  ✓ Batch converted to images ({batch_time:.2f} seconds)")
            print(f"  Processing {len(images)} pages with enhanced OCR...")

            # Process each page in the batch
            for i, image in enumerate(images, page_num):
                page_start = time.time()
                print(f"    [Page {i}] Preprocessing image...", end=" ", flush=True)

                # Preprocess image for better OCR
                processed_image = preprocess_image_for_ocr(image, ocr_enhancement)

                print("Textract OCR...", end=" ", flush=True)

                text = textract_detect_text(processed_image, textract_client=textract_client)

                page_time = time.time() - page_start
                print(f"✓ ({page_time:.2f}s)")

                all_text.append(text)

                # Free memory
                del image
                del processed_image

            # Free memory
            del images

            # Move to next batch
            page_num += batch_size
            batch_num += 1

            # Check if we've processed all pages
            if total_pages and page_num > total_pages:
                break

        except Exception as e:
            # If we get an error (like no more pages), we're done
            if "Unable to get page" in str(e) or "Invalid page" in str(e):
                print(f"\n✓ Reached end of document")
                break
            else:
                print(f"\n✗ Error processing batch: {e}")
                break

    print("\n" + "=" * 60)
    print("Step 2: Processing and formatting text for Excel...")

    # Combine all text
    combined_text = '\n'.join(all_text)

    # Process text to make it Excel-friendly
    processed_text = process_trade_summary(combined_text, delimiter=delimiter)

    # Save to TXT file
    print(f"Step 3: Saving to file: {output_txt_path}")
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write(processed_text)
    print(f"✓ TXT file created!")

    print("=" * 60)
    print(f"\n✓✓✓ SUCCESS! ✓✓✓")
    print(f"\nFile saved: {output_txt_path}")
    print(f"Total characters extracted: {len(processed_text):,}")
    print(f"\nYou can open this file and copy-paste into Excel.")
    print(f"Then use Data > Text to Columns > Delimited > Other: {delimiter}")
    print("=" * 60)

    return output_txt_path


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PDF to Trade Summary Converter (Enhanced OCR)")
    print("=" * 60)

    # Get project root directory
    project_root = Path(__file__).resolve().parent.parent
    pdf_dir = project_root / 'pdf'

    # Get input filename from user
    pdf_file = input("\nEnter the PDF filename (e.g., name_of_pdf.pdf): ").strip()

    # Remove .pdf extension if user included it, then add it back
    if pdf_file.lower().endswith('.pdf'):
        base_name = pdf_file[:-4]
    else:
        base_name = pdf_file
        pdf_file = pdf_file + '.pdf'

    # Construct full path to PDF in pdf/ directory
    pdf_path = pdf_dir / pdf_file

    # Check if PDF exists
    if not pdf_path.exists():
        print(f"\n✗ Error: Could not find file '{pdf_file}'")
        print(f"Please place your PDF files in the 'pdf/' directory: {pdf_dir.absolute()}")
        sys.exit(1)

    # Create output filename
    output_file = f"{base_name}_output.txt"

    # Ask for delimiter
    print("\nColumn Delimiter:")
    print("  Press Enter for default pipe delimiter: |")
    print("  Or enter your preferred delimiter (e.g., comma, tab, etc.)")

    delimiter_input = input("\nDelimiter [default: |]: ").strip()
    delimiter = delimiter_input if delimiter_input else '|'

    # Ask for OCR enhancement level
    print("\nOCR Enhancement Levels:")
    print("  'low'    - Basic enhancement, DPI: 200 (fastest)")
    print("  'medium' - Balanced enhancement, DPI: 300 (recommended)")
    print("  'high'   - Aggressive enhancement, DPI: 400 (best quality, slowest)")

    enhancement = input("\nSelect enhancement level (low/medium/high) [default: medium]: ").strip().lower()
    if enhancement not in ['low', 'medium', 'high']:
        enhancement = 'medium'
        print(f"Using default: {enhancement}")

    print(f"\nInput file: {pdf_path.absolute()}")
    print(f"Output will be saved to: outputs/{output_file}")
    print(f"Enhancement level: {enhancement}")
    print(f"Delimiter: '{delimiter}'")
    print()

    # Run the conversion with batch processing
    try:
        total_start = time.time()
        result = pdf_to_trade_summary(
            str(pdf_path),
            None,  # Let function determine output path
            batch_size=20,
            ocr_enhancement=enhancement,
            delimiter=delimiter
        )
        total_time = time.time() - total_start

        print(f"\nTotal processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

    except FileNotFoundError:
        print(f"\n✗ Error: Could not find file '{pdf_file}'")
        print(f"Make sure the PDF file is in the pdf/ directory: {pdf_dir.absolute()}")
    except Exception as e:
        print(f"\n✗ Error: {e}")

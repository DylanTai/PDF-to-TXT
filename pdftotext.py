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


ENV_PATH = Path(__file__).resolve().parent / ".env"
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


# Export these for use by test script
__all__ = [
    'configure_environment',
    'remove_watermark',
    'preprocess_image_for_ocr',
    'get_dpi_for_enhancement_level',
    'process_for_excel',
    'extract_item_from_buffer',
    'parse_data_fields',
    'format_item_line',
    'get_textract_client',
    'textract_detect_text',
    'POPPLER_PATH',
]


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
# MAIN PDF TO EXCEL CONVERSION FUNCTIONS
# ============================================================================

def pdf_to_excel_ready_text(
    pdf_path,
    output_txt_path=None,
    batch_size=20,
    ocr_enhancement='medium',
    textract_client=None,
):
    """
    Convert PDF to text using Amazon Textract OCR and format for Excel paste.
    
    Args:
        pdf_path: Path to the PDF file
        output_txt_path: Path for output text file (optional)
        batch_size: Number of pages to process at once (to avoid memory issues)
        ocr_enhancement: 'low', 'medium', or 'high' - level of image preprocessing
        textract_client: Optional preconfigured Textract client (useful for tests)
    
    Returns:
        Path to the output text file
    """
    
    # Get DPI based on enhancement level
    dpi = get_dpi_for_enhancement_level(ocr_enhancement)
    
    # Set output path if not provided
    if output_txt_path is None:
        pdf_file = Path(pdf_path)
        output_txt_path = pdf_file.with_suffix('.txt')
    
    textract_client = textract_client or get_textract_client()
    client_meta = getattr(textract_client, 'meta', None)
    region_name = getattr(client_meta, 'region_name', 'unknown')

    print(f"Starting conversion of: {pdf_path}")
    print(f"Operating System: {platform.system()}")
    print(f"OCR Enhancement Level: {ocr_enhancement}")
    print(f"DPI Setting: {dpi} (auto-selected based on enhancement level)")
    print(f"AWS Textract Region: {region_name}")
    print(f"Batch size: {batch_size} pages at a time")
    print("=" * 60)
    
    all_text = []
    
    # First, get the total number of pages
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
    
    print("\nProcessing PDF in batches to avoid memory issues...")
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
    combined_text = ''.join(all_text)
    
    # Process text to make it Excel-friendly
    processed_text = process_for_excel(combined_text)
    
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
    print(f"Then use Data > Text to Columns > Delimited > Other: |")
    print("=" * 60)
    
    return output_txt_path


ITEM_START_PATTERN = re.compile(r'^(\d[\d,]*)[.)]\s+(.*)$')
UNIT_PATTERN = re.compile(r'^(\d+\.?\d*)\s+(EA|ea|PC|pc|LB|lb|OZ|oz|PK|pk)\b')
CURRENCY_PATTERN = re.compile(r'^-?\$[\d,]+(?:\.\d{2})?$')
STOP_PREFIXES = (
    'subtotal',
    'coverage',
    'tai inventory',
    'line item detail',
    'rc benefits',
    'description',
    'qty',
    'estimate amount',
    'taxes',
    'replacement cost',
    'age / cond',
    'less depreciation',
    'actual cash value',
    'paid',
    'estimated remaining',
    'amount of rc benefits',
    'page',
)

FOOTNOTE_PATTERNS = (
    re.compile(r'^\*'),
    re.compile(r'^the amount', re.IGNORECASE),
    re.compile(r'^indicates', re.IGNORECASE),
    re.compile(r'^[A-Z0-9]{6,}$'),
    re.compile(r'^item found', re.IGNORECASE),
    re.compile(r'^duplicate$', re.IGNORECASE),
)


def _normalize_description(text):
    """Collapse excessive whitespace in descriptions."""
    return ' '.join(text.split())


def _collect_description_lines(lines, start_index):
    """
    Gather free-form description lines until a structured value or the next
    item header appears. Returns the concatenated description fragment and
    the new index.
    """
    parts = []
    index = start_index

    while index < len(lines):
        candidate = lines[index].strip()
        if not candidate:
            index += 1
            continue

        lower = candidate.lower()
        next_non_empty = index + 1
        while next_non_empty < len(lines) and not lines[next_non_empty].strip():
            next_non_empty += 1

        if next_non_empty < len(lines) and UNIT_PATTERN.match(lines[next_non_empty].strip()):
            break

        if ITEM_START_PATTERN.match(candidate):
            break
        if UNIT_PATTERN.match(candidate):
            break
        if CURRENCY_PATTERN.match(candidate):
            break
        if any(lower.startswith(prefix) for prefix in STOP_PREFIXES):
            break
        if any(pattern.match(candidate) for pattern in FOOTNOTE_PATTERNS):
            break

        parts.append(candidate)
        index += 1

    description = _normalize_description(' '.join(parts))
    return description, index


def _consume_currency(lines, index):
    """
    Consume a currency-like value from lines[index], returning the value and
    the next index. If the current line does not look like currency, returns
    an empty string without advancing.
    """
    while index < len(lines) and not lines[index].strip():
        index += 1

    if index < len(lines):
        candidate = lines[index].strip()
        if CURRENCY_PATTERN.match(candidate):
            return candidate, index + 1

    return '', index


def _consume_age_line(lines, index):
    """
    Capture age/condition/life lines such as '2y/Avg./10y'.
    """
    while index < len(lines) and not lines[index].strip():
        index += 1

    if index < len(lines):
        candidate = lines[index].strip()
        if 'y/' in candidate.lower():
            return candidate, index + 1

    return '', index


def process_for_excel(text):
    """
    Convert Textract OCR output into pipe-delimited rows ready for Excel.
    The parser walks the OCR text sequentially, respecting the PDF layout:
    item header -> quantity -> monetary columns -> optional notes.
    """
    lines = text.split('\n')

    header = (
        "Number|Description|Qty|Estimate Amount|Taxes|Replacement Cost Total|"
        "Age / Cond. / Life|Less Depreciation|Actual Cash Value|Paid|Estimated Remaining|Page Number"
    )
    processed_lines = [header]
    items_found = []

    index = 0
    total_lines = len(lines)
    last_item_number = None
    current_page = None

    while index < total_lines:
        line = lines[index].strip()

        if not line:
            index += 1
            continue

        if set(line) <= {'='} and line:
            index += 1
            continue

        page_block_match = re.match(r'^PAGE\s+(\d+)', line, re.IGNORECASE)
        if page_block_match:
            current_page = int(page_block_match.group(1))
            index += 1
            continue

        page_colon_match = re.match(r'^Page[:\s]+(\d+)', line, re.IGNORECASE)
        if page_colon_match:
            current_page = int(page_colon_match.group(1))
            index += 1
            continue

        match = ITEM_START_PATTERN.match(line)
        if not match:
            implicit_item = None
            if (
                last_item_number is not None
                and line
                and not CURRENCY_PATTERN.match(line)
                and not UNIT_PATTERN.match(line)
                and not any(line.lower().startswith(prefix) for prefix in STOP_PREFIXES)
                and not any(pattern.match(line) for pattern in FOOTNOTE_PATTERNS)
            ):
                lookahead = index + 1
                while lookahead < total_lines and not lines[lookahead].strip():
                    lookahead += 1

                if lookahead < total_lines and UNIT_PATTERN.match(lines[lookahead].strip()):
                    currency_checks = 0
                    probe = lookahead + 1
                    while probe < total_lines and currency_checks < 3:
                        probe_line = lines[probe].strip()
                        if probe_line:
                            if CURRENCY_PATTERN.match(probe_line):
                                currency_checks += 1
                            else:
                                break
                        probe += 1

                    if currency_checks >= 3:
                        implicit_item = {
                            'number': str(last_item_number + 1),
                            'description': _normalize_description(line),
                        }

            if not implicit_item:
                index += 1
                continue

            item_number = implicit_item['number']
            description = implicit_item['description']
            index += 1
        else:
            item_number = match.group(1).replace(',', '')
            description = _normalize_description(match.group(2))
            index += 1

        duplicate_marker = False
        if index < total_lines and lines[index].strip().lower() == 'duplicate':
            duplicate_marker = True
            index += 1

        extra_desc, index = _collect_description_lines(lines, index)
        if extra_desc:
            description = _normalize_description(f"{description} {extra_desc}")

        qty = ''
        if index < total_lines:
            qty_candidate = lines[index].strip()
            qty_match = UNIT_PATTERN.match(qty_candidate)
            if qty_match:
                qty = f"{qty_match.group(1)} {qty_match.group(2).upper()}"
                index += 1

        estimate_amount, index = _consume_currency(lines, index)
        taxes, index = _consume_currency(lines, index)
        replacement_total, index = _consume_currency(lines, index)
        age_life, index = _consume_age_line(lines, index)
        less_depreciation, index = _consume_currency(lines, index)
        actual_cash_value, index = _consume_currency(lines, index)
        paid, index = _consume_currency(lines, index)
        estimated_remaining, index = _consume_currency(lines, index)

        trailing_desc, index = _collect_description_lines(lines, index)
        if trailing_desc:
            description = _normalize_description(f"{description} {trailing_desc}")

        if duplicate_marker:
            qty = ''
            estimate_amount = ''
            taxes = ''
            replacement_total = ''
            age_life = ''
            less_depreciation = ''
            actual_cash_value = 'DUPLICATE'
            paid = '$0.00'
            estimated_remaining = '$0.00'

        formatted_line = "|".join(
            [
                item_number,
                description,
                qty,
                estimate_amount,
                taxes,
                replacement_total,
                age_life,
                less_depreciation,
                actual_cash_value,
                paid,
                estimated_remaining,
                str(current_page) if current_page is not None else "",
            ]
        )
        processed_lines.append(formatted_line)

        try:
            item_value = int(item_number)
        except ValueError:
            item_value = None
        else:
            if item_value < 100000:
                items_found.append(item_value)
        if item_value is not None:
            last_item_number = item_value

    if items_found:
        items_found.sort()
        print(f"\n  Items processed: {len(items_found)}")
        print(f"  First item: {items_found[0]}")
        print(f"  Last item: {items_found[-1]}")

        item_range = items_found[-1] - items_found[0] + 1
        if item_range < 50000:
            missing = [num for num in range(items_found[0], items_found[-1] + 1) if num not in items_found]
            if missing:
                print(f"  ⚠ Warning: Missing {len(missing)} items")
                if len(missing) <= 20:
                    print(f"  Missing items: {missing}")
                else:
                    print(f"  Missing items include: {missing[:10]}... and {len(missing) - 10} more")
        else:
            print(f"  ⚠ Item range too large ({item_range:,} items) - skipping gap analysis")
            print("  Note: Some item numbers may be OCR errors (very large numbers)")

    return '\n'.join(processed_lines)


def extract_item_from_buffer(line_buffer):
    """
    Extract item number, description, and data from a buffer of lines.
    The last line should contain the data with "EA" or similar unit.
    """
    if not line_buffer:
        return None
    
    # Combine all lines into one text block
    full_text = ' '.join(line_buffer)
    
    # Try to find the item number at the beginning
    # Look for pattern: number (with possible commas) followed by period
    # Examples: "1.", "175.", "1,001.", "2,500."
    number_match = re.search(r'([\d,]+)\s*\.\s*', full_text)
    
    if not number_match:
        return None
    
    # Extract and clean the number (remove commas)
    item_number = number_match.group(1).replace(',', '')
    
    # Everything after the number and period
    rest_of_text = full_text[number_match.end():].strip()
    
    # Find where the data starts (look for quantity + unit pattern)
    data_match = re.search(r'(\d+\.?\d*\s+(?:EA|ea|PC|pc|LB|lb|OZ|oz|PK|pk)\b)', rest_of_text)
    
    if data_match:
        # Split into description and data
        description = rest_of_text[:data_match.start()].strip()
        data = rest_of_text[data_match.start():].strip()
        
        return {
            'number': item_number,
            'text': description,
            'data': data
        }
    
    return None


def parse_data_fields(data_string):
    """
    Parse the data string into individual fields with pipe separators.
    Expected format: Qty | Estimate Amount | Taxes | Replacement Cost Total | Age/Cond./Life | Less Depreciation | Actual Cash Value | Paid | Estimated Remaining
    """
    fields = []
    
    # Pattern 1: Extract Qty (e.g., "1.00 EA")
    qty_match = re.search(r'(\d+\.?\d*\s+(?:EA|ea|PC|pc|LB|lb|OZ|oz|PK|pk))', data_string)
    if qty_match:
        fields.append(qty_match.group(1).strip())
        data_string = data_string[qty_match.end():].strip()
    
    # Pattern 2: Extract all dollar amounts in order
    dollar_amounts = re.findall(r'-?\$\d+\.?\d*', data_string)
    
    # Pattern 3: Extract Age/Cond./Life pattern (e.g., "2 y/ Avg. /10 y")
    age_match = re.search(r'(\d+\.?\d*\s*y/\s*\w+\.?\s*/\s*\d+\.?\d*\s*y)', data_string)
    age_text = age_match.group(1).strip() if age_match else ""
    
    # Now we need to figure out which dollar amounts go where
    # Expected order: Estimate Amount, Taxes, Replacement Cost Total, Less Depreciation, Actual Cash Value, Paid, Estimated Remaining
    
    if len(dollar_amounts) >= 3:
        # First dollar amount: Estimate Amount
        fields.append(dollar_amounts[0])
        
        # Second dollar amount: Taxes
        fields.append(dollar_amounts[1])
        
        # Third dollar amount: Replacement Cost Total
        fields.append(dollar_amounts[2])
        
        # Age/Cond./Life
        fields.append(age_text)
        
        # Remaining dollar amounts
        if len(dollar_amounts) >= 4:
            fields.append(dollar_amounts[3])  # Less Depreciation
        if len(dollar_amounts) >= 5:
            fields.append(dollar_amounts[4])  # Actual Cash Value
        if len(dollar_amounts) >= 6:
            fields.append(dollar_amounts[5])  # Paid
        if len(dollar_amounts) >= 7:
            fields.append(dollar_amounts[6])  # Estimated Remaining
    
    return fields


def format_item_line(item):
    """
    Format a numbered item with pipes separating each individual field.
    Format: Number|Description|Qty|Estimate Amount|Taxes|Replacement Cost Total|Age/Cond./Life|Less Depreciation|Actual Cash Value|Paid|Estimated Remaining
    """
    # Start with the item number, then the description
    result = f"{item['number']}|{item['text'].strip()}"
    
    # Add data if available - parse each field intelligently
    if 'data' in item:
        fields = parse_data_fields(item['data'])
        
        # Add each field with a pipe separator
        for field in fields:
            result += '|' + field
    
    return result


# ============================================================================
# MAIN PROGRAM
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PDF to Excel-Ready Text Converter (Enhanced OCR)")
    print("=" * 60)
    
    # Get input filename from user
    pdf_file = input("\nEnter the PDF filename (e.g., name_of_pdf.pdf): ").strip()
    
    # Remove .pdf extension if user included it, then add it back
    if pdf_file.lower().endswith('.pdf'):
        base_name = pdf_file[:-4]
    else:
        base_name = pdf_file
        pdf_file = pdf_file + '.pdf'
    
    # Create output filename
    output_file = f"{base_name}_output.txt"
    
    # Ask for OCR enhancement level
    print("\nOCR Enhancement Levels:")
    print("  'low'    - Basic enhancement, DPI: 200 (fastest)")
    print("  'medium' - Balanced enhancement, DPI: 300 (recommended)")
    print("  'high'   - Aggressive enhancement, DPI: 400 (best quality, slowest)")
    
    enhancement = input("\nSelect enhancement level (low/medium/high) [default: medium]: ").strip().lower()
    if enhancement not in ['low', 'medium', 'high']:
        enhancement = 'medium'
        print(f"Using default: {enhancement}")
    
    print(f"\nInput file: {pdf_file}")
    print(f"Output will be saved as: {base_name}_output.txt")
    print(f"Enhancement level: {enhancement}")
    print()
    
    # Run the conversion with batch processing (20 pages at a time)
    try:
        total_start = time.time()
        result = pdf_to_excel_ready_text(pdf_file, output_file, batch_size=20, ocr_enhancement=enhancement)
        total_time = time.time() - total_start
        
        print(f"\nTotal processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
    except FileNotFoundError:
        print(f"\n✗ Error: Could not find file '{pdf_file}'")
        print("Make sure the PDF file exists in the same directory as this script.")
    except Exception as e:
        print(f"\n✗ Error: {e}")

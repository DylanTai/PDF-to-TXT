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
from PIL import Image


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
    Remove watermark/draft text from image
    """
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Apply binary threshold to separate text from background
    # Use Otsu's method for automatic threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed (text should be black on white)
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    
    # Use morphological operations to remove large faint text (watermark)
    # while preserving small dark text (actual content)
    
    # Create a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)
    
    # Apply morphological closing to connect nearby text
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(cv2.bitwise_not(closed), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create mask to remove large, low-contrast elements (watermark)
    mask = np.ones_like(gray) * 255
    
    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Remove very large components (likely watermark)
        # Watermarks are typically large and cover significant area
        if area > 50000 or (w > 500 and h > 200):
            cv2.drawContours(mask, [contour], -1, 0, -1)
    
    # Apply mask
    result = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Enhance contrast after watermark removal
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(result)
    
    return Image.fromarray(result)


def preprocess_image_for_ocr(image, enhancement_level='medium'):
    """
    Preprocess image to improve OCR accuracy
    
    Args:
        image: PIL Image object
        enhancement_level: 'low', 'medium', or 'high' - level of preprocessing
    
    Returns:
        PIL Image object (preprocessed)
    """
    
    # First, try to remove watermark
    image = remove_watermark(image)
    
    # Convert PIL Image to numpy array for OpenCV processing
    img_array = np.array(image)
    
    # Already grayscale from watermark removal
    gray = img_array
    
    if enhancement_level == 'low':
        # Light preprocessing - just basic contrast enhancement
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
    elif enhancement_level == 'medium':
        # Medium preprocessing - contrast + denoising + sharpening
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
    else:  # 'high'
        # Aggressive preprocessing - all techniques
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive thresholding for better text separation
        enhanced = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1,1), np.uint8)
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # Sharpen
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
    
    # Convert back to PIL Image
    processed_image = Image.fromarray(enhanced)
    
    return processed_image


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


def process_for_excel(text):
    """
    Process OCR text to make it Excel-friendly.
    NEW APPROACH: Find lines with "EA" (or other units), then extract the item info.
    """
    
    lines = text.split('\n')
    
    # Define the header
    header = "Number|Description|Qty|Estimate Amount|Taxes|Replacement Cost Total|Age / Cond. / Life|Less Depreciation|Actual Cash Value|Paid|Estimated Remaining"
    
    processed_lines = [header]
    items_found = []
    
    # Buffer to collect lines that might be part of an item
    line_buffer = []
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue
        
        # Add to buffer
        line_buffer.append(line)
        
        # Check if this line contains a unit (EA, PC, LB, etc.)
        # This indicates we have the data row
        if re.search(r'\b(EA|ea|PC|pc|LB|lb|OZ|oz|PK|pk)\b', line):
            # We found a data line! Now extract the item from the buffer
            item = extract_item_from_buffer(line_buffer)
            
            if item:
                processed_lines.append(format_item_line(item))
                if item.get('number'):
                    try:
                        item_num = int(item['number'])
                        # Only track reasonable item numbers (less than 100,000)
                        if item_num < 100000:
                            items_found.append(item_num)
                    except:
                        pass
            
            # Clear the buffer
            line_buffer = []
        
        # Keep buffer manageable (max 10 lines)
        if len(line_buffer) > 10:
            line_buffer.pop(0)
    
    # Print statistics
    if items_found:
        items_found.sort()
        print(f"\n  Items processed: {len(items_found)}")
        print(f"  First item: {min(items_found)}")
        print(f"  Last item: {max(items_found)}")
        
        # Only check for gaps if the range is reasonable (less than 50,000 items)
        item_range = max(items_found) - min(items_found) + 1
        
        if item_range < 50000:
            # Check for gaps
            missing = []
            for i in range(min(items_found), max(items_found) + 1):
                if i not in items_found:
                    missing.append(i)
            
            if missing:
                print(f"  ⚠ Warning: Missing {len(missing)} items")
                if len(missing) <= 20:
                    print(f"  Missing items: {missing}")
                else:
                    print(f"  Missing items include: {missing[:10]}... and {len(missing)-10} more")
        else:
            print(f"  ⚠ Item range too large ({item_range:,} items) - skipping gap analysis")
            print(f"  Note: Some item numbers may be OCR errors (very large numbers)")
    
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

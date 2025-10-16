import sys
import platform
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
import re
import time

# ============================================================================
# PLATFORM-SPECIFIC CONFIGURATION
# ============================================================================

def configure_paths():
    """
    Configure Tesseract and Poppler paths based on operating system
    
    Returns:
        tuple: (tesseract_cmd, poppler_path)
    """
    
    system = platform.system()
    
    if system == 'Windows':
        # Windows Configuration
        tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        poppler_path = r'C:\poppler\Library\bin'
        
        print("\nWindows Configuration:")
        print(f"  Tesseract: {tesseract_cmd}")
        print(f"  Poppler: {poppler_path}")
        print("\nIf these paths are incorrect, edit the configure_paths() function.")
        
        return tesseract_cmd, poppler_path
        
    elif system == 'Darwin':  # macOS
        # macOS Configuration - try to auto-detect Tesseract
        import subprocess
        import os
        
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
        print("\nIf Tesseract path is incorrect, edit the configure_paths() function.")
        
        return tesseract_cmd, poppler_path
        
    else:
        print(f"Unsupported operating system: {system}")
        print("This script supports Windows and macOS only.")
        sys.exit(1)


# Configure paths based on OS
TESSERACT_CMD, POPPLER_PATH = configure_paths()
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


# ============================================================================
# MAIN PDF TO EXCEL CONVERSION FUNCTIONS
# ============================================================================

def pdf_to_excel_ready_text(pdf_path, output_txt_path=None, dpi=300, batch_size=20):
    """
    Convert PDF to text using Tesseract OCR and format for Excel paste.
    
    Args:
        pdf_path: Path to the PDF file
        output_txt_path: Path for output text file (optional)
        dpi: DPI for PDF to image conversion (higher = better quality but slower)
        batch_size: Number of pages to process at once (to avoid memory issues)
    
    Returns:
        Path to the output text file
    """
    
    # Set output path if not provided
    if output_txt_path is None:
        pdf_file = Path(pdf_path)
        output_txt_path = pdf_file.with_suffix('.txt')
    
    print(f"Starting conversion of: {pdf_path}")
    print(f"Operating System: {platform.system()}")
    print(f"DPI Setting: {dpi}")
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
            print(f"  Processing {len(images)} pages with OCR...")
            
            # Process each page in the batch
            for i, image in enumerate(images, page_num):
                page_start = time.time()
                print(f"    [Page {i}] Processing OCR...", end=" ", flush=True)
                
                # Perform OCR on the image
                text = pytesseract.image_to_string(image, lang='eng')
                
                page_time = time.time() - page_start
                print(f"✓ ({page_time:.2f}s)")
                
                all_text.append(text)
                
                # Free memory
                del image
            
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
                        items_found.append(int(item['number']))
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
    print("PDF to Excel-Ready Text Converter")
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
    
    print(f"\nInput file: {pdf_file}")
    print(f"Output will be saved as: {base_name}_output.txt")
    print()
    
    # Run the conversion with batch processing (20 pages at a time)
    try:
        total_start = time.time()
        result = pdf_to_excel_ready_text(pdf_file, output_file, dpi=300, batch_size=20)
        total_time = time.time() - total_start
        
        print(f"\nTotal processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        
    except FileNotFoundError:
        print(f"\n✗ Error: Could not find file '{pdf_file}'")
        print("Make sure the PDF file exists in the same directory as this script.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
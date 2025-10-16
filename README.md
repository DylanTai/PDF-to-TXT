# PDF to Excel OCR Converter (Enhanced)

## Description

This program converts PDF files into Excel-ready text files using OCR (Optical Character Recognition). It's specifically designed to extract numbered inventory items from PDF documents and format them into pipe-delimited columns that can be easily imported into Excel.

**NEW: Enhanced OCR with automatic DPI selection and image preprocessing for improved accuracy!**

## What It Does

1. **Reads PDF files** - Converts each page of your PDF into images
2. **Auto-adjusts quality** - Automatically selects optimal DPI based on enhancement level
3. **Enhances images** - Preprocesses images with denoising, contrast enhancement, and sharpening
4. **Performs OCR** - Extracts text from the enhanced images using Tesseract OCR with optimized settings
5. **Intelligent parsing** - Identifies numbered items by looking for unit markers (EA, PC, LB, OZ, etc.)
6. **Formats for Excel** - Organizes data into columns separated by pipe (`|`) delimiters
7. **Handles large files** - Processes PDFs in batches to avoid memory issues
8. **Cross-platform** - Automatically detects and configures for Windows or macOS

## OCR Enhancement Levels

The program offers three levels of enhancement with automatic DPI optimization:

| Level      | DPI | Image Processing                                    | Speed    | Best For                     |
| ---------- | --- | --------------------------------------------------- | -------- | ---------------------------- |
| **Low**    | 200 | Basic contrast enhancement                          | Fastest  | High-quality, clear PDFs     |
| **Medium** | 300 | Contrast + denoising + sharpening                   | Balanced | Most PDFs (recommended)      |
| **High**   | 400 | Aggressive preprocessing with adaptive thresholding | Slowest  | Poor-quality or scanned PDFs |

**DPI is automatically selected** - you just choose the enhancement level!

## Output Format

The program creates a text file with the following columns:

- Number
- Description
- Qty
- Estimate Amount
- Taxes
- Replacement Cost Total
- Age / Cond. / Life
- Less Depreciation
- Actual Cash Value
- Paid
- Estimated Remaining

Each line represents one inventory item, with fields separated by `|` characters.

## Files Included

- `pdftotext.py` - Single script that works on both Windows and macOS

## Requirements

### Windows

- Python 3
- Tesseract OCR
- Poppler for Windows
- Python packages: `pytesseract`, `pdf2image`, `pillow`, `opencv-python`, `numpy`

### Mac

- Python 3
- Tesseract OCR (via Homebrew)
- Poppler (via Homebrew)
- Python packages: `pytesseract`, `pdf2image`, `pillow`, `opencv-python`, `numpy`

## Installation

### Windows

```bash
# Install Python packages
pip install pytesseract pdf2image pillow opencv-python numpy

# Download and install Tesseract OCR from:
# https://github.com/UB-Mannheim/tesseract/wiki

# Download and extract Poppler from:
# https://github.com/oschwartz10612/poppler-windows/releases
# Extract to C:\poppler (or update the path in pdftotext.py)
```

### Mac

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required software
brew install tesseract poppler python3

# Install Python packages
pip3 install pytesseract pdf2image pillow opencv-python numpy
```

## Configuration

The script automatically detects your operating system and configures paths accordingly.

### Windows

If Tesseract or Poppler are installed in different locations, edit the `configure_paths()` function in `pdftotext.py`:

```python
if system == 'Windows':
    tesseract_cmd = r'C:\Your\Custom\Path\tesseract.exe'
    poppler_path = r'C:\Your\Custom\Path\poppler\Library\bin'
```

### Mac

The script automatically detects Tesseract on macOS. If it fails, edit the `configure_paths()` function:

```python
elif system == 'Darwin':
    tesseract_cmd = '/your/custom/path/tesseract'
    poppler_path = None
```

## Usage

1. Place your PDF file in the same directory as the script
2. Run the script:
   - **Windows:** `python pdftotext.py`
   - **Mac:** `python3 pdftotext.py`
3. Enter the filename when prompted (e.g., `inventory.pdf`)
4. **Choose an OCR enhancement level** (low/medium/high)
   - DPI is automatically selected based on your choice
5. Wait for processing to complete
6. Find the output file: `yourfilename_output.txt`

**The script will automatically detect your operating system and use the appropriate configuration!**

### Example Session

```
Enter the PDF filename (e.g., name_of_pdf.pdf): inventory.pdf

OCR Enhancement Levels:
  'low'    - Basic enhancement, DPI: 200 (fastest)
  'medium' - Balanced enhancement, DPI: 300 (recommended)
  'high'   - Aggressive enhancement, DPI: 400 (best quality, slowest)

Select enhancement level (low/medium/high) [default: medium]: medium

Starting conversion of: inventory.pdf
Operating System: Darwin
OCR Enhancement Level: medium
DPI Setting: 300 (auto-selected based on enhancement level)
```

## Importing into Excel

1. Open the output `.txt` file in a text editor
2. Copy all content (Ctrl+A, Ctrl+C / Cmd+A, Cmd+C)
3. Open Excel and paste into cell A1
4. Select all pasted data
5. Go to **Data** → **Text to Columns**
6. Choose **Delimited** → Click **Next**
7. Check **Other** and type `|` in the box → Click **Next**
8. Click **Finish**

Your data will now be properly split into columns!

## Performance Tips

- **Large PDFs (100+ pages)**: The script processes in batches of 20 pages to avoid memory issues
- **Choose the right level**:
  - Start with **medium** for most PDFs
  - Use **low** if processing is too slow and your PDF is already clear
  - Use **high** if you're missing items or text quality is poor
- **Memory errors**: Reduce batch size from 20 to 10 pages by editing the script
- **Processing time estimates** (per page):

| Level  | DPI | Time per Page |
| ------ | --- | ------------- |
| Low    | 200 | 1-3 seconds   |
| Medium | 300 | 3-6 seconds   |
| High   | 400 | 6-12 seconds  |

To manually adjust batch size, modify the function call at the bottom of `pdftotext.py`:

```python
result = pdf_to_excel_ready_text(pdf_file, output_file, batch_size=10, ocr_enhancement=enhancement)
```

## OCR Enhancement Details

### Low Enhancement (DPI: 200)

- **Best for**: Clean, high-quality PDFs with clear text
- **Speed**: Fastest (~1-3 seconds per page)
- **Techniques**: CLAHE contrast enhancement
- **Image processing**: Minimal
- **Use when**: Text is already very readable in the PDF, speed is priority

### Medium Enhancement (DPI: 300) - Recommended

- **Best for**: Most PDFs with standard quality
- **Speed**: Balanced (~3-6 seconds per page)
- **Techniques**: Denoising + CLAHE + Sharpening
- **Image processing**: Moderate
- **Use when**: General purpose, unsure about PDF quality, or starting point

### High Enhancement (DPI: 400)

- **Best for**: Poor-quality scans, faded text, noisy images, or when items are missing
- **Speed**: Slowest but most accurate (~6-12 seconds per page)
- **Techniques**: Adaptive thresholding + morphological operations + denoising + CLAHE + sharpening
- **Image processing**: Aggressive
- **Use when**:
  - OCR is missing many items with lower settings
  - PDF contains scanned documents
  - Text is faint or has artifacts
  - Small font sizes

## Troubleshooting

### Windows

- **"Tesseract not found"**: Update the path in the `configure_paths()` function to match your Tesseract installation
- **"Poppler not found"**: Update the `poppler_path` in the `configure_paths()` function to match where you extracted Poppler
- **"cv2 module not found"**: Run `pip install opencv-python numpy`

### Mac

- **"Tesseract not found"**: Run `brew install tesseract` or update the path in the `configure_paths()` function
- **"Poppler not found"**: Run `brew install poppler`
- **"cv2 module not found"**: Run `pip3 install opencv-python numpy`

### Both Platforms

- **Missing items in output**:
  1. First, try **high** enhancement level
  2. Check the console output for statistics on missing items
  3. Verify items have unit markers (EA, PC, LB, etc.)
- **Memory errors**:
  - Reduce batch size to 10 or 5 in the script
  - Close other applications to free up RAM
  - Process smaller sections of the PDF at a time
- **Incorrect column splitting**: Verify that items contain unit markers (EA, PC, LB, etc.)
- **Too slow**:
  - Use **low** enhancement level
  - Process fewer pages at once
- **Still poor accuracy on high**:
  - Check original PDF quality
  - Consider re-scanning the document at higher resolution
  - Some PDFs may have formatting that's incompatible with the parser

## Limitations

- Only processes items that contain unit measurements (EA, PC, LB, OZ, PK)
- OCR accuracy depends on PDF quality
- Very large numbers (1000+) may have OCR errors with commas
- Requires consistent formatting in the original PDF
- Processing time increases with higher enhancement levels and DPI
- Memory usage increases with higher DPI settings

## Support

If items are missing from the output, try these steps in order:

1. **Switch to 'high' enhancement level** - This automatically uses 400 DPI and aggressive preprocessing
2. **Check console statistics** - Look for which item numbers are missing
3. **Verify unit markers** - Ensure items have EA, PC, LB, OZ, or PK in them
4. **Review PDF quality** - Open the original PDF and check if text is readable
5. **Check formatting** - Ensure item numbers are followed by periods (e.g., "175.")

### Decision Tree for Enhancement Level

```
Is your PDF clear and high-quality?
├─ Yes → Start with LOW (fastest)
│  └─ Missing items? → Try MEDIUM
│     └─ Still missing? → Try HIGH
│
└─ No/Unsure → Start with MEDIUM (recommended)
   └─ Missing items? → Try HIGH
      └─ Still issues? → Check PDF quality/formatting
```

## Example Output

### Raw Output File (`inventory_output.txt`)

```
Number|Description|Qty|Estimate Amount|Taxes|Replacement Cost Total|Age / Cond. / Life|Less Depreciation|Actual Cash Value|Paid|Estimated Remaining
1|Samsung 55-inch LED Smart TV|1.00 EA|$450.00|$36.00|$486.00|3 y/ Good /7 y|-$150.00|$336.00|$0.00|$150.00
2|Office Desk Chair - Ergonomic Black Leather|2.00 EA|$125.00|$10.00|$135.00|1 y/ Excellent /10 y|-$15.00|$120.00|$0.00|$15.00
127|Wireless Bluetooth Speaker Portable|5.00 EA|$35.99|$2.88|$38.87|0.5 y/ New /5 y|-$4.50|$34.37|$0.00|$4.50
```

### After Importing to Excel

| Number | Description                                 | Qty     | Estimate Amount | Taxes  | Replacement Cost Total | Age / Cond. / Life   | Less Depreciation | Actual Cash Value | Paid  | Estimated Remaining |
| ------ | ------------------------------------------- | ------- | --------------- | ------ | ---------------------- | -------------------- | ----------------- | ----------------- | ----- | ------------------- |
| 1      | Samsung 55-inch LED Smart TV                | 1.00 EA | $450.00         | $36.00 | $486.00                | 3 y/ Good /7 y       | -$150.00          | $336.00           | $0.00 | $150.00             |
| 2      | Office Desk Chair - Ergonomic Black Leather | 2.00 EA | $125.00         | $10.00 | $135.00                | 1 y/ Excellent /10 y | -$15.00           | $120.00           | $0.00 | $15.00              |
| 127    | Wireless Bluetooth Speaker Portable         | 5.00 EA | $35.99          | $2.88  | $38.87                 | 0.5 y/ New /5 y      | -$4.50            | $34.37            | $0.00 | $4.50               |

## Quick Start Guide

**For most users:**

1. Install requirements (see Installation section)
2. Run `python3 pdftotext.py` (Mac) or `python pdftotext.py` (Windows)
3. Enter your PDF filename
4. Choose **medium** enhancement level
5. Wait for processing
6. Import the output .txt file into Excel using Text to Columns with `|` delimiter

**If items are missing:**

- Re-run with **high** enhancement level

**If processing is too slow:**

- Re-run with **low** enhancement level

---

**Note**: This tool is designed for inventory PDFs with specific formatting. Results may vary with different document types. The automatic DPI selection ensures optimal balance between speed and accuracy for each enhancement level.

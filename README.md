# PDF to Excel OCR Converter (Enhanced)

## Description

This program converts PDF files into Excel-ready text files using OCR (Optical Character Recognition). It's specifically designed to extract numbered inventory items from PDF documents and format them into pipe-delimited columns that can be easily imported into Excel.

**NEW: Enhanced OCR with image preprocessing for improved accuracy!**

## What It Does

1. **Reads PDF files** - Converts each page of your PDF into images
2. **Enhances images** - Preprocesses images with denoising, contrast enhancement, and sharpening
3. **Performs OCR** - Extracts text from the enhanced images using Tesseract OCR with optimized settings
4. **Intelligent parsing** - Identifies numbered items by looking for unit markers (EA, PC, LB, OZ, etc.)
5. **Formats for Excel** - Organizes data into columns separated by pipe (`|`) delimiters
6. **Handles large files** - Processes PDFs in batches to avoid memory issues
7. **Cross-platform** - Automatically detects and configures for Windows or macOS

## OCR Enhancement Levels

The program offers three levels of image preprocessing to improve OCR accuracy:

- **Low** - Basic contrast enhancement (fastest, good for high-quality PDFs)
- **Medium** - Contrast + denoising + sharpening (balanced, recommended for most PDFs)
- **High** - Aggressive preprocessing with adaptive thresholding (slowest, best for poor-quality or scanned PDFs)

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
5. Wait for processing to complete
6. Find the output file: `yourfilename_output.txt`

**The script will automatically detect your operating system and use the appropriate configuration!**

### Example Session

```
Enter the PDF filename (e.g., name_of_pdf.pdf): inventory.pdf

OCR Enhancement Levels:
  'low'    - Basic contrast enhancement (fastest)
  'medium' - Contrast + denoising + sharpening (balanced)
  'high'   - Aggressive preprocessing for difficult PDFs (slowest)

Select enhancement level (low/medium/high) [default: medium]: medium
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
- **Low-quality PDFs**: Use 'high' enhancement level for better accuracy (slower)
- **High-quality PDFs**: Use 'low' or 'medium' enhancement level for faster processing
- **Slow processing**: Lower the DPI from 300 to 200 for faster results
- **Memory errors**: Reduce batch size from 20 to 10 pages
- **Processing time**:
  - Low enhancement: ~2-4 seconds per page
  - Medium enhancement: ~3-6 seconds per page
  - High enhancement: ~4-8 seconds per page

To adjust settings, modify the last line of `pdftotext.py`:

```python
result = pdf_to_excel_ready_text(pdf_file, output_file, dpi=200, batch_size=10, ocr_enhancement='low')
```

## OCR Enhancement Details

### Low Enhancement

- **Best for**: Clean, high-quality PDFs with clear text
- **Speed**: Fastest
- **Techniques**: CLAHE contrast enhancement
- **Use when**: Text is already very readable in the PDF

### Medium Enhancement (Recommended)

- **Best for**: Most PDFs with standard quality
- **Speed**: Balanced
- **Techniques**: Denoising + CLAHE + Sharpening
- **Use when**: General purpose, unsure about PDF quality

### High Enhancement

- **Best for**: Poor-quality scans, faded text, or noisy images
- **Speed**: Slowest but most accurate for difficult PDFs
- **Techniques**: Adaptive thresholding + morphological operations + denoising + CLAHE + sharpening
- **Use when**: OCR is missing many items with lower settings

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
  - Try a higher enhancement level ('high')
  - Check the console output for statistics on missing items
  - Increase DPI to 400 or 600 for small text
- **Memory errors**: Reduce batch size or DPI in the function call at the bottom of `pdftotext.py`
- **Incorrect column splitting**: Verify that items contain unit markers (EA, PC, LB, etc.)
- **OCR accuracy issues**:
  - Try different enhancement levels
  - Increase DPI (300-600)
  - Check original PDF quality

## Limitations

- Only processes items that contain unit measurements (EA, PC, LB, OZ, PK)
- OCR accuracy depends on PDF quality (300 DPI recommended, higher for poor quality)
- Very large numbers (1000+) may have OCR errors with commas
- Requires consistent formatting in the original PDF
- Processing time increases with higher enhancement levels

## Support

If items are missing from the output, try these steps in order:

1. Use a higher enhancement level ('high' instead of 'medium')
2. Increase DPI to 400 or 600 (slower but more accurate)
3. Check that items contain unit markers (EA, PC, etc.)
4. Review the console output for missing item numbers
5. Verify original PDF quality and resolution

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

## What's New in Enhanced Version

- ✨ **Image Preprocessing**: Denoising, contrast enhancement, and sharpening for better OCR
- ✨ **Three Enhancement Levels**: Choose based on your PDF quality
- ✨ **Optimized Tesseract Settings**: Better OCR engine configuration
- ✨ **Improved Accuracy**: Especially for poor-quality or scanned PDFs
- ✨ **Smart Processing**: Automatically adjusts preprocessing based on selected level

---

**Note**: This tool is designed for inventory PDFs with specific formatting. Results may vary with different document types. For best results with difficult PDFs, use 'high' enhancement level and higher DPI (400-600).

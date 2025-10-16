# PDF to Excel OCR Converter

## Description

This program converts PDF files into Excel-ready text files using OCR (Optical Character Recognition). It's specifically designed to extract numbered inventory items from PDF documents and format them into pipe-delimited columns that can be easily imported into Excel.

## What It Does

1. **Reads PDF files** - Converts each page of your PDF into images
2. **Performs OCR** - Extracts text from the images using Tesseract OCR
3. **Intelligent parsing** - Identifies numbered items by looking for unit markers (EA, PC, LB, OZ, etc.)
4. **Formats for Excel** - Organizes data into columns separated by pipe (`|`) delimiters
5. **Handles large files** - Processes PDFs in batches to avoid memory issues
6. **Cross-platform** - Automatically detects and configures for Windows or macOS

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
- Python packages: `pytesseract`, `pdf2image`, `pillow`

### Mac

- Python 3
- Tesseract OCR (via Homebrew)
- Poppler (via Homebrew)
- Python packages: `pytesseract`, `pdf2image`, `pillow`

## Installation

### Windows

```bash
# Install Python packages
pip install pytesseract pdf2image pillow

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
pip3 install pytesseract pdf2image pillow
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
4. Wait for processing to complete
5. Find the output file: `yourfilename_output.txt`

**The script will automatically detect your operating system and use the appropriate configuration!**

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
- **Slow processing**: Lower the DPI from 300 to 200 for faster (but slightly less accurate) results
- **Memory errors**: Reduce batch size from 20 to 10 pages
- **Processing time**: Expect approximately 2-5 seconds per page depending on your computer

To adjust settings, modify the last line of `pdftotext.py`:

```python
result = pdf_to_excel_ready_text(pdf_file, output_file, dpi=200, batch_size=10)
```

## Troubleshooting

### Windows

- **"Tesseract not found"**: Update the path in the `configure_paths()` function to match your Tesseract installation
- **"Poppler not found"**: Update the `poppler_path` in the `configure_paths()` function to match where you extracted Poppler

### Mac

- **"Tesseract not found"**: Run `brew install tesseract` or update the path in the `configure_paths()` function
- **"Poppler not found"**: Run `brew install poppler`

### Both Platforms

- **Missing items in output**: Check the console output for statistics on missing items
- **Memory errors**: Reduce batch size or DPI in the function call at the bottom of `pdftotext.py`
- **Incorrect column splitting**: Verify that items contain unit markers (EA, PC, LB, etc.)

## Limitations

- Only processes items that contain unit measurements (EA, PC, LB, OZ, PK)
- OCR accuracy depends on PDF quality (300 DPI recommended)
- Very large numbers (1000+) may have OCR errors with commas
- Requires consistent formatting in the original PDF

## Support

If items are missing from the output, check:

1. That they contain unit markers (EA, PC, etc.)
2. The console output for missing item numbers
3. PDF quality and resolution
4. That item numbers are followed by periods (e.g., "175.")

## Example Output

### Raw Output File (`inventory_output.txt`)

```
Number|Description|Qty|Estimate Amount|Taxes|Replacement Cost Total|Age / Cond. / Life|Less Depreciation|Actual Cash Value|Paid|Estimated Remaining
1|DYPASA, Garbage Can 1.6 Gal with Foot Pedal - Stainless Steel|1.00 EA|$95.00|$7.96|$102.96|2 y/ Avg. /10 y|-$20.59|$82.37|$0.00|$20.59
2|LuxenHome, 2 Pieces Metal Wall Art Multi-Color Circles Garnet|1.00 EA|$78.99|$6.62|$85.61|1.75 y/ Avg. /10 y|-$14.98|$70.63|$0.00|$14.98
175|Aeropostale, Lightweight Quilted Puffer Vest|2.00 EA|$109.90|$4.81|$114.71|1.75 y/ Avg. /8 y|-$25.09|$89.62|$0.00|$25.09
```

### After Importing to Excel

| Number | Description                                                   | Qty     | Estimate Amount | Taxes | Replacement Cost Total | Age / Cond. / Life | Less Depreciation | Actual Cash Value | Paid  | Estimated Remaining |
| ------ | ------------------------------------------------------------- | ------- | --------------- | ----- | ---------------------- | ------------------ | ----------------- | ----------------- | ----- | ------------------- |
| 1      | DYPASA, Garbage Can 1.6 Gal with Foot Pedal - Stainless Steel | 1.00 EA | $95.00          | $7.96 | $102.96                | 2 y/ Avg. /10 y    | -$20.59           | $82.37            | $0.00 | $20.59              |
| 2      | LuxenHome, 2 Pieces Metal Wall Art Multi-Color Circles Garnet | 1.00 EA | $78.99          | $6.62 | $85.61                 | 1.75 y/ Avg. /10 y | -$14.98           | $70.63            | $0.00 | $14.98              |
| 175    | Aeropostale, Lightweight Quilted Puffer Vest                  | 2.00 EA | $109.90         | $4.81 | $114.71                | 1.75 y/ Avg. /8 y  | -$25.09           | $89.62            | $0.00 | $25.09              |

---

**Note**: This tool is designed for inventory PDFs with specific formatting. Results may vary with different document types.

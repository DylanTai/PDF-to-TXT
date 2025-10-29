# PDF to Excel OCR Converter (AWS Textract)

**Last Updated: October 2025**

## Overview

This project converts insurance trade summary PDFs into Excel-ready text files. Each page is preprocessed with OpenCV, passed through **Amazon Textract** for OCR, and parsed into pipe-delimited columns that import cleanly into Excel.

The tool is designed for trade summary documents with:

- Section headers with 3-letter category codes (e.g., AMA, APM, APS)
- Item descriptions that may wrap onto multiple lines
- Quantities with units (`EA`, `PC`, `LB`, etc.)
- Four financial columns: Replacement Cost Total, ACV, Non-Recoverable Depreciation, Max Additional Amount Available
- Section totals summarized at the end

By parsing this format, we can import the OCR results straight into Excel with **Data → Text to Columns** while preserving category groupings and totals.

### Highlights

- **Interactive menu** with automatic PDF file detection
- Three processing modes: process all pages, first N pages, or view raw OCR output
- Converts multi-page PDFs in batches to conserve memory
- Auto-selects DPI based on enhancement level (`low`, `medium`, `high`)
- Applies light watermark suppression and noise cleanup to boost text clarity
- Gently deskews, crops, and balances brightness/contrast without over-processing
- Uses Textract's synchronous `DetectDocumentText` API per page
- Detects section headers and categorizes items by type
- Collects section totals and appends them at the end
- Supports custom delimiters (pipe, comma, tab, etc.)
- Outputs to `outputs/` folder for easy organization

> ⚠️ **AWS Costs & Quotas**: Textract billing is per page. Monitor AWS usage and limit IAM permissions to only what you need.

### How It Works

1. **Image preparation** – `pdf2image` (Poppler) renders each PDF page; OpenCV then deskews, trims borders, upsamples if needed, removes faint watermarks, and applies light denoising.
2. **OCR** – The cleaned image is sent to Textract (`DetectDocumentText`). Pre-processing speeds up recognition and avoids oversized payloads.
3. **Parsing** – `process_trade_summary` walks Textract's raw lines, detects 3-letter section codes, stitches wrapped descriptions, interprets ordered currency columns, and tags each item with its category.
4. **Totals collection** – Section totals are extracted and compiled into a summary at the end of the output.
5. **Output** – The final text file uses your chosen delimiter (default: `|`) and is saved to `outputs/<pdf_name>_output.txt`.

### Output Format

```
Description|Line Item Qty|Repl. Cost Total|ACV|Non-Rec. Deprec.|Max Addl. Amt Avail.|Type
AMA --- AUTOMOTIVE & MOTORCYCLE ACC.
Maxboost, Magnetic air-vent car mount|1.00 EA|$14.08|$8.44|$0.00|$5.64|AMA
...

TOTALS
TOTAL AUTOMOTIVE & MOTORCYCLE ACC. (AMA)||$97.47|$45.05|$0.00|$52.42|
```

- **Description** includes continuation text and may wrap from the PDF
- **Line Item Qty** retains the unit (EA, PC, etc.)
- **Type** column shows the 3-letter category code for filtering/grouping
- **TOTALS** section at the end summarizes each category

## Directory Structure

```
251024_PDF-Reader/
├── src/                      # Python source files
│   ├── read_some.py         # Core processing functions
│   └── read_raw.py          # Raw OCR viewer functions
├── pdf/                      # Place your PDF files here
│   ├── .gitkeep             # Keeps folder in git
│   └── Example.pdf          # (ignored by git)
├── outputs/                  # Generated output files (auto-created)
│   └── .gitkeep             # Keeps folder in git
├── venv/                     # Python virtual environment
├── .env                      # AWS credentials (not in git)
├── .gitignore                # Git ignore rules
├── run.py                    # Main entry point
├── requirements.txt          # Python dependencies
└── README.md
```

## Files Included

- **`run.py`** – Main entry point with interactive menu for all processing modes
- **`src/read_some.py`** – Core processing functions for batch PDF processing and OCR
- **`src/read_raw.py`** – Raw Textract OCR output viewer for debugging

**Important**:

- The main script is `run.py` in the project root
- PDF files should be placed in the `pdf/` directory
- Output files are saved to the `outputs/` directory
- Both `pdf/` and `outputs/` folders are tracked in git, but their contents are ignored

## Requirements

### Common

- Python 3.9+
- AWS account with Textract permissions
- AWS credentials configured locally (see **AWS Setup**)
- Poppler utilities (required by `pdf2image`)
- Python packages: `boto3`, `botocore`, `pdf2image`, `pillow`, `opencv-python`, `numpy`

### Setup Instructions

#### macOS

1. Install Poppler (required for PDF processing):

```bash
brew install poppler
```

2. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install Python dependencies:

```bash
pip install -r requirements.txt
```

Homebrew places Poppler on your PATH automatically.

#### Windows

1. Install Poppler for Windows and note the `poppler\Library\bin` path (default assumed is `C:\poppler\Library\bin`).
2. Create and activate virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate
```

3. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

Update `configure_environment()` in `src/read_some.py` if Poppler is installed elsewhere.

### Using the Virtual Environment

**Always activate the virtual environment before running any scripts:**

macOS/Linux:

```bash
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

**When you're done, deactivate the environment:**

```bash
deactivate
```

## AWS Setup

1. Create an IAM user/role with Textract permissions (e.g. `AmazonTextractFullAccess`, or a custom least-privilege policy).
2. Generate access keys if you are using an IAM user.
3. Configure credentials locally:

```bash
aws configure
# or set environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_REGION=us-east-1   # choose any Textract-supported region
```

Textract regions include `us-east-1`, `us-west-2`, `eu-west-1`, and others. The script defaults to `us-east-1` when no region is provided.

### Local .env Support

- Copy the sample: `cp .env.example .env`
- Edit `.env` with your AWS access key, secret key, optional session token, and preferred region.
- Keep the `.env` file out of source control (already covered by `.gitignore` if you add it there).
- The scripts automatically load `.env` on startup. If `python-dotenv` is installed the file is parsed with it; otherwise a lightweight built-in loader handles simple `KEY=VALUE` pairs.

### Installing Dependencies with `requirements.txt`

- Activate your virtual environment, then run `pip install -r requirements.txt`.
- This installs `boto3`, `botocore`, `numpy`, `opencv-python`, `pdf2image`, `pillow`, and `python-dotenv`, which cover all imports used by the scripts.
- Re-run the command whenever the file changes to keep your environment up to date.

## Usage

### Quick Start

1. **Activate your virtual environment:**

   ```bash
   source venv/bin/activate  # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

2. **Place your PDF files in the `pdf/` directory**

3. **Run the main script:**

   ```bash
   python3 run.py
   ```

4. **Follow the interactive prompts:**
   - Choose processing mode (1, 2, or 3)
   - Select PDF file from numbered list
   - Configure options (pages, enhancement level, delimiter)

### Processing Modes

The tool offers three processing modes:

#### 1. **read_all** - Process All Pages

Processes every page of the selected PDF and outputs formatted text ready for Excel import.

**Best for:** Final conversion of complete documents

**Prompts:**

- PDF selection (automatic list)
- OCR enhancement level (`low`, `medium`, `high`)
- Column delimiter (default: `|`)

**Output:**
- Section format: `outputs/<pdf_name>_all_section.txt`
- Name format: `outputs/<pdf_name>_all_name.txt` (if selected)

**Features:**

- Batch processing (20 pages at a time) to conserve memory
- Automatically detects section headers (e.g., "AMA --- AUTOMOTIVE & MOTORCYCLE ACC.")
- Tags each item with its section code in the Type column
- Collects section totals and adds them at the end under "TOTALS"
- Supports custom delimiters

#### 2. **read_some** - Process First N Pages

Processes only the first N pages starting from page 1.

**Best for:** Testing OCR quality or processing partial documents

**Prompts:**

- PDF selection (automatic list)
- Number of pages to process
- OCR enhancement level (`low`, `medium`, `high`)
- Column delimiter (default: `|`)

**Output:**
- Section format: `outputs/<pdf_name>_some_<N>_section.txt`
- Name format: `outputs/<pdf_name>_some_<N>_name.txt` (if selected)

**Features:**

- Same formatting and parsing as read_all
- Faster processing for testing
- Useful for validating settings before processing large PDFs

#### 3. **read_raw** - View Raw OCR Output

Shows the raw, unformatted Textract OCR output from the first N pages.

**Best for:** Understanding document structure, debugging parsers

**Prompts:**

- PDF selection (automatic list)
- Number of pages to process
- OCR enhancement level (`low`, `medium`, `high`)

**Output:** `outputs/<pdf_name>_raw_<N>_section.txt`

**Features:**

- No parsing or formatting applied
- See exactly what Textract returns
- Console preview of first 2000 characters
- Useful for writing custom parsers

### Enhancement Levels

Choose the appropriate enhancement level based on your PDF quality:

- **`low`** – 200 DPI, light preprocessing (fastest, good for high-quality PDFs)
- **`medium`** – 300 DPI, denoising + sharpening (recommended default)
- **`high`** – 400 DPI, aggressive preprocessing (best for noisy scans, slowest)

### Output Formats

You can choose how to organize the output data:

- **Section format** (`_section.txt`) - Items organized by category sections with section headers (e.g., "AMA --- AUTOMOTIVE & MOTORCYCLE ACC.")
- **Name format** (`_name.txt`) - Items only, no section headers, all items in one list
- **Both** - Creates both format files

The section format is useful for reviewing items by category. The name format is useful for importing all items as a flat list without category divisions.

### Example Session

```bash
$ python3 run.py

============================================================
PDF PROCESSING TOOL
============================================================

Available options:
  1. read_all    - Process all pages of a PDF
  2. read_some   - Process first N pages of a PDF
  3. read_raw    - Show raw Textract output from first N pages
============================================================

Enter your choice (1, 2, or 3): 1

============================================================
AVAILABLE PDF FILES:
============================================================
  1. Example.pdf (1.43 MB)
  2. Document.pdf (2.87 MB)
============================================================

Select a PDF (1-2): 1

✓ Selected: Example.pdf

OCR Enhancement Levels:
  'low'    - Basic enhancement, DPI: 200 (fastest)
  'medium' - Balanced enhancement, DPI: 300 (recommended)
  'high'   - Aggressive enhancement, DPI: 400 (best quality)

Select enhancement level (low/medium/high) [default: medium]:

Using default: medium

Column Delimiter:
  Press Enter for default pipe delimiter: |
  Or enter your preferred delimiter (e.g., comma, tab, etc.)

Delimiter [default: |]:

Processing ALL pages from: /path/to/pdf/Example.pdf
Enhancement level: medium
Delimiter: '|'

[Processing output...]

✓✓✓ SUCCESS! ✓✓✓

File saved: outputs/Example_all_section.txt
```

## Importing into Excel

1. Open the output file (e.g., `Example_all_section.txt` or `Example_all_name.txt`).
2. Copy all content and paste into Excel (cell `A1`).
3. Use **Data → Text to Columns**.
4. Choose **Delimited**, tick **Other**, and set the delimiter to `|` (or your chosen delimiter).
5. Finish to split values into columns.

## Troubleshooting & Tips

- **AWS credentials not found**: Ensure `aws sts get-caller-identity` succeeds in the same environment. The script prints the Textract region in use.
- **Textract 5 MB limit**: The script compresses large images, but if a page still exceeds the limit, drop the enhancement level or reduce DPI.
- **Poppler missing**: Install Poppler (Homebrew on macOS) or adjust the Windows path in `configure_environment()` in `src/read_some.py`.
- **No PDF files found**: Make sure your PDF files are in the `pdf/` directory and have the `.pdf` extension.
- **Throughput**: Textract synchronous calls are rate limited. For very large PDFs consider batching runs or using the asynchronous Textract APIs.
- **Costs**: Every page scanned incurs charges. Test with smaller documents first and clean up IAM users/keys when finished.

## Customisation

- Modify `process_trade_summary()` in `src/read_some.py` to parse different document formats
- Adjust preprocessing parameters in `preprocess_image_for_ocr()` for different image quality requirements
- Change batch size in `pdf_to_trade_summary()` (default: 20 pages) to balance memory usage and performance
- For huge PDFs, consider uploading to S3 and swapping in Textract's asynchronous `StartDocumentTextDetection` workflow

## Git Configuration

The repository is configured to track the `pdf/` and `outputs/` folders but ignore their contents:

- `.gitkeep` files ensure empty folders are tracked
- PDF files and output files are ignored to avoid bloating the repository
- Add important output files manually with `git add -f` if needed

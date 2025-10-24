# PDF to Excel OCR Converter (AWS Textract)

**Last Updated: October 2025**

## Overview

This project turns inventory-style PDF documents into Excel-ready text files. Each page is preprocessed with OpenCV, passed through **Amazon Textract** for OCR, and parsed into pipe-delimited columns that drop cleanly into Excel.

Highlights:

- Converts multi-page PDFs in batches to conserve memory
- Auto-selects DPI based on enhancement level (`low`, `medium`, `high`)
- Applies watermark removal, denoising, and sharpening to boost text clarity
- Uses Textract’s synchronous `DetectDocumentText` API per page
- Detects line items by searching for unit markers (EA, PC, LB, OZ, PK, etc.)
- Produces `Number|Description|Qty|Estimate Amount|...` output ready for “Text to Columns”

> ⚠️ **AWS Costs & Quotas**: Textract billing is per page. Monitor AWS usage and limit IAM permissions to only what you need.

## Files Included

- `pdftotext.py` – Batch processor that writes `<pdf_name>_output.txt`
- `pdftotext_test.py` – Convenience script to process only the first and last page and save intermediate artefacts under `test_files/`

## Requirements

### Common

- Python 3.9+
- AWS account with Textract permissions
- AWS credentials configured locally (see **AWS Setup**)
- Poppler utilities (required by `pdf2image`)
- Python packages: `boto3`, `botocore`, `pdf2image`, `pillow`, `opencv-python`, `numpy`

### macOS

```bash
brew install poppler python
python3 -m venv .venv
source .venv/bin/activate
pip install boto3 botocore pdf2image pillow opencv-python numpy
```

Homebrew places Poppler on your PATH automatically.

### Windows

1. Install Poppler for Windows and note the `poppler\Library\bin` path (default assumed in the script is `C:\poppler\Library\bin`).
2. Install Python 3 and create a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install boto3 botocore pdf2image pillow opencv-python numpy
```

Update `configure_environment()` in `pdftotext.py` if Poppler lives elsewhere.

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

## Usage

### Full Run (`pdftotext.py`)

1. Place the target PDF in this directory.
2. Activate your virtual environment (if applicable).
3. Run:

```bash
python pdftotext.py    # or python3 on macOS/Linux
```

4. Enter the PDF file name when prompted (e.g. `inventory.pdf`).
5. Pick an enhancement level:
   - `low` – 200 DPI, light preprocessing (fastest)
   - `medium` – 300 DPI, denoising + sharpening (default)
   - `high` – 400 DPI, aggressive preprocessing (best for noisy scans)
6. Each page is uploaded to Textract; progress and timing print to the console.
7. The output `<pdf_name>_output.txt` appears beside the original PDF.

### Test Mode (`pdftotext_test.py`)

Quickly inspect OCR quality by processing only the first and last page.

```bash
mkdir -p test_files
python pdftotext_test.py
```

Outputs:

- `test_files/preprocessed_images/page_XXXX_original.png`
- `test_files/preprocessed_images/page_XXXX_preprocessed.png`
- `test_files/raw_ocr_text.txt`
- `test_files/formatted_output.txt`

You can inject a custom Textract client (for example, with retries or a Stubber) by calling `test_pdf_processing(..., textract_client=my_client)`.

## Importing into Excel

1. Open `<pdf_name>_output.txt`.
2. Copy all content and paste into Excel (cell `A1`).
3. Use **Data → Text to Columns**.
4. Choose **Delimited**, tick **Other**, and set the delimiter to `|`.
5. Finish to split values into columns.

## Troubleshooting & Tips

- **AWS credentials not found**: Ensure `aws sts get-caller-identity` succeeds in the same environment. The script prints the Textract region in use.
- **Textract 5 MB limit**: The script compresses large images, but if a page still exceeds the limit, drop the enhancement level or reduce DPI.
- **Poppler missing**: Install Poppler (Homebrew on macOS) or adjust the Windows path in `configure_environment()`.
- **Throughput**: Textract synchronous calls are rate limited. For very large PDFs consider batching runs or using the asynchronous Textract APIs.
- **Costs**: Every page scanned incurs charges. Test with smaller documents first and clean up IAM users/keys when finished.

## Customisation

- Pass a pre-configured client into `pdf_to_excel_ready_text(..., textract_client=...)` to add retries, metrics, or take advantage of AWS profiles.
- Adjust `process_for_excel` if your downstream data model differs.
- For huge PDFs, consider uploading to S3 and swapping in Textract’s asynchronous `StartDocumentTextDetection` workflow.

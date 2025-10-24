# PDF to Excel OCR Converter (AWS Textract)

**Last Updated: October 2025**

## Overview

This project turns insurance-style “Line Item Detail” statements into Excel-ready text files. Each page is preprocessed with OpenCV, passed through **Amazon Textract** for OCR, and parsed into pipe-delimited columns that drop cleanly into Excelx The sample that I originally had was a 227-page contents inventory where every row contains the fields we extract:

- an item number (`1.` … `5,295.`)
- description text that may wrap onto multiple lines
- a unit (`EA`, `PC`, `LB`, etc.) plus quantity
- six financial columns (Estimate Amount → Estimated Remaining)
- notation for depreciation, RC benefit policy limits, or “DUPLICATE” markers
- page numbers the adjuster included in the footer

By mirroring that format we can import the OCR results straight into Excel with **Data → Text to Columns** and keep the document context (including which page a line came from) for auditing.

Highlights:

- Converts multi-page PDFs in batches to conserve memory
- Auto-selects DPI based on enhancement level (`low`, `medium`, `high`)
- Applies light watermark suppression and noise cleanup to boost text clarity
- Gently deskews, crops, and balances brightness/contrast without over-processing
- Uses Textract’s synchronous `DetectDocumentText` API per page
- Detects line items by searching for unit markers (EA, PC, LB, OZ, PK, etc.)
- Produces `Number|Description|Qty|Estimate Amount|...|Page Number` output ready for “Text to Columns”
- Preserves Textract `DUPLICATE` flags and maps them to safe zero-dollar values

> ⚠️ **AWS Costs & Quotas**: Textract billing is per page. Monitor AWS usage and limit IAM permissions to only what you need.

### How It Works

1. **Image preparation** – `pdf2image` (Poppler) renders each PDF page; OpenCV then deskews, trims borders, upsamples if needed, removes faint watermarks, and applies light denoising.
2. **OCR** – The cleaned image is sent to Textract (`DetectDocumentText`). Textract can cope with the faint diagonal watermark, but pre-processing speeds it up and avoids oversized payloads.
3. **Parsing** – `process_for_excel` walks Textract’s raw lines, reconstructs each inventory row, stitches wrapped descriptions (e.g., “Hammermill, Printer Paper …” plus “5 x 11-1 Ream (new)”), interprets ordered currency columns, and tracks the current PDF page.
4. **Duplicate handling** – Textract sometimes labels repeated SKUs with a standalone `DUPLICATE` line. Those items are preserved with blank numeric values, `Actual Cash Value=DUPLICATE`, and both `Paid` and `Estimated Remaining` forced to `$0.00` so reviewers can spot them quickly.
5. **Output** – The final text file uses pipes (`|`) and includes the source page number so you can jump back to the PDF during reconciliation.

### Output Columns

```
Number|Description|Qty|Estimate Amount|Taxes|Replacement Cost Total|Age / Cond. / Life|Less Depreciation|Actual Cash Value|Paid|Estimated Remaining|Page Number
```

- **Description** may include continuation text from the PDF (e.g., “5 x 11-1 Ream (new)”).
- **Qty** retains the unit; if Textract marks an item as `DUPLICATE`, quantity and dollar fields are blank.
- **Actual Cash Value / Paid / Estimated Remaining** show `DUPLICATE`, `$0.00`, `$0.00` for repeated items so downstream formulas stay safe.
- **Page Number** records where the row was read (useful when cross-checking with `mom.pdf`).

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
pip install -r requirements.txt
```

Homebrew places Poppler on your PATH automatically.

### Windows

1. Install Poppler for Windows and note the `poppler\Library\bin` path (default assumed in the script is `C:\poppler\Library\bin`).
2. Install Python 3 and create a virtual environment.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
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

### Local .env Support

- Copy the sample: `cp .env.example .env`
- Edit `.env` with your AWS access key, secret key, optional session token, and preferred region.
- Keep the `.env` file out of source control (already covered by `.gitignore` if you add it there).
- Both scripts automatically load `.env` on startup. If `python-dotenv` is installed the file is parsed with it; otherwise a lightweight built-in loader handles simple `KEY=VALUE` pairs.

### Installing Dependencies with `requirements.txt`

- Activate your virtual environment, then run `pip install -r requirements.txt`.
- This installs `boto3`, `botocore`, `numpy`, `opencv-python`, `pdf2image`, `pillow`, and `python-dotenv`, which cover all imports used by the scripts.
- Re-run the command whenever the file changes to keep your environment up to date.

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
8. Every row includes the original PDF page number as the final column so you can jump back to the source document (helpful when reviewing `mom.pdf`).

### Test Mode (`pdftotext_test.py`)

Quickly inspect OCR quality by processing up to the first 10 and last 10 pages (entire PDF if shorter).

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

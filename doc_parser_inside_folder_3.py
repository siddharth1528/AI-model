import os
import json
import re
import pandas as pd
import requests
from pdf2image import convert_from_path
from PIL import Image
import pytesseract

# Configuration
PDF_DIR = r"C:\Temp\Py_Files_sid\Drawings"  # Folder with drawing PDFs
POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"
NVIDIA_API_KEY = "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"
NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "meta/llama-4-maverick-17b-128e-instruct"


def extract_text_from_pdf(pdf_path):
    print("  Converting PDF to images...")
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    all_text = []

    for i, page in enumerate(pages):
        print(f"  OCR on Page {i + 1}")
        gray = page.convert("L")  # grayscale
        text = pytesseract.image_to_string(gray, config='--psm 6')
        all_text.append(f"--- Page {i + 1} ---\n{text.strip()}")

    return "\n\n".join(all_text)


def build_prompt(ocr_text):
    return f"""
You are an AI assistant that extracts structured data from OCR-scanned engineering drawing documents.

Extract and return the following fields from the OCR text. Look for common variations in field labels:

1. **"Drawing Number"** - look for labels such as "DRAWING NO.", "DRG. NO", "DRG NO", "DWG. NO.", "DRG_NO", or any similar variations.
2. **"Title"** - may appear as "DRAWING TITLE", "TITLE", or be located near the top or centered in bold font.
3. **"Revision"** - check near labels like "REV", "REVISION", "REV. NO.", or beside the Drawing Number block.
4. **"Date"** - look near Revision or Approval sections; may appear under "DATE", "ISSUE DATE", or adjacent to signatures.
5. **"Description"** - usually found near revision tables or notes on the drawing.

Also, extract any "Version History" table with these columns (if present):
- Version
- Date
- Name
- Description of Review Changes

⚠️ Important Instructions:
- Return **only valid JSON**. No explanations or commentary.
- If a field is missing, return it as an empty string ("").
- For the Version History, return a list of rows under the "Version History" key.
- Ignore any logos, stamps, or graphical elements. Focus only on structured text fields that match engineering drawing conventions.
- If duplicate or conflicting information is present, choose the most complete or clearly formatted version.

Here is the OCR text:
\"\"\"
{ocr_text}
\"\"\"

Respond with JSON only in the format below:

{{
  "Drawing Number": "",
  "Title": "",
  "Revision": "",
  "Date": "",
  "Description": "",
  "Version History": [
    {{
      "Version": "",
      "Date": "",
      "Name": "",
      "Description of Review Changes": ""
    }}
  ]
}}
"""


def call_nvidia_llm(prompt_text):
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    response = requests.post(NVIDIA_ENDPOINT, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']


def extract_json_like_block(text):
    brace_count = 0
    start_index = -1

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_index = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_index != -1:
                return text[start_index:i+1]
    return None


def save_response_to_csv(response_text, fallback_text, output_csv_path):
    print("💾 Saving response to CSV...")

    try:
        extracted = json.loads(response_text)
    except json.JSONDecodeError:
        print("❌ JSON decode failed. Trying to extract JSON block...")
        json_block = extract_json_like_block(response_text)
        if json_block:
            try:
                extracted = json.loads(json_block)
            except json.JSONDecodeError:
                print("⚠️ Extracted JSON block is still invalid. Falling back.")
                extracted = {"Raw Text": fallback_text.strip()}
        else:
            print("⚠️ No JSON-like block found. Saving raw OCR text.")
            extracted = {"Raw Text": fallback_text.strip()}

    # Save metadata
    if "Raw Text" in extracted:
        pd.DataFrame([extracted]).to_csv(output_csv_path, index=False)
        return extracted

    flat_data = {k: v for k, v in extracted.items() if k != "Version History"}
    pd.DataFrame([flat_data]).to_csv(output_csv_path, index=False)

    if "Version History" in extracted and isinstance(extracted["Version History"], list):
        pd.DataFrame(extracted["Version History"]).to_csv(
            output_csv_path.replace(".csv", "_history.csv"), index=False
        )

    return extracted


def process_pdf_file(pdf_path):
    print(f"\n🔍 Processing: {pdf_path}")
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_csv_path = os.path.join(os.path.dirname(pdf_path), f"{pdf_name}_metadata.csv")

    try:
        ocr_text = extract_text_from_pdf(pdf_path)
        prompt = build_prompt(ocr_text)
        llm_response = call_nvidia_llm(prompt)
        result = save_response_to_csv(llm_response, ocr_text, output_csv_path)
        print(f"✅ Extracted metadata for: {pdf_name}")
        return result
    except Exception as e:
        print(f"❌ Failed to process {pdf_path}: {e}")
        return None


def main():
    print(f"📁 Scanning folder: {PDF_DIR}")
    pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDF files found.")
        return

    print(f"📂 Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:
        process_pdf_file(pdf_file)


if __name__ == "__main__":
    main()

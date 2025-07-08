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
    print(" Converting PDF to image...")
    pages = convert_from_path(pdf_path, dpi=300, poppler_path=POPPLER_PATH)
    pages = pages[:10]
    text_blocks = []

    for i, page in enumerate(pages):
        print(f" Running OCR on Page {i+1}")
        gray = page.convert("L")
        text = pytesseract.image_to_string(gray, config='--psm 6 -c preserve_interword_spaces=1')
        text_blocks.append(text.strip())

    return "\n\n".join(text_blocks)


def build_prompt(ocr_text):
    return f"""
You are an AI assistant that extracts structured data from OCR-scanned engineering drawing documents.

Extract and return the following fields from the OCR text. Look for common variations in field labels:

1. **"Drawing Number"** - look for labels like:
   - "DRAWING NO.", "DRG. NO", "DRG NO", "DWG. NO.", "DRG_NO"
   - "DOCUMENT NO.", "DOCUMENT NUMBER", "DOC NO.", "DOC NUMBER"
   - "DRAWING No", "DRAWING #", "DRAWING NUMBER"
   - Also consider values located directly next to "REVISION" blocks, or at the bottom right of the drawing.

2. **"Title"** - look for:
   - Labels like "DRAWING TITLE", "TITLE", or headers near the top
   - Uppercase block text stacked over multiple lines, often centered
   - Prefer the full multi-line block if it appears immediately after "DRAWING TITLE"

3. **"Revision"** - check for:
   - "REV", "REVISION", "REV. NO.", "REV NO", often near Drawing Number block
   - Appears as a single digit or letter (e.g., "00", "A", "C")

4. **"Date"** - look near:
   - Revision or Approval blocks
   - Labels like "DATE", "ISSUE DATE", "APPROVAL DATE", etc.
   - Must be in date format, even partial (e.g., "09-Dec-2020", "2022")

5. **"Description"** - look for:
   - Notes near the revision area or drawing title block
   - Phrases that describe system/component or changes made
   - This may be a short sentence or summary paragraph.

⚠️ Important Instructions:
- Return only **valid minified JSON** (no line breaks, comments, or extra text).
- Use **double quotes** around all field names and values.
- If a field is not found, return it as an empty string (`""`).
- Do not include version history, tables, or drawing stamps.
- Focus strictly on extracting the five fields only.
- Use your best judgment when field labels are ambiguous or missing. Prioritize proximity and layout.


Here is the OCR text:
{ocr_text}

Respond with **only valid minified JSON**, without line breaks or comments. Your output must strictly follow this structure and use double quotes:

{{
  "Drawing Number": "",
  "Title": "",
  "Revision": "",
  "Date": "",
  "Description": "",
  
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


def save_response_to_csv(response_text, fallback_text):
    try:
        extracted = json.loads(response_text)
    except json.JSONDecodeError:
        print("❌ JSON decode failed. Trying to extract JSON block...")
        json_block = extract_json_like_block(response_text)
        if json_block:
            try:
                extracted = json.loads(json_block)
            except json.JSONDecodeError:
                extracted = {}
        else:
            extracted = {}

    filtered_data = {
        "Drawing Number": extracted.get("Drawing Number", ""),
        "Title": extracted.get("Title", ""),
        "Revision": extracted.get("Revision", ""),
        "Date": extracted.get("Date", ""),
        "Description": extracted.get("Description", "")
    }
    return filtered_data

def process_pdf_file(pdf_path):
    try:
        print(f"\n🔍 Processing file: {os.path.basename(pdf_path)}")
        ocr_text = extract_text_from_pdf(pdf_path)
        prompt = build_prompt(ocr_text)
        llm_response = call_nvidia_llm(prompt)

        try:
            extracted = json.loads(llm_response)
        except json.JSONDecodeError:
            print("⚠️ JSON parsing failed. Trying to extract valid block...")
            json_block = extract_json_like_block(llm_response)
            if json_block:
                extracted = json.loads(json_block)
            else:
                print("❌ Could not extract valid JSON.")
                return None

        # Filter only required fields
        fields = ["Drawing Number", "Title", "Revision", "Date", "Description"]
        result = {field: extracted.get(field, "") for field in fields}

        return result

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return None

def main():
    print(f"📁 Scanning directory: {PDF_DIR}")
    pdf_files = [os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]

    if not pdf_files:
        print("⚠️ No PDF files found.")
        return

    print(f"📂 Found {len(pdf_files)} PDF files. Starting extraction...\n")
    all_results = []

    for pdf_file in pdf_files:
        data = process_pdf_file(pdf_file)
        if data:
            data["Filename"] = os.path.basename(pdf_file)
            all_results.append(data)

    if all_results:
        df = pd.DataFrame(all_results)
        output_path = os.path.join(PDF_DIR, "All_Drawings_Metadata.csv")
        df.to_csv(output_path, index=False)
        print(f"\n✅ Combined CSV saved at: {output_path}")
    else:
        print("⚠️ No valid extractions found.")


if __name__ == "__main__":
    main()

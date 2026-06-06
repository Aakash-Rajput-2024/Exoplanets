import os
import sys

def try_extract_text(pdf_path):
    print(f"Analyzing {pdf_path}...")
    # Check what packages are available
    for pkg in ['pypdf', 'fitz', 'pdfplumber', 'PyPDF2']:
        try:
            mod = __import__(pkg)
            print(f"Found {pkg}!")
            if pkg == 'pypdf':
                reader = mod.PdfReader(pdf_path)
                text = reader.pages[0].extract_text()
                return text[:1000]
            elif pkg == 'fitz':
                doc = mod.open(pdf_path)
                text = doc[0].get_text()
                return text[:1000]
            elif pkg == 'pdfplumber':
                with mod.open(pdf_path) as pdf:
                    text = pdf.pages[0].extract_text()
                    return text[:1000]
            elif pkg == 'PyPDF2':
                reader = mod.PdfReader(pdf_path)
                text = reader.pages[0].extract_text()
                return text[:1000]
        except ImportError:
            pass
    
    # If no library is found, try to run a command or print PDF header
    print("No PDF extraction libraries found in python paths. Trying to read binary metadata...")
    try:
        with open(pdf_path, 'rb') as f:
            head = f.read(2048)
            # Find any DOI or title in plain text
            import re
            dois = re.findall(rb'10\.\d{4,9}/[-._;()/:A-Z0-9]+', head, re.IGNORECASE)
            if dois:
                print("Found DOIs in first 2048 bytes:", [d.decode('utf-8', errors='ignore') for d in dois])
            # Print print-ready ASCII characters
            ascii_chars = "".join([chr(b) if 32 <= b < 127 or b in [10, 13] else '.' for b in head])
            print("Ascii snippet:", ascii_chars[:500])
    except Exception as e:
        print("Error reading bytes:", e)
    return None

if __name__ == "__main__":
    path = "/Users/aakashrajput/MachineLearning/Exoplanets/Lit_rew/Causality/make-01-00019-v2.pdf"
    res = try_extract_text(path)
    if res:
        print("--- First 1000 chars of page 1 ---")
        print(res)

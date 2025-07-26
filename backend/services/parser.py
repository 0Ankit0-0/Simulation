import fitz
import docx
import os
from services.ocr import extract_text_from_image

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_from_pdf(file_path)
    elif ext == '.docx':
        return extract_from_docx(file_path)
    elif ext in ['.jpg', '.jpeg', '.png']:
        return extract_text_from_image(file_path)
    else:
        return "Unsupported file type"
    
def extract_from_pdf(file_path):
    try:
        text = ""
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {e}"
    
def extract_from_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return '\n'.join([p.text for p in doc.paragraphs])
    except Exception as e:
        return f"Error extracting text from DOCX: {e}"
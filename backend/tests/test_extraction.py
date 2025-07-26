import os
from services.parser import extract_from_pdf, extract_from_docx
from services.ocr import extract_text_from_image

def test_pdf_extraction():
    pdf_path = os.path.join(os.getcwd(), 'tests', 'samples', 'elle.pdf')
    text = extract_from_pdf(pdf_path)
    assert isinstance(text, str)
    assert len(text.strip()) > 10  # Adjust based on actual file

def test_docx_extraction():
    docx_path = os.path.join(os.getcwd(), 'tests', 'samples', 'AI Courtroom Simulation(Synopsis).docx')
    text = extract_from_docx(docx_path)
    assert isinstance(text, str)
    assert "Example" in text  # Use a known word from the file

def test_ocr_extraction():
    image_path = os.path.join(os.getcwd(), 'tests', 'samples', 'sample.jpeg')
    text = extract_text_from_image(image_path)
    assert isinstance(text, str)
    assert len(text.strip()) > 5  # Adjust based on image content

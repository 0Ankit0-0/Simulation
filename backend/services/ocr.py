from PIL import Image
import pytesseract as pt
import cv2
import numpy as np
import re

pt.pytesseract.tesseract_cmd = r"E:\Ankit\tesseract.exe"


def clean_text(raw_text):
    # Fix common unicode and junk chars
    text = raw_text.replace("\u2014", "-").replace("\u00bb", ">>")
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_text_from_image(image_path):
    try:
        # 1. Preprocess
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        pil_img = Image.fromarray(processed)

        # 2. OCR
        raw_text = pt.image_to_string(pil_img)

        # 3. Clean
        cleaned = clean_text(raw_text)

        # 4. Basic classification
        result = {"raw_text": raw_text, "cleaned_text": cleaned, "analysis": {}}

        if "attendance" in cleaned.lower():
            result["analysis"]["type"] = "notice"
            result["analysis"]["meaning"] = "Likely a rule about attendance"
        elif "enter" in cleaned.lower() and "leave" in cleaned.lower():
            result["analysis"]["type"] = "timeline/flow"
            result["analysis"]["meaning"] = "Possibly a process chart"

        return result

    except Exception as e:
        return {"error": f"Failed to extract text: {str(e)}"}

import cv2
import pytesseract
import re
from src.ersgan import ESRGAN, apply_clahe

class PlateOCR:
    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        self.tesseract_config = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 8 --oem 3'
        self.pattern = re.compile(r'^[A-Z0-9]{1,8}$')
        self.esrgan = ESRGAN()

    def preprocess_ocr(self, img):
        enhanced = self.esrgan.enhance(img)
        enhanced_clahe = apply_clahe(enhanced)
        gray = cv2.cvtColor(enhanced_clahe, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh, enhanced_clahe

    def perform_ocr(self, img):
        preprocessed, enhanced = self.preprocess_ocr(img)
        raw_text = pytesseract.image_to_string(preprocessed, config=self.tesseract_config)
        return raw_text.strip(), enhanced

    def valid_plate(self, text):
        if not text:
            return False
        text = text.upper()
        return bool(self.pattern.match(text.upper()))
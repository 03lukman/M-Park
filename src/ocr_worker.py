import threading
import queue
from src.ocr import PlateOCR

class OcrWorker(threading.Thread):
    def __init__(self, ocr_queue: queue.Queue, result_queue: queue.Queue):
        super().__init__(daemon=True)
        self.ocr_queue = ocr_queue
        self.result_queue = result_queue
        self.ocr = PlateOCR()
        self.running = True

    def run(self):
        while self.running:
            try:
                is_entry, crop_img = self.ocr_queue.get(timeout=1)
                plate_text, enhanced_plate = self.ocr.perform_ocr(crop_img)
                valid = self.ocr.valid_plate(plate_text)
                if not valid:
                    plate_text = None
                self.result_queue.put((is_entry, plate_text, enhanced_plate))
            except queue.Empty:
                continue



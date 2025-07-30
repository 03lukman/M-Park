import torch
from ultralytics import YOLO
from src.ersgan import ESRGAN, apply_clahe

class PlateDetector:
    def __init__(self, model_path='model/platIndo2_YOLOv8s.pt', conf_threshold=0.8, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)
        self.conf_threshold = conf_threshold

        self.esrgan = ESRGAN()

    def detect(self, frame, roi=None):
        x1_roi, y1_roi = 0, 0  #Inisialisasi default
        if roi:
            x1_roi, y1_roi, x2_roi, y2_roi = roi
            frame_for_detection = frame[y1_roi:y2_roi, x1_roi:x2_roi]
        else:
            frame_for_detection = frame

        results = self.model(frame_for_detection, conf=self.conf_threshold, verbose=False)[0]

        plates = []
        for box in results.boxes:
            conf = box.conf.cpu().item()
            if conf < self.conf_threshold:
                continue

            x1, y1, x2, y2 = box.xyxy.cpu().numpy().astype(int)[0]

            if roi:
                x1 += x1_roi
                y1 += y1_roi
                x2 += x1_roi
                y2 += y1_roi

            cropped_plate = frame[y1:y2, x1:x2]
            enhanced_plate = self.esrgan.enhance(cropped_plate)
            enhanced_plate = apply_clahe(enhanced_plate)

            plates.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "enhanced_plate": enhanced_plate,
                "cropped_plate": cropped_plate
            })

        return plates
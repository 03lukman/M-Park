import torch
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path='model/platIndo2_yolov8s.pt', conf_threshold=0.8, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(model_path).to(self.device)
        self.conf_threshold = conf_threshold

    def detect(self, frame, roi=None):
        x1_roi, y1_roi = 0, 0
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
            if cropped_plate.size == 0:
                continue  # Skip jika crop invalid

            plates.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf,
                "cropped_plate": cropped_plate
            })

        return plates
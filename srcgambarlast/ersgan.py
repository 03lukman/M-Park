import os
import cv2
import sys
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ESRGAN_PATH = os.path.join(BASE_DIR, '../ESRGAN')
if ESRGAN_PATH not in sys.path:
    sys.path.append(ESRGAN_PATH)

try:
    import RRDBNet_arch as rrdbnet
except ImportError as e:
    raise ImportError("Gagal import modul RRDBNet_arch dari folder ESRGAN.") from e

class ESRGAN:
    def __init__(self, model_path=None, device=None):
        if model_path is None:
            model_path = os.path.join(BASE_DIR, '../ESRGAN/models/RRDB_ESRGAN_x4.pth')

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = rrdbnet.RRDBNet(3, 3, 64, 23, gc=32)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model ESRGAN tidak di temukan di path: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        self.model.to(self.device)

    def enhance(self, img):
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))  # BGR ke RGB
        img = torch.from_numpy(img).unsqueeze(0).to(self.device).float()

        with torch.no_grad():
            output = self.model(img).squeeze(0).float().cpu().clamp_(0, 1).numpy()

        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # RGB ke BGR
        output = (output * 255.0).round().astype(np.uint8)
        return output

def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

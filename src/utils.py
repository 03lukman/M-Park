import numpy as np

def format_duration(td):
    total_seconds = int(td.total_seconds())
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    if days > 0:
        return f"{days} hari {hours:02d}:{minutes:02d}"
    else:
        return f"{hours:02d}:{minutes:02d}"

def is_same_crop(crop1, crop2, threshold=0.2):
    if crop1 is None or crop2 is None:
        return False
    if crop1.shape != crop2.shape:
        return False

    diff = np.sum(np.abs(crop1.astype(np.int16) - crop2.astype(np.int16)))
    total_pixels = crop1.size
    diff_ratio = diff / total_pixels

    return diff_ratio < threshold

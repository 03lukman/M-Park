from datetime import datetime, timedelta

class PlateCache:
    def __init__(self, expiry_seconds=60):
        self.cache = {}
        self.expiry_seconds = expiry_seconds

    def is_recent(self, plate_text):
        now = datetime.now()
        for cached_plate, last_time in self.cache.items():
            if (plate_text in cached_plate) or (cached_plate in plate_text):
                if now - last_time < timedelta(seconds=self.expiry_seconds):
                    return True
        return False

    def update(self, plate_text):
        self.cache[plate_text] = datetime.now()

    def clear(self):
        now = datetime.now()
        to_remove = [plate for plate, last_time in self.cache.items()
                     if now - last_time > timedelta(seconds=self.expiry_seconds)]
        for plate in to_remove:
            del self.cache[plate]

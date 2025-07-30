import time
import threading

class Camera(threading.Thread):
    def __init__(self, cap_getter, frame_setter, error_callback, fps=20):
        super().__init__(daemon=True)
        self.cap_getter = cap_getter
        self.frame_setter = frame_setter
        self.error_callback = error_callback
        self.running = True
        self.fps = fps
        self.frame_delay = 1.0 / fps

        self.cap = self.cap_getter()
        self.frame_count = 0
        self.start_time = time.perf_counter()
        self.current_fps = 0

    def run(self):
        while self.running:
            start_time = time.time()

            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Gagal membaca frame")
                self.error_callback()
                self.running = False  # stop thread
                return  # keluar dari run

            self.frame_setter(frame)
            time.sleep(self.frame_delay)

            # Update penghitung frame
            self.frame_count += 1
            elapsed_total = time.time() - self.start_time

            # Update FPS setiap 1 detik
            if elapsed_total >= 1.0:
                self.current_fps = self.frame_count / elapsed_total
                self.frame_count = 0
                self.start_time = time.time()

            elapsed = time.time() - start_time
            sleep_time = self.frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()

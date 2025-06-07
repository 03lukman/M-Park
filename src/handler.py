import cv2
import torch
import time
import queue
from datetime import datetime, timedelta
from PIL import Image, ImageTk
from src.db import ParkingDatabase
from src.deteksi import PlateDetector
from src.cache import PlateCache
from src.utils import format_duration, is_same_crop
from src.ocr_worker import OcrWorker
from src.camera import Camera
class MParkingApp:

    def __init__(self, root):
        self.root = root
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.detector = PlateDetector(device=self.device)
        self.db = ParkingDatabase()

        self.cap_in = None
        self.cap_out = None

        self.frame_width = 640
        self.frame_height = 360

        self.prev_time_in = None
        self.prev_time_out = None
        self.camera_thread_in = None
        self.camera_thread_out = None
        self.last_ocr_crop_in = None
        self.last_ocr_text_in = None
        self.last_ocr_time_in = 0
        self.ocr_skipped_in = False

        self.last_ocr_crop_out = None
        self.last_ocr_text_out = None
        self.last_ocr_time_out = 0
        self.ocr_skipped_out = False

        self.ocr_interval_seconds = 3

        self.plate_cache_in = PlateCache()
        self.plate_cache_out = PlateCache()

        self.roi_manager = None

        # UI widgets set by ui.py
        self.cuda_label = None
        self.entry_in = None
        self.entry_out = None
        self.canvas_in = None
        self.canvas_out = None
        self.preview_crop_in = None
        self.preview_crop_in_img = None
        self.preview_crop_out = None
        self.preview_crop_out_img = None
        self.fps_in_label = None
        self.fps_out_label = None
        self.tree = None
        self.imgtk_in = None
        self.imgtk_out = None
        self.running = True

        # Queues for OCR communication
        self.ocr_queue = queue.Queue()
        self.ocr_result_queue = queue.Queue()

        # Start OCR worker thread
        self.ocr_worker = OcrWorker(self.ocr_queue, self.ocr_result_queue)
        self.ocr_worker.start()

        # Schedule OCR result checking
        self.root.after(100, self.check_ocr_result)

        self.update_table()

    def set_camera_sources(self, source_in, source_out):
        def parse_source(src):
            if src.isdigit():
                return int(src)
            return src

        if source_in:
            new_cap_in = cv2.VideoCapture(parse_source(source_in))
            if new_cap_in.isOpened():
                if self.cap_in is not None:
                    self.cap_in.release()
                    time.sleep(0.5)
                self.cap_in = new_cap_in
                self.camera_thread_in = Camera(
                    cap_getter=lambda: self.cap_in,
                    frame_setter=lambda frame: setattr(self, 'latest_frame_in', frame),
                    error_callback=lambda: self.display_message_on_canvas(True, "Tidak bisa baca frame Kamera Masuk")
                )

                self.camera_thread_in.start()
                print(f"[INFO] Kamera Masuk di-set ke: {source_in}")
                if self.canvas_in:
                    self.canvas_in.delete("all")
            else:
                print(f"[WARNING] Gagal buka Kamera Masuk: {source_in}")

        if source_out:
            new_cap_out = cv2.VideoCapture(parse_source(source_out))
            if new_cap_out.isOpened():
                if self.cap_out is not None:
                    self.cap_out.release()
                    time.sleep(0.5)
                self.cap_out = new_cap_out
                self.camera_thread_out = Camera(
                    cap_getter=lambda: self.cap_out,
                    frame_setter=lambda frame: setattr(self, 'latest_frame_out', frame),
                    error_callback=lambda: self.display_message_on_canvas(False, "Tidak bisa baca frame Kamera Keluar")
                )

                self.camera_thread_out.start()
                print(f"[INFO] Kamera Keluar di-set ke: {source_out}")
                if self.canvas_out:
                    self.canvas_out.delete("all")
            else:
                print(f"[WARNING] Gagal buka Kamera Keluar: {source_out}")
            return

    def check_ocr_result(self):
        try:
            while True:
                is_entry, plate_text, enhanced_plate = self.ocr_result_queue.get_nowait()
                self.handle_ocr_result(is_entry, plate_text, enhanced_plate)
        except queue.Empty:
            pass
        self.root.after(100, self.check_ocr_result)

    def handle_ocr_result(self, is_entry, plate_text, enhanced_plate):
        last_ocr_text_attr = 'last_ocr_text_in' if is_entry else 'last_ocr_text_out'
        last_ocr_time_attr = 'last_ocr_time_in' if is_entry else 'last_ocr_time_out'
        ocr_skipped_flag = 'ocr_skipped_in' if is_entry else 'ocr_skipped_out'

        if plate_text:
            setattr(self, last_ocr_text_attr, plate_text)
            setattr(self, ocr_skipped_flag, False)
            setattr(self, last_ocr_time_attr, time.time())

            plate_cache = self.plate_cache_in if is_entry else self.plate_cache_out
            if not plate_cache.is_recent(plate_text):
                try:
                    if is_entry:
                        self.db.insert_entry(plate_text, datetime.now())
                    else:
                        self.db.update_exit(plate_text)
                    plate_cache.update(plate_text)
                except Exception as e:
                    print(f"[ERROR] Gagal update DB {'masuk' if is_entry else 'keluar'}: {e}")

            preview_label = self.preview_crop_in if is_entry else self.preview_crop_out
            preview_attr = 'preview_crop_in_img' if is_entry else 'preview_crop_out_img'
            if enhanced_plate is not None and preview_label:
                self.update_preview_crop(preview_label, enhanced_plate, preview_attr)

        else:
            setattr(self, ocr_skipped_flag, True)

    def handle_plate_detection(self, det, is_entry, last_ocr_time_attr, last_ocr_crop_attr, last_ocr_text_attr):
        now = time.time()

        bbox = det["bbox"]
        conf = det["confidence"]

        last_ocr_time = getattr(self, last_ocr_time_attr)
        last_ocr_crop = getattr(self, last_ocr_crop_attr)
        last_ocr_text = getattr(self, last_ocr_text_attr)

        do_ocr = False
        if now - last_ocr_time > self.ocr_interval_seconds:
            if last_ocr_crop is None or not is_same_crop(last_ocr_crop, det["cropped_plate"]):
                do_ocr = True

        ocr_skipped_flag = 'ocr_skipped_in' if is_entry else 'ocr_skipped_out'

        if do_ocr:
            self.ocr_queue.put((is_entry, det["cropped_plate"]))
            setattr(self, last_ocr_crop_attr, det["cropped_plate"].copy())
            setattr(self, ocr_skipped_flag, False)
            setattr(self, last_ocr_time_attr, now)

            preview_label = self.preview_crop_in if is_entry else self.preview_crop_out
            preview_attr = 'preview_crop_in_img' if is_entry else 'preview_crop_out_img'
            self.update_preview_crop(preview_label, det["cropped_plate"], preview_attr)

            plate_text = last_ocr_text or "..."
        else:
            plate_text = last_ocr_text or "..."
            if not getattr(self, ocr_skipped_flag):
                setattr(self, ocr_skipped_flag, True)

        return bbox, conf, plate_text

    def process_stream(self, roi_attr, last_ocr_time_attr, last_ocr_crop_attr, last_ocr_text_attr
                      , is_entry):
        frame = getattr(self, 'latest_frame_in' if is_entry else 'latest_frame_out', None)
        roi_val = getattr(self.roi_manager, roi_attr)


        if frame is None:
            self.display_message_on_canvas(is_entry, f"Tidak bisa baca frame Kamera {'Masuk' if is_entry else 'Keluar'}")
            return

        frame_resized = cv2.resize(frame, (self.frame_width, self.frame_height))
        plates = self.detector.detect(frame_resized, roi=roi_val)
        self.draw_roi(frame_resized, roi_val, stream_type='in' if is_entry else 'out')

        for det in plates:
            bbox, conf, plate_text = self.handle_plate_detection(det, is_entry,
                                                                 last_ocr_time_attr,
                                                                 last_ocr_crop_attr,
                                                                 last_ocr_text_attr)
            color = (0, 255, 0) if is_entry else (0, 0, 255)
            cv2.rectangle(frame_resized, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            conf_text = f"{conf:.2f}"
            cv2.putText(frame_resized, conf_text, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


        canvas = self.canvas_in if is_entry else self.canvas_out
        if canvas:
            self.show_frame(frame_resized, canvas)

    def on_close(self):
        self.running = False
        if self.cap_in:
            self.cap_in.release()
        if self.cap_out:
            self.cap_out.release()
        if self.camera_thread_in:
            self.camera_thread_in.stop()
        if self.camera_thread_out:
            self.camera_thread_out.stop()
        self.root.destroy()

    def update_streams(self):
        # Kalau thread kamera sudah berhenti karena error, langsung tampilkan pesan error
        if self.camera_thread_in and not self.camera_thread_in.running:
            self.display_message_on_canvas(True, "Tidak bisa baca frame Kamera Masuk")
        else:
            self.process_stream(
                roi_attr='roi_masuk',
                last_ocr_time_attr='last_ocr_time_in',
                last_ocr_crop_attr='last_ocr_crop_in',
                last_ocr_text_attr='last_ocr_text_in',
                is_entry=True
            )

        if self.camera_thread_out and not self.camera_thread_out.running:
            self.display_message_on_canvas(False, "Tidak bisa baca frame Kamera Keluar")
        else:
            self.process_stream(
                roi_attr='roi_keluar',
                last_ocr_time_attr='last_ocr_time_out',
                last_ocr_crop_attr='last_ocr_crop_out',
                last_ocr_text_attr='last_ocr_text_out',
                is_entry=False
            )
        # Update label FPS dari camera thread langsung (lebih akurat)
        if self.camera_thread_in and self.fps_in_label:
            fps_in = getattr(self.camera_thread_in, 'current_fps', 0)
            self.fps_in_label.config(text=f"FPS Masuk: {fps_in:.1f}")

        if self.camera_thread_out and self.fps_out_label:
            fps_out = getattr(self.camera_thread_out, 'current_fps', 0)
            self.fps_out_label.config(text=f"FPS Keluar: {fps_out:.1f}")

        self.root.after(30, self.update_streams)

    def display_message_on_canvas(self, is_entry, message):
        canvas = self.canvas_in if is_entry else self.canvas_out
        if canvas:
            canvas.delete("all")
            canvas.create_text(self.frame_width // 2, self.frame_height // 2,
                               text=message,
                               fill="white", font=("Helvetica", 16))

    def update_preview_crop(self, label_widget, image_np, attr_name):
        img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize((200, 60))
        photo_img = ImageTk.PhotoImage(img_pil)
        setattr(self, attr_name, photo_img)
        if label_widget:
            label_widget.config(image=photo_img)

    @staticmethod
    def draw_roi(frame, roi, stream_type='in', thickness=2):

        x1, y1, x2, y2 = roi
        color = (255, 0, 0) if stream_type == 'in' else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    def show_frame(self, frame, canvas):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
        if canvas == self.canvas_in:
            self.imgtk_in = imgtk
        else:
            self.imgtk_out = imgtk
        canvas.create_image(0, 0, anchor='nw', image=imgtk)

    def update_table(self):
        records = self.db.fetch_all_entries()
        records.sort(key=lambda x: x.get('waktu_masuk', datetime.min), reverse=True)
        if self.tree:
            self.tree.delete(*self.tree.get_children())

            max_duration = timedelta(hours=3)

            for entry in records:
                plat_nomor = entry.get('plat_nomor', 'Tidak Dikenal')
                if plat_nomor == 'Tidak Dikenal':
                    continue

                waktu_masuk = entry.get('waktu_masuk', '-')
                waktu_keluar = entry.get('waktu_keluar', '-')

                if isinstance(waktu_masuk, datetime):
                    waktu_masuk_str = waktu_masuk.strftime('%H:%M')
                    tanggal_str = waktu_masuk.strftime('%d/%m/%Y')
                else:
                    waktu_masuk_str = waktu_masuk
                    tanggal_str = '-'

                if isinstance(waktu_keluar, datetime):
                    waktu_keluar_str = waktu_keluar.strftime('%H:%M')
                else:
                    waktu_keluar_str = '-'

                if waktu_masuk and isinstance(waktu_masuk, datetime):
                    end_time = waktu_keluar if isinstance(waktu_keluar, datetime) else datetime.now()
                    durasi_td = end_time - waktu_masuk
                    durasi_str = format_duration(durasi_td)
                    is_overdue = durasi_td > max_duration
                else:
                    durasi_str = "-"
                    is_overdue = False

                row_id = self.tree.insert('', 'end',
                                          values=(tanggal_str, plat_nomor, waktu_masuk_str, waktu_keluar_str,
                                                  durasi_str))

                if waktu_masuk != '-':
                    self.tree.item(row_id, tags=('green',))

                if waktu_keluar_str == '-':
                    if is_overdue:
                        self.tree.item(row_id, tags=('overdue',))
                else:
                    if is_overdue:
                        self.tree.item(row_id, tags=('overdue',))
                    else:
                        self.tree.item(row_id, tags=('ontime',))

            self.tree.tag_configure('green', foreground='#006D5B')
            self.tree.tag_configure('overdue', foreground='white', background='#800000')
            self.tree.tag_configure('ontime', foreground='white', background='#0F52BA')

        self.root.after(3000, self.update_table)
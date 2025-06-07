import tkinter as tk
from tkinter import ttk
from src.handler import MParkingApp
from src.roi import ROIManager

class UI:
    def __init__(self, app: MParkingApp):
        self.app = app
        self.app.root.title("M-Parking")
        self.build_ui()
        self.app.update_streams()

    def build_ui(self):
        main_frame = tk.Frame(self.app.root)
        main_frame.pack(fill="both", expand=True)

        device_name = "GPU" if self.app.device == 'cuda' else 'cpu'
        cuda_frame = tk.Frame(self.app.root, height=40)
        cuda_frame.pack(fill="x", pady=10)
        self.app.cuda_label = tk.Label(cuda_frame, text=f"Running on {device_name}", font=("Helvetica", 9))
        self.app.cuda_label.pack(pady=5)

        input_frame = tk.Frame(main_frame)
        input_frame.pack(fill='x', padx=20, pady=10)

        tk.Label(input_frame, text="Stream Masuk:").grid(row=0, column=0, sticky='w')
        self.app.entry_in = tk.Entry(input_frame, width=30)
        self.app.entry_in.grid(row=0, column=1, padx=5)

        tk.Label(input_frame, text="Stream Keluar:").grid(row=1, column=0, sticky='w')
        self.app.entry_out = tk.Entry(input_frame, width=30)
        self.app.entry_out.grid(row=1, column=1, padx=5)

        def on_set_camera():
            source_in = self.app.entry_in.get().strip()
            source_out = self.app.entry_out.get().strip()
            self.app.set_camera_sources(source_in, source_out)

        btn_set = tk.Button(input_frame, text="Set Kamera", command=on_set_camera)
        btn_set.grid(row=0, column=2, rowspan=2, padx=10)

        stream_frame = tk.Frame(main_frame)
        stream_frame.pack(pady=10, fill="both", expand=True, padx=20)

        labels = ['Stream Masuk', 'Stream Keluar']
        for i, label in enumerate(labels):
            tk.Label(stream_frame, text=label, font=("Helvetica", 14, "bold")).grid(row=0, column=i, padx=10, pady=10)

        wrapper_in = tk.Frame(stream_frame)
        wrapper_in.grid(row=1, column=0, sticky="nsew")
        wrapper_out = tk.Frame(stream_frame)
        wrapper_out.grid(row=1, column=1, sticky="nsew")

        stream_frame.grid_rowconfigure(1, weight=1)
        stream_frame.grid_columnconfigure(0, weight=1)
        stream_frame.grid_columnconfigure(1, weight=1)

        self.app.canvas_in = tk.Canvas(wrapper_in, width=self.app.frame_width, height=self.app.frame_height, bg='grey')
        self.app.canvas_in.pack(expand=True)

        self.app.canvas_out = tk.Canvas(wrapper_out, width=self.app.frame_width, height=self.app.frame_height, bg='grey')
        self.app.canvas_out.pack(expand=True)

        self.app.preview_crop_in = tk.Label(stream_frame, text="Preview Crop Masuk", bg="grey", fg="white")
        self.app.preview_crop_in.grid(row=2, column=0, pady=5)
        self.app.preview_crop_in_img = None

        self.app.preview_crop_out = tk.Label(stream_frame, text="Preview Crop Keluar", bg="grey", fg="white")
        self.app.preview_crop_out.grid(row=2, column=1, pady=5)
        self.app.preview_crop_out_img = None

        self.app.fps_in_label = tk.Label(stream_frame, text=f"FPS Masuk: 0", font=("Helvetica", 12))
        self.app.fps_out_label = tk.Label(stream_frame, text=f"FPS Keluar: 0", font=("Helvetica", 12))
        self.app.fps_in_label.grid(row=3, column=0, pady=5)
        self.app.fps_out_label.grid(row=3, column=1, pady=5)

        table_frame = tk.Frame(main_frame)
        table_frame.pack(pady=10, fill="both", expand=True, padx=20)

        tk.Label(table_frame, text="Tabel Monitoring", font=("Helvetica", 16, "bold")).pack(pady=10)

        tree_frame = tk.Frame(table_frame)
        tree_frame.pack(fill="both", expand=True)

        columns = ('Tanggal', 'Plat Nomor', 'Waktu Masuk', 'Waktu Keluar', 'Durasi')
        self.app.tree = ttk.Treeview(tree_frame, columns=columns, show='headings')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure('Treeview.Heading', foreground='black', font=('Helvetica', 12, 'bold'))
        style.map('Treeview.Heading', background=[('active', '#808080')])

        for col in columns:
            self.app.tree.heading(col, text=col)
            self.app.tree.column(col, width=120, anchor='center')

        scrollbar = tk.Scrollbar(tree_frame, orient="vertical", command=self.app.tree.yview)
        self.app.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.app.tree.pack(fill="both", expand=True, padx=100)


        self.app.roi_manager = ROIManager(self.app.canvas_in, self.app.canvas_out, self.app.frame_width, self.app.frame_height)

        self.app.canvas_in.bind("<ButtonPress-1>", lambda e: self.app.roi_manager.on_mouse_down(e, 'in'))
        self.app.canvas_in.bind("<B1-Motion>", lambda e: self.app.roi_manager.on_mouse_move(e, 'in'))

        self.app.canvas_out.bind("<ButtonPress-1>", lambda e: self.app.roi_manager.on_mouse_down(e, 'out'))
        self.app.canvas_out.bind("<B1-Motion>", lambda e: self.app.roi_manager.on_mouse_move(e, 'out'))

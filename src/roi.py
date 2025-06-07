class ROIManager:

    def __init__(self, canvas_in, canvas_out, frame_width, frame_height):
        self.canvas_in = canvas_in
        self.canvas_out = canvas_out
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.roi_masuk = (200, 200, 580, 300)
        self.roi_keluar = (200, 200, 580, 300)

        self.roi_handle_size = 8

        # Drag state
        self.drag_data = {
            'canvas': None,
            'start_x': 0,
            'start_y': 0,
            'dragging': False,
            'mode': None,
            'resize_corner': None,
            'stream_type': None,
        }

        self.redraw_roi('in')
        self.redraw_roi('out')

    def get_roi_and_canvas(self, stream_type):
        if stream_type == 'in':
            return self.roi_masuk, self.canvas_in
        else:
            return self.roi_keluar, self.canvas_out

    def set_roi(self, stream_type, roi):
        if stream_type == 'in':
            self.roi_masuk = roi
        else:
            self.roi_keluar = roi

    def on_mouse_down(self, event, stream_type):
        roi, canvas = self.get_roi_and_canvas(stream_type)
        x, y = event.x, event.y

        handle = self.get_handle_hit(roi, x, y)
        if handle:
            self.drag_data.update({
                'canvas': canvas,
                'start_x': x,
                'start_y': y,
                'dragging': True,
                'mode': 'resize',
                'resize_corner': handle,
                'stream_type': stream_type
            })
            return

        if self.point_in_rect(x, y, roi):
            self.drag_data.update({
                'canvas': canvas,
                'start_x': x,
                'start_y': y,
                'dragging': True,
                'mode': 'move',
                'stream_type': stream_type
            })
        else:
            self.drag_data['dragging'] = False

    def on_mouse_move(self, event, stream_type):
        if not self.drag_data['dragging']:
            return
        if self.drag_data.get('stream_type') != stream_type:
            return

        roi = self.roi_masuk if stream_type == 'in' else self.roi_keluar

        dx = event.x - self.drag_data['start_x']
        dy = event.y - self.drag_data['start_y']

        x1, y1, x2, y2 = roi

        if self.drag_data['mode'] == 'move':
            new_x1 = max(0, min(self.frame_width - (x2 - x1), x1 + dx))
            new_y1 = max(0, min(self.frame_height - (y2 - y1), y1 + dy))
            new_x2 = new_x1 + (x2 - x1)
            new_y2 = new_y1 + (y2 - y1)
            new_roi = (new_x1, new_y1, new_x2, new_y2)

        elif self.drag_data['mode'] == 'resize':
            corner = self.drag_data['resize_corner']
            if corner == 'tl':
                new_x1 = max(0, min(x2 - 10, x1 + dx))
                new_y1 = max(0, min(y2 - 10, y1 + dy))
                new_roi = (new_x1, new_y1, x2, y2)
            elif corner == 'tr':
                new_x2 = min(self.frame_width, max(x1 + 10, x2 + dx))
                new_y1 = max(0, min(y2 - 10, y1 + dy))
                new_roi = (x1, new_y1, new_x2, y2)
            elif corner == 'bl':
                new_x1 = max(0, min(x2 - 10, x1 + dx))
                new_y2 = min(self.frame_height, max(y1 + 10, y2 + dy))
                new_roi = (new_x1, y1, x2, new_y2)
            elif corner == 'br':
                new_x2 = min(self.frame_width, max(x1 + 10, x2 + dx))
                new_y2 = min(self.frame_height, max(y1 + 10, y2 + dy))
                new_roi = (x1, y1, new_x2, new_y2)
            else:
                new_roi = roi
        else:
            new_roi = roi

        self.set_roi(stream_type, new_roi)
        self.drag_data['start_x'] = event.x
        self.drag_data['start_y'] = event.y

        self.redraw_roi(stream_type)

    def on_mouse_up(self, _event, stream_type):
        """Handler mouse up, parameter _event tidak digunakan secara langsung."""
        if self.drag_data['dragging'] and self.drag_data.get('stream_type') == stream_type:
            self.drag_data['dragging'] = False
            self.drag_data['mode'] = None
            self.drag_data['resize_corner'] = None

    @staticmethod
    def point_in_rect(x, y, rect):
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def get_handle_hit(self, roi, x, y):
        x1, y1, x2, y2 = roi
        handles = {
            'tl': (x1, y1),
            'tr': (x2, y1),
            'bl': (x1, y2),
            'br': (x2, y2),
        }
        size = self.roi_handle_size
        for name, (hx, hy) in handles.items():
            if (hx - size <= x <= hx + size) and (hy - size <= y <= hy + size):
                return name
        return None

    def redraw_roi(self, stream_type):
        roi, canvas = self.get_roi_and_canvas(stream_type)
        canvas.delete("roi_rect")
        canvas.delete("roi_handle")

        x1, y1, x2, y2 = roi
        if stream_type == 'in':
            outline_color = "blue"
            handle_fill = "blue"
        else:
            outline_color = "red"
            handle_fill = "red"

        canvas.create_rectangle(x1, y1, x2, y2, outline=outline_color, width=2, tag="roi_rect")

        for (hx, hy) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            canvas.create_rectangle(hx - self.roi_handle_size, hy - self.roi_handle_size,
                                    hx + self.roi_handle_size, hy + self.roi_handle_size,
                                    fill=handle_fill, outline="black", tag="roi_handle")

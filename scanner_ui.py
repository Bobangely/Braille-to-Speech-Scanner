"""Presentation only: cached HighGUI dashboard, using the existing Pillow fonts."""

import time
import unicodedata

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# RGB palette. Status is also written in text, never conveyed by color alone.
BG, CARD, BORDER = '#0e141e', '#171f2c', '#2b3749'
TEXT, MUTED, ACCENT = '#edf3fb', '#94a5ba', '#67cfca'
STATUS = {'Ready': MUTED, 'Scanning': '#76baff', 'Detected': '#67d9af',
          'Check image': '#efbc68', 'Error': '#f18f96'}


class ScannerUI:
    def __init__(self, font_loader):
        self.font_loader = font_loader
        self.size = (1440, 900)
        self.preview_rect = None
        self.result_card = None
        self.buttons = []
        self.scroll_offset = 0
        self.max_scroll = 0
        self._body_key = self._header_key = self._notice_key = self._wrap_key = None
        self._text = ''
        self._fps = (0., 0.)
        self._fps_at = 0.
        self._state = {}

    def resize(self, width, height):
        # HighGUI scales this bounded canvas on 4K displays; inference retains
        # every source pixel. Preserve the window aspect ratio, including portrait.
        if width < 160 or height < 120:
            return
        scale = min(1., 1920 / width, 1200 / height)
        self.size = (max(1, round(width * scale)), max(1, round(height * scale)))

    def _font(self, size, bold=False):
        font = self.font_loader(max(9, round(size * self.scale)), bold)
        return font if isinstance(font, (ImageFont.FreeTypeFont, ImageFont.ImageFont)) else ImageFont.load_default()

    def _label(self, draw, xy, text, size=15, color=TEXT, bold=False):
        draw.text(xy, text, font=self._font(size, bold), fill=color)

    def _layout(self):
        w, h = self.size
        self.scale = min(1.25, w / 1050, h / 720)
        s = self.scale
        m, gap = max(4, round(10*s)), max(4, round(8*s))
        self.header_h = round(88*s)
        top, bottom = self.header_h + gap, h - m
        if w / max(h, 1) >= 1.25:
            sidebar = round(256*s)
            left = w - m - sidebar
            self.preview_card = (m, top, left-gap, bottom)
            split = bottom - round(248*s)
            self.result_card = (left, top, w-m, split-gap)
            self.control_card = (left, split, w-m, bottom)
        else:
            split = max(top + round(160*s), bottom - round(250*s))
            self.preview_card = (m, top, w-m, split-gap)
            middle = round(w*.54)
            self.result_card = (m, split, middle-gap//2, bottom)
            self.control_card = (middle+gap//2, split, w-m, bottom)
        x1, y1, x2, y2 = self.preview_card
        self.video_box = (x1+1, y1+round(30*s), x2-1, y2-round(46*s))
        self.notice_box = (x1+1, y2-round(44*s), x2-1, y2-1)

    @staticmethod
    def _inside(box, x, y):
        return box is not None and box[0] <= x < box[2] and box[1] <= y < box[3]

    def hit_key(self, x, y):
        return next((key for box, key in self.buttons if self._inside(box, x, y)), None)

    def scroll(self, delta):
        self.scroll_offset = max(0, min(self.max_scroll, self.scroll_offset + delta))

    def _wrapped(self, text, width, size):
        key = text, width, size, self.scale
        if key != self._wrap_key:
            font = self._font(size)
            lines = []
            for paragraph in text.split('\n'):
                clusters = []
                for char in paragraph:
                    if clusters and unicodedata.category(char).startswith('M'):
                        clusters[-1] += char
                    else:
                        clusters.append(char)
                line = ''
                for cluster in clusters:
                    if line and font.getlength(line+cluster) > width:
                        lines.append(line)
                        line = ''
                    line += cluster
                lines.append(line)
            self._wrap_key, self._lines = key, lines
        return self._lines

    def _button(self, draw, box, label, key, active=False):
        draw.rounded_rectangle(box, radius=round(7*self.scale), fill='#253e47' if active else '#222d3d',
                               outline='#4ba9ab' if active else BORDER)
        x1, y1, x2, y2 = box
        font = self._font(12)
        for size in (11, 10, 9):
            if font.getlength(label) <= x2-x1-8*self.scale:
                break
            font = self._font(size)
        bounds = font.getbbox(label)
        draw.text(((x1+x2-font.getlength(label))/2,
                   (y1+y2-(bounds[3]-bounds[1]))/2-bounds[1]), label, font=font, fill=TEXT)
        self.buttons.append((box, key))

    def _build_body(self, state):
        self._layout()
        s = self.scale
        image = Image.new('RGB', self.size, BG)
        draw = ImageDraw.Draw(image)
        for box in (self.preview_card, self.result_card, self.control_card):
            draw.rounded_rectangle(box, radius=round(12*s), fill=CARD, outline=BORDER)
        self.buttons = []
        x, y, right, bottom = self.preview_card
        self._label(draw, (x+10*s, y+7*s), 'LIVE PREVIEW', 12, MUTED, True)
        summary = f"{state.get('cells', 0)} cells  /  {state.get('dots', 0)} dots  /  {state.get('lines', 0)} lines"
        self._label(draw, (right-self._font(12).getlength(summary)-10*s, y+7*s), summary, 12, MUTED)
        draw.rectangle(self.video_box, fill='#080d14')

        x, y, right, bottom = self.result_card
        self._label(draw, (x+10*s, y+10*s), 'DETECTION RESULT', 12, MUTED, True)
        self._label(draw, (x+10*s, y+34*s), 'ข้อความที่ยืนยันแล้ว' if state.get('lang') == 'thai' else 'Confirmed text', 13)
        text = state.get('text', '')
        if text != self._text:
            self.scroll_offset, self._text = 0, text
        lines = self._wrapped(text, right-x-20*s, 20) if text else []
        capacity = max(1, int((bottom-y-111*s) / (29*s)))
        self.max_scroll = max(0, len(lines)-capacity)
        self.scroll_offset = min(self.scroll_offset, self.max_scroll)
        if lines:
            for i, line in enumerate(lines[self.scroll_offset:self.scroll_offset+capacity]):
                self._label(draw, (x+10*s, y+(62+29*i)*s), line, 20)
        else:
            self._label(draw, (x+10*s, y+62*s), 'รอผลอ่านที่นิ่ง' if state.get('lang') == 'thai' else 'Waiting for a stable result', 16, MUTED)
        count, total = state.get('confirmed', 0), max(1, state.get('required', 6))
        progress_y = bottom-17*s
        draw.rounded_rectangle((x+10*s, progress_y, right-10*s, progress_y+4*s), radius=2, fill=BORDER)
        if count:
            draw.rounded_rectangle((x+10*s, progress_y, x+10*s+(right-x-20*s)*min(count/total, 1), progress_y+4*s), radius=2, fill=ACCENT)
        self._label(draw, (x+10*s, bottom-42*s), f'Confirmation  {count}/{total}', 12, MUTED)
        if self.max_scroll:
            self._button(draw, (right-72*s, bottom-45*s, right-45*s, bottom-23*s), '^', ord('['))
            self._button(draw, (right-37*s, bottom-45*s, right-10*s, bottom-23*s), 'v', ord(']'))

        x, y, right, bottom = self.control_card
        self._label(draw, (x+10*s, y+10*s), 'CONTROLS', 12, MUTED, True)
        color = state.get('color', 'blue').capitalize()
        language = 'Thai' if state.get('lang') == 'thai' else 'English'
        controls = [(f'{color}  [C]', 'c'), (f'{language}  [L]', 'l'),
                    ('Resolution  [V]', 'v'), (f"Sharp {state.get('sharp', 'OFF')}  [E]", 'e'),
                    ('Zoom -  [X]', 'x'), ('Zoom +  [Z]', 'z'),
                    ('Reset zoom  [R]', 'r'), ('Save image  [P]', 'p'),
                    ('Diagnostic  [D]', 'd'), ('Details  [H]', 'h')]
        gap, left, bw = 6*s, x+10*s, (right-x-26*s)/2
        for i, (label, key) in enumerate(controls):
            bx, by = left+(i % 2)*(bw+gap), y+(34+(i//2)*31)*s
            self._button(draw, (bx, by, bx+bw, by+26*s), label, ord(key), key == 'h' and state.get('details', False))
        self._label(draw, (left, y+191*s), 'Wheel: zoom / click: pan', 11, MUTED)
        self._button(draw, (left, bottom-34*s, right-10*s, bottom-8*s), 'Close scanner  [Q / Esc]', ord('q'))
        self._body = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    def compose(self, preview, state):
        self._state = dict(state)
        key = (self.size, tuple(sorted(state.items())), self.scroll_offset)
        if key != self._body_key:
            self._build_body(state)
            self._body_key = (self.size, tuple(sorted(state.items())), self.scroll_offset)
        canvas = self._body.copy()
        self.preview_rect = None
        if preview is not None and preview.size:
            x1, y1, x2, y2 = self.video_box
            ph, pw = preview.shape[:2]
            scale = min((x2-x1)/pw, (y2-y1)/ph)
            dw, dh = max(1, round(pw*scale)), max(1, round(ph*scale))
            x, y = x1+(x2-x1-dw)//2, y1+(y2-y1-dh)//2
            canvas[y:y+dh, x:x+dw] = cv2.resize(preview, (dw, dh), interpolation=cv2.INTER_LINEAR)
            self.preview_rect = (x, y, x+dw, y+dh)
        self.header(canvas)
        self.notice(canvas, '', state.get('status', 'Ready'))
        return canvas

    def header(self, canvas, fps=None):
        if fps is not None and time.monotonic()-self._fps_at >= .25:
            self._fps, self._fps_at = tuple(round(v, 1) for v in fps), time.monotonic()
        state, s = self._state, self.scale
        status = state.get('status', 'Ready')
        metrics = (f'{self._fps[0]:.1f} fps', f'{self._fps[1]:.1f} fps', state.get('resolution', ''),
                   state.get('mode', 'COLOR'), f"{state.get('zoom', 1.):.1f}x")
        key = self.size, metrics, status, state.get('lang'), state.get('color')
        if key != self._header_key:
            strip = Image.new('RGB', (self.size[0], self.header_h), BG)
            draw = ImageDraw.Draw(strip)
            self._label(draw, (12*s, 6*s), 'BRAILLE / SCANNER', 20, TEXT, True)
            self._label(draw, (13*s, 32*s), 'อ่านอักษรเบรลล์จากจุดสี' if state.get('lang') == 'thai' else 'Painted Braille recognition', 11, MUTED)
            color = STATUS.get(status, MUTED)
            draw.rounded_rectangle((self.size[0]-165*s, 9*s, self.size[0]-12*s, 38*s), radius=14*s, fill=CARD, outline=color)
            draw.ellipse((self.size[0]-152*s, 20*s, self.size[0]-146*s, 26*s), fill=color)
            self._label(draw, (self.size[0]-136*s, 15*s), status.upper(), 12, color, True)
            gap = (self.size[0]-24*s)/5
            for i, (label, value) in enumerate(zip(('PREVIEW', 'READ RATE', 'RESOLUTION', 'MODE', 'ZOOM'), metrics)):
                self._label(draw, (12*s+i*gap, 54*s), label, 9, MUTED)
                self._label(draw, (12*s+i*gap, 68*s), value, 13, TEXT, True)
            self._header = cv2.cvtColor(np.asarray(strip), cv2.COLOR_RGB2BGR)
            self._header_key = key
        canvas[:self.header_h] = self._header

    def notice(self, canvas, message, status=None):
        if status is None:
            status = ('Error' if any(word in message for word in ('ERROR', 'FAILED', 'UNAVAILABLE')) else
                      'Check image' if ('UNREADABLE' in message or
                          (message.startswith('C') and message[1:2].isdigit())) else
                      'Scanning' if message.startswith(('TRACKING', 'AI BUSY', 'CONFIRMING', 'CAMERA WAITING'))
                      else self._state.get('status', 'Scanning'))
        if status != self._state.get('status'):
            self._state['status'] = status
            self.header(canvas)
        if not message:
            message = {'Ready': 'วางอักษรเบรลล์ที่แต้มสีให้อยู่ในภาพ', 'Detected': 'ยืนยันผลแล้ว · พร้อมอ่านภาพถัดไป'}.get(status, 'กำลังตรวจจับและยืนยันผลอ่าน')
        if not self._state.get('details'):
            for prefix, friendly in [('SCAN ERROR', 'อ่านเฟรมนี้ไม่สำเร็จ ระบบจะลองเฟรมถัดไป'),
                ('UNREADABLE', 'ปรับโฟกัสหรือแสงให้เห็นจุดสีชัดขึ้น'),
                ('TRACKING', 'กำลังยืนยันตำแหน่งอักษรเบรลล์'),
                ('AI BUSY', 'กำลังประมวลผลภาพ กล้องยังทำงาน'),
                ('CAMERA UNAVAILABLE', 'ไม่พบกล้อง กรุณาตรวจสอบการเชื่อมต่อ'),
                ('CAMERA WAITING', 'กำลังรอสัญญาณภาพจากกล้อง')]:
                if message.startswith(prefix):
                    message = friendly
                    break
        key = self.size, self.notice_box, message, status
        if key != self._notice_key:
            x1, y1, x2, y2 = self.notice_box
            image = Image.new('RGB', (x2-x1, y2-y1), CARD)
            draw = ImageDraw.Draw(image)
            color, s = STATUS.get(status, MUTED), self.scale
            font = self._font(12)
            while message and font.getlength(message) > x2-x1-20*s:
                message = message[:-2].rstrip('…') + '…'
            self._label(draw, (10*s, 3*s), message, 12, color)
            self._label(draw, (10*s, 24*s), 'Green: active / Small ring: inactive    [H] Dot numbers / Details  [D] Trace', 10, MUTED)
            self._notice = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
            self._notice_key = key
        x1, y1, x2, y2 = self.notice_box
        canvas[y1:y2, x1:x2] = self._notice

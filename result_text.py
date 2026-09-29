"""Shaped Result-panel text when Windows Pillow lacks optional libraqm.

DrawTextW uses Windows complex-script layout rather than placing Thai marks
independently. Only cached CPU masks are retained; every GDI handle is released.
"""

import ctypes
from ctypes import wintypes as w
from functools import lru_cache
import sys

from PIL import Image, ImageFont


@lru_cache(maxsize=1)
def _win_api():
    gdi, user = ctypes.WinDLL('gdi32'), ctypes.WinDLL('user32')
    signatures = [
        (gdi.CreateCompatibleDC, [w.HDC], w.HDC),
        (gdi.CreateFontW, [ctypes.c_int] * 5 + [w.DWORD] * 8 + [w.LPCWSTR], w.HANDLE),
        (gdi.SelectObject, [w.HDC, w.HANDLE], w.HANDLE),
        (gdi.DeleteObject, [w.HANDLE], w.BOOL),
        (gdi.DeleteDC, [w.HDC], w.BOOL),
        (gdi.SetTextColor, [w.HDC, w.DWORD], w.DWORD),
        (gdi.SetBkColor, [w.HDC, w.DWORD], w.DWORD),
        (gdi.CreateDIBSection, [w.HDC, ctypes.c_void_p, w.UINT,
                               ctypes.POINTER(ctypes.c_void_p), w.HANDLE, w.DWORD], w.HANDLE),
        (gdi.GdiFlush, [], w.BOOL),
        (user.DrawTextW, [w.HDC, w.LPCWSTR, ctypes.c_int, ctypes.POINTER(w.RECT), w.UINT], ctypes.c_int),
    ]
    for function, arguments, result in signatures:
        function.argtypes, function.restype = arguments, result
    return gdi, user


class _BitmapInfo(ctypes.Structure):
    _fields_ = [('size', w.DWORD), ('width', w.LONG), ('height', w.LONG),
                ('planes', w.WORD), ('bits', w.WORD), ('compression', w.DWORD),
                ('image_size', w.DWORD), ('xppm', w.LONG), ('yppm', w.LONG),
                ('colors', w.DWORD), ('important', w.DWORD)]


def _native_text(text, family, size, raster=False):
    gdi, user = _win_api()
    dc = gdi.CreateCompatibleDC(None)
    if not dc:
        raise OSError('Cannot create Result text device context')
    font = bitmap = old_font = old_bitmap = None
    try:
        # Negative height = character size; grayscale AA keeps masks free of
        # ClearType RGB fringes when composited onto the dashboard background.
        font = gdi.CreateFontW(-size, 0, 0, 0, 400, 0, 0, 0, 1, 0, 0, 4, 0, family)
        if not font:
            raise OSError('Cannot create Result text font')
        old_font = gdi.SelectObject(dc, font)
        rect = w.RECT()
        flags = 0x20 | 0x800  # DT_SINGLELINE | DT_NOPREFIX (literal &)
        if not user.DrawTextW(dc, text, -1, ctypes.byref(rect), flags | 0x400):
            raise OSError('Cannot measure Result text')
        width, height = rect.right, rect.bottom
        if not raster:
            return width, height
        # Padding protects overhanging marks; do not crop individual glyphs.
        pad = max(2, size // 4)
        bw, bh = max(1, width + 2*pad), max(1, height + 2*pad)
        info = _BitmapInfo(ctypes.sizeof(_BitmapInfo), bw, -bh, 1, 32)
        pixels = ctypes.c_void_p()
        bitmap = gdi.CreateDIBSection(dc, ctypes.byref(info), 0, ctypes.byref(pixels), None, 0)
        if not bitmap or not pixels.value:
            raise OSError('Cannot create Result text bitmap')
        old_bitmap = gdi.SelectObject(dc, bitmap)
        ctypes.memset(pixels, 0, bw*bh*4)
        gdi.SetBkColor(dc, 0)
        gdi.SetTextColor(dc, 0xFFFFFF)
        rect = w.RECT(pad, pad, pad+width, pad+height)
        if not user.DrawTextW(dc, text, -1, ctypes.byref(rect), flags | 0x100):  # DT_NOCLIP
            raise OSError('Cannot draw Result text')
        gdi.GdiFlush()
        rgb = Image.frombytes('RGB', (bw, bh), ctypes.string_at(pixels, bw*bh*4), 'raw', 'BGRX')
        return rgb.getchannel('R'), pad
    finally:
        if old_bitmap:
            gdi.SelectObject(dc, old_bitmap)
        if bitmap:
            gdi.DeleteObject(bitmap)
        if old_font:
            gdi.SelectObject(dc, old_font)
        if font:
            gdi.DeleteObject(font)
        gdi.DeleteDC(dc)


class ResultText:
    """Use native shaping only for Thai Result text on BASIC Windows builds."""

    def __init__(self):
        # Instance-owned, bounded caches: resizing does not retain GDI handles.
        self._measure = lru_cache(maxsize=512)(_native_text)
        self._mask = lru_cache(maxsize=128)(
            lambda text, family, size: _native_text(text, family, size, raster=True))

    @staticmethod
    def native(font, text):
        return (sys.platform == 'win32' and isinstance(font, ImageFont.FreeTypeFont)
                and font.layout_engine == ImageFont.Layout.BASIC
                and any('\u0e00' <= char <= '\u0e7f' for char in text))

    def width(self, font, text):
        if self.native(font, text):
            return self._measure(text, font.getname()[0], font.size)[0]
        return font.getlength(text)

    def line_height(self, font, text):
        if self.native(font, text):
            return self._measure(text, font.getname()[0], font.size)[1]
        ascent, descent = font.getmetrics() if hasattr(font, 'getmetrics') else (font.getbbox(text)[3], 0)
        return ascent + descent

    def draw(self, image, draw, xy, text, font, fill):
        if self.native(font, text):
            mask, pad = self._mask(text, font.getname()[0], font.size)
            image.paste(fill, (round(xy[0])-pad, round(xy[1])-pad), mask)
        else:
            draw.text(xy, text, font=font, fill=fill)

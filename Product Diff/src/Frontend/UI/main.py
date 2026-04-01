import os
import sys
import re
import csv
import zipfile
from datetime import datetime

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QMessageBox, QHeaderView, QSplitter, QFrame, QScrollArea,
    QSizePolicy, QLayout, QDialog, QListWidget, QProgressBar,
    QTextEdit, QFileDialog, QGraphicsDropShadowEffect, QComboBox
)
from PySide6.QtGui import (
    QPixmap, QFont, QColor, QPalette, QPainter, QPen, QBrush,
    QFontMetrics, QIcon, QMovie
)
from PySide6.QtCore import Qt, QRect, QPoint, QSize, Signal, QThread
from PySide6.QtWidgets import QAbstractItemView

# ============================================================
# PATH SETUP  -- add backend to import path
# ============================================================
if getattr(sys, 'frozen', False):
    # Running inside a PyInstaller bundle
    BUNDLE_DIR = sys._MEIPASS
    SCRIPT_DIR = os.path.dirname(sys.executable)
    PROJECT_DIR = SCRIPT_DIR
    if BUNDLE_DIR not in sys.path:
        sys.path.insert(0, BUNDLE_DIR)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
    BUNDLE_DIR = None
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from defect_traceback_vlm import run_traceback, DefectBox  


# ============================================================
# FLOW LAYOUT
# ============================================================
class FlowLayout(QLayout):
    def __init__(self, parent=None, margin=10, spacing=12):
        super().__init__(parent)
        self.items = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)

    def addItem(self, item):
        self.items.append(item)

    def count(self):
        return len(self.items)

    def itemAt(self, index):
        if 0 <= index < len(self.items):
            return self.items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self.items):
            return self.items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._doLayout(QRect(0, 0, width, 0), True)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        return QSize(0, self._doLayout(QRect(0, 0, 0, 0), True))

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._doLayout(rect, False)

    def _doLayout(self, rect, testOnly):
        x = rect.x()
        y = rect.y()
        lineHeight = 0
        space = self.spacing()

        for item in self.items:
            nextX = x + item.sizeHint().width() + space
            if nextX - space > rect.right() and lineHeight > 0:
                x = rect.x()
                y += lineHeight + space
                nextX = x + item.sizeHint().width() + space
                lineHeight = 0

            if not testOnly:
                item.setGeometry(QRect(QPoint(x, y), item.sizeHint()))

            x = nextX
            lineHeight = max(lineHeight, item.sizeHint().height())

        return y + lineHeight - rect.y()


# ============================================================
# CONFIG
# ============================================================
def _resolve(relpath):
    """Resolve a path relative to the bundle (frozen) or project dir."""
    if BUNDLE_DIR:
        return os.path.join(BUNDLE_DIR, relpath)
    return os.path.join(PROJECT_DIR, relpath)

OutputRoot = os.path.join(PROJECT_DIR, "output")
CloudRoot = r"\\VNATSHFS.intel.com\VNATAnalysis$\MAOATM\VN\Applications\TE\Image_Tracer\result"
CsvName = "vid_data.csv"

LOGO_PATH = _resolve(os.path.join("Frontend", "assets", "logo.png"))
LOADING_GIF = _resolve(os.path.join("Frontend", "assets", "loading.gif"))

# ============================================================
# THEME
# ============================================================
CLR_PRIMARY = "#0071C5"
CLR_PRIMARY_DARK = "#005a9e"
CLR_PRIMARY_LIGHT = "#0096FF"
CLR_ACCENT = "#00b4d8"
CLR_DANGER = "#e74c3c"
CLR_BG = "#f0f4f8"
CLR_CARD = "#ffffff"
CLR_TEXT = "#1a1a2e"
CLR_TEXT_SEC = "#5a6474"
CLR_BORDER = "#d0dde8"
CLR_TABLE_ALT = "#f7f9fc"
CLR_TABLE_HDR = "#e6f0fb"
CLR_TABLE_HDR_TXT = "#0b3c6f"

GLOBAL_SS = f"""
    * {{
        font-family: "Segoe UI", "Inter", "Helvetica Neue", sans-serif;
    }}
    QToolTip {{
        background: {CLR_TEXT};
        color: white;
        border: none;
        padding: 5px 8px;
        border-radius: 4px;
        font-size: 12px;
    }}
"""

BTN_SS = """
    QPushButton {{
        background: {bg};
        color: white;
        padding: 0 {pad};
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
    }}
    QPushButton:hover {{ background: {hover}; }}
    QPushButton:pressed {{ background: {pressed}; }}
    QPushButton:disabled {{
        background: #bcc5d1;
        color: #f0f0f0;
    }}
"""


def _shadow(parent, blur=18, dy=3, color=QColor(0, 0, 0, 35)):
    eff = QGraphicsDropShadowEffect(parent)
    eff.setBlurRadius(blur)
    eff.setOffset(0, dy)
    eff.setColor(color)
    return eff


# ============================================================
# HELPERS
# ============================================================
def parse_date(s):
    for f in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, f)
        except Exception:
            pass
    return datetime.min


def safe_visual_id(s):
    return "".join(c for c in s if c.isalnum() or c in "_-")


def get_images(p):
    return [f for f in os.listdir(p)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]


def map_module(t):
    t = t.upper().strip()
    if t.startswith("HBI"):
        return "HDBI"
    if t.startswith("HXV"):
        return "HDMx"
    if t.startswith("DOB"):
        return "OLB"
    return "Module"


def read_vid_csv(path, vid):
    rows = []
    with open(path, encoding="utf-8") as f:
        sample = f.read(512)
        delim = ";" if ";" in sample.split("\n")[0] else ","

    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            if r["VISUAL_ID"].strip().lower() == vid.lower():
                tester = r["TESTER_ID"].strip()
                rows.append({
                    "OPERATION": r["OPERATION"].strip(),
                    "MODULE": map_module(tester),
                    "LOT": r["LOT"].strip(),
                    "TESTER": tester,
                    "END": r["TEST_END_DATE"].strip(),
                    "KEY": parse_date(r["TEST_END_DATE"].strip())
                })
    return rows


def read_dvi_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            r.pop("IMAGE_FULL_PATH", None)
            rows.append(r)
    return rows


OG_PAT = re.compile(r'X\w+_\d+_\d+_.*FRAME\d+.*\.(jpg|jpeg|png)', re.IGNORECASE)
PROC_PAT = re.compile(r'(?:\w+_)?\d+_(In|Out)\.(jpg|jpeg|png)', re.IGNORECASE)


def proc_sort_key(fname):
    """Sort process images by number, Out before In (matches backend logic)."""
    m = re.search(r'(\d+)_(In|Out)', fname, re.IGNORECASE)
    if m:
        num = int(m.group(1))
        is_out = m.group(2).lower() == 'out'
        return (num, 0 if is_out else 1)
    return (0, 0)


def _isdir_safe(path):
    """os.path.isdir that won't hang on unreachable UNC paths."""
    try:
        return os.path.isdir(path)
    except OSError:
        return False


def _isfile_safe(path):
    """os.path.isfile that won't hang on unreachable UNC paths."""
    try:
        return os.path.isfile(path)
    except OSError:
        return False


def find_traceback_dir(vid):
    """Find the directory containing OG images and DVI CSV for a VID.
    Checks local dirs first, then cloud (UNC) as fallback.
    """
    # Check local candidates first (fast)
    for d in [os.path.join(PROJECT_DIR, vid), os.path.join(OutputRoot, vid)]:
        if os.path.isdir(d):
            return d
    # Cloud fallback (may be slow if network is unreachable)
    cloud = os.path.join(CloudRoot, vid)
    if _isdir_safe(cloud):
        return cloud
    return None


# ============================================================
# DRAWABLE IMAGE WIDGET  (manual defect box drawing)
# ============================================================
class DrawableImageWidget(QWidget):
    """QPainter canvas that displays an image and lets the user
    draw rectangles (defect boxes) on it with mouse drag.
    Scroll wheel to zoom, right-click drag to pan."""

    boxes_changed = Signal()

    ZOOM_MIN = 1.0
    ZOOM_MAX = 20.0
    ZOOM_STEP = 1.15

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

        self._src_pixmap = QPixmap(image_path)
        self._img_w = self._src_pixmap.width()
        self._img_h = self._src_pixmap.height()

        self.boxes = []          # list of (x, y, w, h) in image-pixel coords
        self._drawing = False
        self._start_pt = None    # widget coords
        self._current_pt = None  # widget coords

        # zoom / pan state
        self._zoom = 1.0         # 1.0 = fit-to-widget
        self._pan_x = 0.0        # pan offset in widget pixels
        self._pan_y = 0.0
        self._panning = False
        self._pan_anchor = None  # widget QPoint where pan started

    # ---- coordinate helpers ----
    def _base_scale(self):
        """Scale factor to fit image into widget at zoom=1."""
        ww, wh = self.width(), self.height()
        return min(ww / self._img_w, wh / self._img_h)

    def _display_rect(self):
        """Return (dx, dy, dw, dh) for the zoomed/panned image."""
        bs = self._base_scale() * self._zoom
        dw = int(self._img_w * bs)
        dh = int(self._img_h * bs)
        # center then apply pan
        dx = (self.width() - dw) / 2.0 + self._pan_x
        dy = (self.height() - dh) / 2.0 + self._pan_y
        return dx, dy, dw, dh

    def _widget_to_image(self, pt):
        """Convert widget QPoint to image-pixel (x, y)."""
        dx, dy, dw, dh = self._display_rect()
        sx = self._img_w / dw
        sy = self._img_h / dh
        ix = (pt.x() - dx) * sx
        iy = (pt.y() - dy) * sy
        return max(0, min(ix, self._img_w)), max(0, min(iy, self._img_h))

    def reset_view(self):
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.update()

    # ---- paint ----
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        # dark canvas background
        p.fillRect(self.rect(), QColor("#1e1e2e"))

        dx, dy, dw, dh = self._display_rect()
        p.drawPixmap(int(dx), int(dy), dw, dh, self._src_pixmap)

        scale = dw / self._img_w

        # committed boxes -- red with semi-transparent fill
        for idx, (bx, by, bw, bh) in enumerate(self.boxes):
            rx = int(dx + bx * scale)
            ry = int(dy + by * scale)
            rw = int(bw * scale)
            rh = int(bh * scale)
            p.setPen(QPen(QColor(255, 60, 60), 2))
            p.setBrush(QBrush(QColor(255, 60, 60, 30)))
            p.drawRect(rx, ry, rw, rh)
            # label
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(255, 60, 60, 200))
            label = f"D{idx+1}"
            fm = p.fontMetrics()
            tw = fm.horizontalAdvance(label) + 8
            p.drawRoundedRect(rx, ry - 18, tw, 18, 3, 3)
            p.setPen(QColor("white"))
            p.drawText(rx + 4, ry - 4, label)
            p.setBrush(Qt.NoBrush)

        # current drag preview -- lime dashed
        if self._drawing and self._start_pt and self._current_pt:
            pen_lime = QPen(QColor(0, 255, 120), 2, Qt.DashLine)
            p.setPen(pen_lime)
            p.setBrush(QBrush(QColor(0, 255, 120, 25)))
            r = QRect(self._start_pt.toPoint(), self._current_pt.toPoint()).normalized()
            p.drawRect(r)

        # zoom indicator
        if self._zoom > 1.01:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 140))
            p.drawRoundedRect(8, 8, 70, 22, 6, 6)
            p.setPen(QColor("white"))
            p.drawText(16, 24, f"{self._zoom:.1f}x")

        p.end()

    # ---- mouse events ----
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_anchor = event.position()
            self.setCursor(Qt.ClosedHandCursor)
        elif event.button() == Qt.LeftButton:
            self._drawing = True
            self._start_pt = event.position()
            self._current_pt = event.position()

    def mouseMoveEvent(self, event):
        if self._panning and self._pan_anchor:
            delta = event.position() - self._pan_anchor
            self._pan_x += delta.x()
            self._pan_y += delta.y()
            self._pan_anchor = event.position()
            self.update()
        elif self._drawing:
            self._current_pt = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if (event.button() == Qt.RightButton or event.button() == Qt.MiddleButton) \
                and self._panning:
            self._panning = False
            self._pan_anchor = None
            self.setCursor(Qt.CrossCursor)
        elif event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            ix1, iy1 = self._widget_to_image(self._start_pt)
            ix2, iy2 = self._widget_to_image(self._current_pt)
            x = min(ix1, ix2)
            y = min(iy1, iy2)
            w = abs(ix2 - ix1)
            h = abs(iy2 - iy1)
            if w > 5 and h > 5:
                self.boxes.append((x, y, w, h))
                self.boxes_changed.emit()
            self._start_pt = None
            self._current_pt = None
            self.update()

    def wheelEvent(self, event):
        """Zoom toward the cursor position."""
        mouse_pos = event.position()
        old_img_x, old_img_y = self._widget_to_image(
            QPoint(int(mouse_pos.x()), int(mouse_pos.y())))

        # Apply zoom
        if event.angleDelta().y() > 0:
            self._zoom = min(self._zoom * self.ZOOM_STEP, self.ZOOM_MAX)
        else:
            self._zoom = max(self._zoom / self.ZOOM_STEP, self.ZOOM_MIN)

        # Adjust pan so the image point under cursor stays under cursor
        dx, dy, dw, dh = self._display_rect()
        scale = dw / self._img_w
        new_wx = dx + old_img_x * scale
        new_wy = dy + old_img_y * scale
        self._pan_x += mouse_pos.x() - new_wx
        self._pan_y += mouse_pos.y() - new_wy

        self.update()

    def undo(self):
        if self.boxes:
            self.boxes.pop()
            self.boxes_changed.emit()
            self.update()

    def clear_boxes(self):
        self.boxes.clear()
        self.boxes_changed.emit()
        self.update()


# ============================================================
# DRAWING DIALOG 
# ============================================================
class DrawingDialog(QDialog):
    """Dialog where user draws defect boxes on an OG image."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Defect Boxes")
        self.resize(1000, 720)
        self.result_boxes = []

        self.setStyleSheet(f"""
            QDialog {{ background: {CLR_BG}; }}
            QLabel {{ color: {CLR_TEXT}; }}
        """)

        root = QHBoxLayout(self)
        root.setSpacing(16)
        root.setContentsMargins(12, 12, 12, 12)

        # canvas
        self.canvas = DrawableImageWidget(image_path)
        self.canvas.setStyleSheet("border-radius: 8px;")
        root.addWidget(self.canvas, stretch=5)

        # sidebar
        side_frame = QFrame()
        side_frame.setFixedWidth(220)
        side_frame.setStyleSheet(f"""
            QFrame {{
                background: {CLR_CARD};
                border: 1px solid {CLR_BORDER};
                border-radius: 10px;
            }}
        """)
        side = QVBoxLayout(side_frame)
        side.setContentsMargins(12, 14, 12, 14)
        side.setSpacing(10)

        title = QLabel("Defect Boxes")
        title.setStyleSheet(f"font-size:14px;font-weight:700;color:{CLR_PRIMARY};border:none;")
        side.addWidget(title)

        hint = QLabel("Left-drag: draw box\nScroll: zoom\nRight-drag: pan")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size:11px;color:{CLR_TEXT_SEC};border:none;")
        side.addWidget(hint)

        self.box_list = QListWidget()
        self.box_list.setStyleSheet(f"""
            QListWidget {{
                background: {CLR_BG};
                border: 1px solid {CLR_BORDER};
                border-radius: 6px;
                font-size: 12px;
                color: {CLR_TEXT};
            }}
            QListWidget::item {{ padding: 4px 6px; }}
            QListWidget::item:selected {{ background: {CLR_TABLE_HDR}; color: {CLR_PRIMARY}; }}
        """)
        side.addWidget(self.box_list, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        btn_undo = QPushButton("Undo")
        btn_undo.setFixedHeight(32)
        btn_undo.setStyleSheet(BTN_SS.format(
            bg=CLR_TEXT_SEC, hover="#4a5568", pressed="#3a4558", pad="14px"))
        btn_undo.clicked.connect(self._undo)
        btn_row.addWidget(btn_undo)

        btn_clear = QPushButton("Clear")
        btn_clear.setFixedHeight(32)
        btn_clear.setStyleSheet(BTN_SS.format(
            bg=CLR_DANGER, hover="#c0392b", pressed="#a93226", pad="14px"))
        btn_clear.clicked.connect(self._clear)
        btn_row.addWidget(btn_clear)
        side.addLayout(btn_row)

        btn_reset = QPushButton("Reset View")
        btn_reset.setFixedHeight(28)
        btn_reset.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {CLR_TEXT_SEC};
                border: 1px solid {CLR_BORDER};
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
            }}
            QPushButton:hover {{ background: {CLR_BG}; color: {CLR_TEXT}; }}
        """)
        btn_reset.clicked.connect(self.canvas.reset_view)
        side.addWidget(btn_reset)

        side.addStretch()

        btn_ok = QPushButton("Confirm")
        btn_ok.setFixedHeight(38)
        btn_ok.setStyleSheet(BTN_SS.format(
            bg=CLR_PRIMARY, hover=CLR_PRIMARY_DARK, pressed="#004080", pad="20px"))
        btn_ok.clicked.connect(self._confirm)
        side.addWidget(btn_ok)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.setFixedHeight(34)
        btn_cancel.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {CLR_TEXT_SEC};
                border: 1px solid {CLR_BORDER};
                border-radius: 8px;
                padding: 0 20px;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton:hover {{ background: {CLR_BG}; color: {CLR_TEXT}; }}
        """)
        btn_cancel.clicked.connect(self.reject)
        side.addWidget(btn_cancel)

        root.addWidget(side_frame)
        self.canvas.boxes_changed.connect(self._refresh_list)

    def _refresh_list(self):
        self.box_list.clear()
        for i, (x, y, w, h) in enumerate(self.canvas.boxes):
            self.box_list.addItem(
                f"D{i+1}  ({int(x)},{int(y)})  {int(w)}x{int(h)}")

    def _undo(self):
        self.canvas.undo()

    def _clear(self):
        self.canvas.clear_boxes()

    def _confirm(self):
        if not self.canvas.boxes:
            QMessageBox.warning(self, "No boxes", "Draw at least one defect box.")
            return
        self.result_boxes = list(self.canvas.boxes)
        self.accept()


# ============================================================
# TRACEBACK WORKER  (QThread)
# ============================================================
class TracebackWorker(QThread):
    progress = Signal(int, int, str)   # step, total, message
    finished = Signal(dict)            # result dict from run_traceback
    error = Signal(str)

    def __init__(self, uploads_dir, outdir, defect_boxes=None, ref_image_key=None):
        super().__init__()
        self.uploads_dir = uploads_dir
        self.outdir = outdir
        self.defect_boxes = defect_boxes
        self.ref_image_key = ref_image_key

    def run(self):
        try:
            result = run_traceback(
                uploads_dir=self.uploads_dir,
                outdir=self.outdir,
                defect_boxes=self.defect_boxes,
                ref_image_key=self.ref_image_key,
                progress_callback=lambda s, t, m: self.progress.emit(s, t, m),
            )
            if "error" in result:
                self.error.emit(result["error"])
            else:
                self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ============================================================
# REPORT DIALOG
# ============================================================
class ReportDialog(QDialog):
    """Shows the traceback report text and lets user save it."""

    def __init__(self, report_text: str, report_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Traceback Report")
        self.resize(750, 550)
        self._report_path = report_path

        self.setStyleSheet(f"QDialog {{ background: {CLR_BG}; }}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        header = QLabel("Traceback Report")
        header.setStyleSheet(f"font-size:16px;font-weight:700;color:{CLR_PRIMARY};")
        lay.addWidget(header)

        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont("Cascadia Code", 10) if QFont("Cascadia Code").exactMatch()
                    else QFont("Consolas", 10))
        te.setPlainText(report_text)
        te.setStyleSheet(f"""
            QTextEdit {{
                background: {CLR_CARD};
                color: {CLR_TEXT};
                border: 1px solid {CLR_BORDER};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        lay.addWidget(te)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        lay.addLayout(bar)
        bar.addStretch()

        btn_save = QPushButton("Save As...")
        btn_save.setFixedHeight(36)
        btn_save.setStyleSheet(BTN_SS.format(
            bg=CLR_PRIMARY, hover=CLR_PRIMARY_DARK, pressed="#004080", pad="18px"))
        btn_save.clicked.connect(self._save)
        bar.addWidget(btn_save)

        btn_close = QPushButton("Close")
        btn_close.setFixedHeight(36)
        btn_close.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {CLR_TEXT_SEC};
                border: 1px solid {CLR_BORDER};
                border-radius: 8px;
                padding: 0 18px;
                font-weight: 600;
                font-size: 13px;
            }}
            QPushButton:hover {{ background: {CLR_BG}; color: {CLR_TEXT}; }}
        """)
        btn_close.clicked.connect(self.accept)
        bar.addWidget(btn_close)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", self._report_path, "Text (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.findChild(QTextEdit).toPlainText())


# ============================================================
# PANEL IMAGE POPUP
# ============================================================
class PanelViewDialog(QDialog):
    """Popup that shows the traceback panel image at full size."""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Traceback Panel")
        self.setStyleSheet(f"QDialog {{ background: #1e1e2e; }}")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea { border: none; background: #1e1e2e; }
            QScrollBar { background: #2a2a3e; }
            QScrollBar::handle { background: #555; border-radius: 4px; }
        """)

        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setStyleSheet("background: #1e1e2e;")
        px = QPixmap(image_path)
        img_label.setPixmap(px)

        scroll.setWidget(img_label)
        lay.addWidget(scroll)

        self.setWindowState(Qt.WindowMaximized)


# ============================================================
# UI APP
# ============================================================
TABLE_SS = f"""
    QTableWidget {{
        background: {CLR_CARD};
        color: {CLR_TEXT};
        font-size: 13px;
        alternate-background-color: {CLR_TABLE_ALT};
        border: 1px solid {CLR_BORDER};
        border-radius: 8px;
        gridline-color: {CLR_BORDER};
        selection-background-color: #dbe9f9;
        selection-color: {CLR_TEXT};
    }}
    QHeaderView::section {{
        background: {CLR_TABLE_HDR};
        color: {CLR_TABLE_HDR_TXT};
        padding: 7px;
        font-weight: 600;
        border: none;
        border-bottom: 2px solid {CLR_PRIMARY};
    }}
"""


class VisionApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Stain Detective")
        self.setWindowIcon(QIcon(LOGO_PATH))
        self.resize(1150, 780)
        self._center()

        self.setFont(QFont("Segoe UI", 10))
        self.setStyleSheet(GLOBAL_SS)
        pal = QPalette()
        pal.setColor(QPalette.Window, QColor(CLR_BG))
        self.setPalette(pal)

        # state
        self._current_vid = ""
        self._search_rows = []
        self._worker = None
        self._last_results = None
        self._traceback_dir = None
        self._defect_entries = []

        main = QVBoxLayout(self)
        main.setContentsMargins(8, 6, 8, 8)
        main.setSpacing(4)

        # ===== HEADER =====
        header = QLabel("  Stain Detective")
        header.setFixedHeight(36)
        header.setStyleSheet(f"""
            QLabel {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {CLR_PRIMARY}, stop:1 {CLR_PRIMARY_LIGHT}
                );
                color: white;
                padding: 7px 18px;
                font-size: 18px;
                font-weight: 700;
                letter-spacing: 0.5px;
                border-radius: 10px;
            }}
        """)
        header.setGraphicsEffect(_shadow(header))
        main.addWidget(header)

        # ==================================================
        # SEARCH BAR ROW
        # ==================================================
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)

        self.vid_input = QLineEdit()
        self.vid_input.setPlaceholderText("Enter Visual ID...")
        self.vid_input.setFixedHeight(36)
        self.vid_input.setStyleSheet(f"""
            QLineEdit {{
                background: {CLR_CARD};
                color: {CLR_TEXT};
                border: 2px solid {CLR_BORDER};
                border-radius: 8px;
                padding: 0 10px;
                font-size: 14px;
            }}
            QLineEdit:focus {{
                border-color: {CLR_PRIMARY};
            }}
        """)
        self.vid_input.returnPressed.connect(self.perform_search)
        top.addWidget(self.vid_input)

        # --- LOADING GIF LABEL ---
        self.loading_label = QLabel()
        self.loading_label.setFixedSize(32, 32)
        self.loading_label.setAlignment(Qt.AlignCenter)
        self.movie = QMovie(LOADING_GIF)
        self.loading_label.setMovie(self.movie)
        self.loading_label.hide()
        top.addWidget(self.loading_label)

# --- SEARCH BUTTON ---
        btn_search = QPushButton("Search")
        btn_search.setFixedHeight(36)
        btn_search.setStyleSheet(BTN_SS.format(
            bg=CLR_PRIMARY, hover=CLR_PRIMARY_DARK, pressed="#004080", pad="22px"))
        btn_search.clicked.connect(self.perform_search)
        top.addWidget(btn_search)

        # --- MANUAL BUTTON ---
        self.btn_manual = QPushButton("Manual")
        self.btn_manual.setFixedHeight(36)
        self.btn_manual.setEnabled(False)
        self.btn_manual.setToolTip("Draw defect boxes manually")
        self.btn_manual.setStyleSheet(BTN_SS.format(
            bg=CLR_PRIMARY_DARK, hover="#004b85", pressed="#003a6b", pad="18px"))
        self.btn_manual.clicked.connect(self.on_manual)
        top.addWidget(self.btn_manual)

        # --- AUTO BUTTON ---
        self.btn_auto = QPushButton("Auto")
        self.btn_auto.setFixedHeight(36)
        self.btn_auto.setEnabled(False)
        self.btn_auto.setToolTip("Use DVI_box_data.csv defect boxes")
        self.btn_auto.setStyleSheet(BTN_SS.format(
            bg=CLR_PRIMARY, hover=CLR_PRIMARY_DARK, pressed="#004080", pad="18px"))
        self.btn_auto.clicked.connect(self.on_auto)
        top.addWidget(self.btn_auto)

        
        main.addLayout(top)

        # ==================================================
        # PROGRESS BAR + STATUS
        # ==================================================
        prog_row = QHBoxLayout()
        prog_row.setContentsMargins(0, 0, 0, 0)
        prog_row.setSpacing(6)
        main.addLayout(prog_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 7)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setMaximumHeight(6)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background: {CLR_BORDER};
                border-radius: 3px;
                border: none;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 {CLR_PRIMARY}, stop:1 {CLR_ACCENT}
                );
                border-radius: 3px;
            }}
        """)
        self.progress_bar.hide()
        prog_row.addWidget(self.progress_bar)

        self.status_label = QLabel("")
        self.status_label.setFixedHeight(16)
        self.status_label.setStyleSheet(f"color:{CLR_PRIMARY};font-size:11px;font-weight:500;")
        prog_row.addWidget(self.status_label)

        # ==================================================
        # RESULTS BAR (Defect selector / View Report / Clear)
        # ==================================================
        results_row = QHBoxLayout()
        results_row.setContentsMargins(0, 0, 0, 0)
        results_row.setSpacing(6)
        main.addLayout(results_row)

        self.defect_combo_label = QLabel("Defect:")
        self.defect_combo_label.setStyleSheet(
            f"font-size:12px;font-weight:600;color:{CLR_TEXT};")
        self.defect_combo_label.hide()
        results_row.addWidget(self.defect_combo_label)

        self.defect_combo = QComboBox()
        self.defect_combo.setFixedHeight(28)
        self.defect_combo.setMinimumWidth(200)
        self.defect_combo.setStyleSheet(f"""
            QComboBox {{
                background: {CLR_CARD};
                color: {CLR_PRIMARY_DARK};
                border: 1px solid {CLR_BORDER};
                border-radius: 6px;
                padding: 2px 8px;
                font-size: 12px;
                font-weight: 600;
            }}
            QComboBox:hover {{ border-color: {CLR_PRIMARY}; }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                color: {CLR_PRIMARY_DARK};
                background: {CLR_CARD};
                selection-background-color: {CLR_TABLE_HDR};
                selection-color: {CLR_PRIMARY};
                font-size: 12px;
                border: 1px solid {CLR_BORDER};
            }}
        """)
        self.defect_combo.currentIndexChanged.connect(self._on_defect_selected)
        self.defect_combo.hide()
        results_row.addWidget(self.defect_combo)

        self.btn_report = QPushButton("View Report")
        self.btn_report.setFixedHeight(28)
        self.btn_report.setStyleSheet(BTN_SS.format(
            bg=CLR_PRIMARY, hover=CLR_PRIMARY_DARK, pressed="#004080", pad="16px"))
        self.btn_report.clicked.connect(self._view_report)
        self.btn_report.hide()
        results_row.addWidget(self.btn_report)

        self.btn_clear = QPushButton("Clear Results")
        self.btn_clear.setFixedHeight(28)
        self.btn_clear.setStyleSheet(BTN_SS.format(
            bg=CLR_DANGER, hover="#c0392b", pressed="#a93226", pad="16px"))
        self.btn_clear.clicked.connect(self._clear_results)
        self.btn_clear.hide()
        results_row.addWidget(self.btn_clear)

        self.btn_download = QPushButton("Download ZIP")
        self.btn_download.setFixedHeight(28)
        self.btn_download.setStyleSheet(BTN_SS.format(
            bg="#27ae60", hover="#219a52", pressed="#1a7a40", pad="16px"))
        self.btn_download.clicked.connect(self._download_zip)
        self.btn_download.hide()
        results_row.addWidget(self.btn_download)

        results_row.addStretch()

        # ==================================================
        # SPLITTER
        # ==================================================
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background: {CLR_BORDER};
                border-radius: 3px;
                margin: 40px 0px;
            }}
        """)
        main.addWidget(splitter)

        # LEFT PANEL
        left = QFrame()
        left.setStyleSheet(f"""
            QFrame {{
                background: {CLR_CARD};
                border: 1px solid {CLR_BORDER};
                border-radius: 10px;
            }}
        """)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 10, 10, 10)
        ll.setSpacing(8)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Operation", "Module", "LOT", "Tester", "End Date"]
        )
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet(TABLE_SS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        ll.addWidget(self.table)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background: transparent;
            }}
            QScrollBar:vertical {{
                width: 8px;
                background: transparent;
            }}
            QScrollBar::handle:vertical {{
                background: {CLR_BORDER};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {CLR_PRIMARY};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        self.flow_container = QWidget()
        self.flow_container.setStyleSheet("background: transparent; border: none;")
        self.flow_layout = FlowLayout(self.flow_container)
        scroll.setWidget(self.flow_container)
        ll.addWidget(scroll)

        self.dvi_title = QLabel("DVI Record")
        self.dvi_title.setStyleSheet(
            f"font-size:14px;font-weight:700;color:{CLR_PRIMARY};"
            f"margin-top:6px;border:none;")
        self.dvi_title.hide()
        ll.addWidget(self.dvi_title)

        self.dvi_table = QTableWidget()
        self.dvi_table.verticalHeader().setVisible(False)
        self.dvi_table.setAlternatingRowColors(True)
        self.dvi_table.setStyleSheet(TABLE_SS)
        self.dvi_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.dvi_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        self.dvi_table.hide()
        ll.addWidget(self.dvi_table)

        splitter.addWidget(left)

        # RIGHT PREVIEW
        right = QFrame()
        right.setStyleSheet("border: none; background: transparent;")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        self.preview_box = QFrame()
        self.preview_box.setFixedWidth(360)
        self.preview_box.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.preview_box.setStyleSheet(f"""
            QFrame {{
                background: {CLR_CARD};
                border: 1px solid {CLR_BORDER};
                border-radius: 12px;
            }}
        """)
        self.preview_box.setGraphicsEffect(_shadow(self.preview_box, blur=14, dy=2))
        pb = QVBoxLayout(self.preview_box)
        pb.setContentsMargins(8, 8, 8, 8)

        self.preview = QLabel("Select an image")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet(f"""
            color: {CLR_TEXT_SEC};
            font-size: 13px;
            font-style: italic;
            border: none;
        """)
        pb.addWidget(self.preview)

        rl.addWidget(self.preview_box)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 20)
        splitter.setStretchFactor(1, 1)

    # --------------------------------------------------
    def _center(self):
        f = self.frameGeometry()
        screen = QApplication.primaryScreen().availableGeometry().center()
        f.moveCenter(screen)
        self.move(f.topLeft())

    def _start_searching(self):
        self.vid_input.setEnabled(False)
        self.movie.start()
        self.loading_label.show()
        QApplication.processEvents()

    def _end_searching(self):
        self.movie.stop()
        self.loading_label.hide()
        self.vid_input.setEnabled(True)

    def _set_buttons_enabled(self, enabled):
        self.btn_manual.setEnabled(enabled)
        self.btn_auto.setEnabled(enabled)

    def show_preview(self, path):
        px = QPixmap(path).scaled(
            340, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview.setStyleSheet("border: none;")
        self.preview.setPixmap(px)

    def make_thumb(self, path, caption):
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background: {CLR_CARD};
                border: 1px solid {CLR_BORDER};
                border-radius: 10px;
            }}
            QFrame:hover {{
                border-color: {CLR_PRIMARY};
            }}
        """)
        lay = QVBoxLayout(card)
        lay.setAlignment(Qt.AlignCenter)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        thumb = QLabel()
        thumb.setFixedSize(130, 220)
        thumb.setAlignment(Qt.AlignCenter)
        thumb.setStyleSheet("border: none; border-radius: 6px;")
        thumb.setPixmap(
            QPixmap(path).scaled(126, 216, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        thumb.setCursor(Qt.PointingHandCursor)
        thumb.mousePressEvent = lambda e, p=path: self.show_preview(p)

        lbl = QLabel(caption.replace("(", " - ").replace(")", ""))
        lbl.setFixedWidth(130)
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"font-size:11px;color:{CLR_TEXT_SEC};font-weight:500;border:none;")
        lbl.setText(QFontMetrics(lbl.font()).elidedText(
            caption, Qt.ElideMiddle, 130))

        lay.addWidget(thumb)
        lay.addWidget(lbl)
        return card

    def load_thumbnails(self, out_dir, rows):
        while self.flow_layout.count():
            it = self.flow_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        imgs = get_images(out_dir)
        items = []
        for r in rows:
            for img in imgs:
                if img.lower().startswith(r["OPERATION"].lower()):
                    items.append(
                        (r["KEY"], os.path.join(out_dir, img),
                         f"Operation {r['OPERATION']}"))

        items.sort(key=lambda x: x[0])

        for _, p, c in items:
            self.flow_layout.addWidget(self.make_thumb(p, c))

    # --------------------------------------------------
    # SEARCH
    # --------------------------------------------------
    def perform_search(self):
        vid = safe_visual_id(self.vid_input.text().strip())
        if not vid:
            return

        out_dir = os.path.join(OutputRoot, f"{vid}_output")

        self._start_searching()
        self._clear_results()

        self._current_vid = vid
        self._traceback_dir = find_traceback_dir(vid)

        # Look for vid_data.csv — check local paths first, cloud last
        vid_csv = None
        for candidate in [
            os.path.join(out_dir, CsvName),
            os.path.join(self._traceback_dir, CsvName) if self._traceback_dir else "",
            os.path.join(CloudRoot, vid, CsvName),
        ]:
            if not candidate:
                continue
            check = _isfile_safe if candidate.startswith("\\\\") else os.path.isfile
            if check(candidate):
                vid_csv = candidate
                break
        has_vid_csv = vid_csv is not None
        # Directory where vid_csv lives (for loading thumbnails)
        data_dir = os.path.dirname(vid_csv) if vid_csv else self._traceback_dir

        # --- Process table from vid_data.csv, or fallback from images ---
        rows = []
        if has_vid_csv:
            rows = read_vid_csv(vid_csv, vid)
            rows.sort(key=lambda r: r["KEY"])
        elif self._traceback_dir:
            # Build stub rows from image filenames so the table still shows steps
            step = 0
            for f in sorted(os.listdir(self._traceback_dir), key=proc_sort_key):
                fpath = os.path.join(self._traceback_dir, f)
                if not os.path.isfile(fpath):
                    continue
                if OG_PAT.match(f):
                    rows.append({
                        "OPERATION": f, "MODULE": "", "LOT": "",
                        "TESTER": "", "END": "",
                        "KEY": datetime.min,
                    })
                elif PROC_PAT.match(f):
                    step += 1
                    rows.append({
                        "OPERATION": f"{step} - {f}",
                        "MODULE": "", "LOT": "",
                        "TESTER": "", "END": "",
                        "KEY": datetime.min,
                    })
        self._search_rows = rows

        self.table.setRowCount(0)
        for r in rows:
            i = self.table.rowCount()
            self.table.insertRow(i)
            for c, v in enumerate(
                [r["OPERATION"], r["MODULE"], r["LOT"], r["TESTER"], r["END"]]
            ):
                self.table.setItem(i, c, QTableWidgetItem(v))

        # --- DVI table (check local dirs first, cloud last) ---
        dvi_csv = None
        for d in [self._traceback_dir, out_dir, os.path.join(CloudRoot, vid)]:
            if not d:
                continue
            candidate = os.path.join(d, "DVI_box_data.csv")
            check = _isfile_safe if candidate.startswith("\\\\") else os.path.isfile
            if check(candidate):
                dvi_csv = candidate
                break

        if dvi_csv:
            dvi_rows = read_dvi_csv(dvi_csv)
            if dvi_rows:
                headers = list(dvi_rows[0].keys())
                self.dvi_table.setColumnCount(len(headers))
                self.dvi_table.setHorizontalHeaderLabels(headers)
                self.dvi_table.setRowCount(0)
                for r in dvi_rows:
                    i = self.dvi_table.rowCount()
                    self.dvi_table.insertRow(i)
                    for c, h in enumerate(headers):
                        self.dvi_table.setItem(i, c, QTableWidgetItem(str(r[h])))
                self.dvi_table.resizeColumnsToContents()
                # Fit height to content: header + rows + small margin
                row_h = self.dvi_table.verticalHeader().defaultSectionSize()
                hdr_h = self.dvi_table.horizontalHeader().height()
                self.dvi_table.setFixedHeight(
                    hdr_h + row_h * len(dvi_rows) + 4)
                self.dvi_title.show()
                self.dvi_table.show()
            else:
                self.dvi_title.hide()
                self.dvi_table.hide()
        else:
            self.dvi_title.hide()
            self.dvi_table.hide()

        # --- Thumbnails ---
        if has_vid_csv and rows and data_dir:
            self.load_thumbnails(data_dir, rows)
        elif self._traceback_dir:
            # No vid_data.csv: show images sorted like the backend (proc_sort_key)
            self._load_thumbnails_from_dir(self._traceback_dir)
        else:
            # Nothing found at all
            self._end_searching()
            QMessageBox.warning(
                self, "Not Found",
                f"No data found for {vid}.\n"
                f"Checked {out_dir} and {self._traceback_dir or 'N/A'}")
            return

        self._end_searching()

        # Enable traceback buttons based on available data
        if self._traceback_dir:
            self.btn_manual.setEnabled(True)
            # Auto only when DVI_box_data.csv exists
            has_dvi = dvi_csv is not None
            self.btn_auto.setEnabled(has_dvi)
        else:
            self._set_buttons_enabled(False)

    def _load_thumbnails_from_dir(self, directory):
        """Load thumbnails when vid_data.csv is unavailable.
        Uses proc_sort_key ordering (same as backend), OG images first."""
        while self.flow_layout.count():
            it = self.flow_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        og_files = []
        proc_files = []
        for f in os.listdir(directory):
            fpath = os.path.join(directory, f)
            if not os.path.isfile(fpath):
                continue
            if OG_PAT.match(f):
                og_files.append(f)
            elif PROC_PAT.match(f):
                proc_files.append(f)

        proc_files.sort(key=proc_sort_key)

        for f in og_files:
            path = os.path.join(directory, f)
            self.flow_layout.addWidget(self.make_thumb(path, f"OG: {f}"))
        for f in proc_files:
            path = os.path.join(directory, f)
            self.flow_layout.addWidget(self.make_thumb(path, f))

    # --------------------------------------------------
    # AUTO MODE  (use DVI_box_data.csv)
    # --------------------------------------------------
    def _local_output_dir(self):
        """Always write traceback output locally, even if source is on cloud."""
        local = os.path.join(OutputRoot, f"{self._current_vid}_output")
        os.makedirs(local, exist_ok=True)
        return local

    def on_auto(self):
        if not self._traceback_dir:
            QMessageBox.warning(self, "Error", "No traceback data directory found.")
            return

        dvi_csv = os.path.join(self._traceback_dir, "DVI_box_data.csv")
        if not _isfile_safe(dvi_csv):
            QMessageBox.warning(
                self, "Error",
                "DVI_box_data.csv not found.\nUse Manual mode to draw defect boxes.")
            return

        outdir = self._local_output_dir()
        self._run_traceback(self._traceback_dir, outdir,
                            defect_boxes=None, ref_image_key=None)

    # --------------------------------------------------
    # MANUAL MODE  (draw boxes on latest OG image)
    # --------------------------------------------------
    def on_manual(self):
        if not self._traceback_dir:
            QMessageBox.warning(self, "Error", "No traceback data directory found.")
            return

        og_images = []
        proc_images = []
        for f in sorted(os.listdir(self._traceback_dir)):
            if not os.path.isfile(os.path.join(self._traceback_dir, f)):
                continue
            if OG_PAT.match(f):
                og_images.append(f)
            elif PROC_PAT.match(f):
                proc_images.append(f)

        if og_images:
            # Prefer FRAME2 as reference
            ref_name = og_images[0]
            for f in og_images:
                if "FRAME2" in f.upper():
                    ref_name = f
                    break
        elif proc_images:
            # Fallback: use latest process image (smallest step number, Out)
            proc_images.sort(key=proc_sort_key)
            ref_name = proc_images[0]
        else:
            QMessageBox.warning(self, "Error", "No images found in traceback directory.")
            return

        ref_path = os.path.join(self._traceback_dir, ref_name)

        dlg = DrawingDialog(ref_path, self)
        if dlg.exec() != QDialog.Accepted or not dlg.result_boxes:
            return

        # Convert drawn boxes to DefectBox objects with PIXEL coord_space
        defect_boxes = []
        for i, (bx, by, bw, bh) in enumerate(dlg.result_boxes):
            cx = bx + bw / 2.0
            cy = by + bh / 2.0
            db = DefectBox(
                lot=self._search_rows[0]["LOT"] if self._search_rows else "MANUAL",
                visual_id=self._current_vid,
                dr_result="REJECT",
                dr_sub_item=f"MANUAL-DEFECT-{i+1:02d}",
                box_ctr_x=cx,
                box_ctr_y=cy,
                box_side_x=bw,
                box_side_y=bh,
                image_path=ref_path,
                coord_space="PIXEL",
            )
            defect_boxes.append(db)

        outdir = self._local_output_dir()
        self._run_traceback(self._traceback_dir, outdir,
                            defect_boxes=defect_boxes,
                            ref_image_key=ref_name)

    # --------------------------------------------------
    # RUN TRACEBACK (shared by auto/manual)
    # --------------------------------------------------
    def _run_traceback(self, uploads_dir, outdir, defect_boxes=None, ref_image_key=None):
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Busy", "Traceback is already running.")
            return

        self._set_buttons_enabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        self.status_label.setText("Starting traceback...")

        self._worker = TracebackWorker(
            uploads_dir, outdir, defect_boxes, ref_image_key)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_traceback_done)
        self._worker.error.connect(self._on_traceback_error)
        self._worker.start()

    def _on_progress(self, step, total, message):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(step)
        self.status_label.setText(message)

    def _on_traceback_done(self, result):
        self._last_results = result
        self.progress_bar.setValue(self.progress_bar.maximum())
        self.status_label.setText("Traceback complete!")
        self._set_buttons_enabled(True)

        self._show_results(result)

    def _on_traceback_error(self, msg):
        self.progress_bar.hide()
        self.status_label.setText("")
        self._set_buttons_enabled(True)
        QMessageBox.critical(self, "Traceback Error", msg)

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------
    def _show_results(self, result):
        self.btn_report.show()
        self.btn_clear.show()
        self.btn_download.show()

        # Build per-defect index from origin_summary + output_images
        # origin_summary is [(dr_sub_item, origin), ...]
        # output_images filenames encode defect via PANEL_/ZONE_/TB_ prefix
        all_results = result.get("all_results", [])
        summaries = result.get("origin_summary", [])
        output_images = result.get("output_images", [])

        # Build a list of defect entries, each with its associated images
        self._defect_entries = []
        for d, _verdicts, origin, _panel in all_results:
            # Unique tag used in filenames: dr_sub_item_ctrX_Y
            tag = f"{d.dr_sub_item}_ctr{int(d.box_ctr_x)}_{int(d.box_ctr_y)}"
            # Collect images belonging to this defect
            my_images = [p for p in output_images
                         if tag in os.path.basename(p)]
            # Also grab OG images (shared across defects)
            og_images = [p for p in output_images
                         if os.path.basename(p).upper().startswith("OG_")]
            label = f"{d.dr_sub_item}  ({int(d.box_ctr_x)},{int(d.box_ctr_y)})  ->  {origin}"
            self._defect_entries.append({
                "label": label,
                "tag": tag,
                "origin": origin,
                "images": my_images,
                "og_images": og_images,
            })

        # Populate combo box
        if len(self._defect_entries) > 0:
            self.defect_combo.blockSignals(True)
            self.defect_combo.clear()
            self.defect_combo.addItem(f"All defects ({len(self._defect_entries)})")
            for entry in self._defect_entries:
                self.defect_combo.addItem(entry["label"])
            self.defect_combo.blockSignals(False)
            self.defect_combo_label.show()
            self.defect_combo.show()

        # Show all by default
        self._display_defect_images(-1)

        # Show first panel in popup
        panels = [p for p in output_images
                  if "PANEL_" in os.path.basename(p).upper()]
        if panels:
            self.show_preview(panels[0])
            dlg = PanelViewDialog(panels[0], self)
            dlg.exec()

        # Status
        if summaries:
            parts = [f"{name}: {orig}" for name, orig in summaries]
            self.status_label.setText("Origins: " + " | ".join(parts))

    def _on_defect_selected(self, index):
        """Combo box changed: index 0 = all, 1+ = specific defect."""
        if not self._defect_entries:
            return
        self._display_defect_images(index - 1)

        # Show that defect's panel in preview + popup
        if index >= 1:
            entry = self._defect_entries[index - 1]
            panels = [p for p in entry["images"]
                      if "PANEL_" in os.path.basename(p).upper()]
            if panels and os.path.isfile(panels[0]):
                self.show_preview(panels[0])
                dlg = PanelViewDialog(panels[0], self)
                dlg.exec()

    def _display_defect_images(self, defect_index):
        """Show thumbnails for a specific defect (-1 = all)."""
        while self.flow_layout.count():
            it = self.flow_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()

        if defect_index < 0:
            # Show all defects' zone + panel images
            entries = self._defect_entries
        else:
            entries = [self._defect_entries[defect_index]]

        for entry in entries:
            for img_path in entry["images"]:
                name = os.path.basename(img_path).upper()
                if not (name.startswith("ZONE_") or name.startswith("PANEL_")):
                    continue
                if os.path.isfile(img_path):
                    self.flow_layout.addWidget(
                        self.make_thumb(img_path, os.path.basename(img_path)))

    def _view_report(self):
        if not self._last_results:
            return
        dlg = ReportDialog(
            self._last_results.get("report_text", "No report."),
            self._last_results.get("report_path", "report.txt"),
            self)
        dlg.exec()

    def _download_zip(self):
        """Bundle queried images and result images into a ZIP file."""
        vid = self._current_vid or "export"
        default_name = f"{vid}_traceback.zip"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ZIP", default_name, "ZIP files (*.zip)")
        if not path:
            return

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Result images + report
            if self._last_results:
                for img_path in self._last_results.get("output_images", []):
                    name = os.path.basename(img_path)
                    if name.upper().startswith("OG_"):
                        continue
                    if os.path.isfile(img_path):
                        zf.write(img_path, f"results/{name}")
                rpt = self._last_results.get("report_path", "")
                if rpt and os.path.isfile(rpt):
                    zf.write(rpt, f"results/{os.path.basename(rpt)}")

        QMessageBox.information(self, "Saved", f"ZIP saved to:\n{path}")

    def _clear_results(self):
        self._last_results = None
        self._defect_entries = []
        self.btn_report.hide()
        self.btn_clear.hide()
        self.btn_download.hide()
        self.defect_combo_label.hide()
        self.defect_combo.hide()
        self.defect_combo.clear()
        self.progress_bar.hide()
        self.progress_bar.setValue(0)
        self.status_label.setText("")

        # Reload original thumbnails if search data is available
        if self._current_vid and self._search_rows:
            out_dir = os.path.join(OutputRoot, f"{self._current_vid}_output")
            img_dir = out_dir if os.path.isdir(out_dir) else self._traceback_dir
            if img_dir and _isdir_safe(img_dir):
                self.load_thumbnails(img_dir, self._search_rows)


# ======================================================
# MAIN
# ======================================================
if __name__ == "__main__":
    app = QApplication([])
    viewer = VisionApp()
    viewer.show()
    app.exec()

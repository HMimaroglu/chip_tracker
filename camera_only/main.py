"""
Raspberry Pi Poker Chip Tracker — Webcam + PyQt + ArUco calibration

Features
- PyQt5 setup wizard to configure chip colors → HSV ranges → value, chip thickness, and number of players.
- ArUco-based pixels-per-mm calibration from a printed marker (DICT_4X4_50).
- ROI editor to draw player zones on a live camera feed.
- Real-time stack height estimation from a side webcam (no laser required for v1).
- EMA smoothing, per-player totals, and a lightweight dashboard.

Install (Raspberry Pi OS 64-bit)
    sudo apt update && sudo apt install -y python3-pip libatlas-base-dev
    pip3 install -r requirements.txt

If cv2.aruco is missing, prefer the pip wheel:
    pip3 uninstall -y opencv-python opencv-contrib-python || true
    pip3 install opencv-contrib-python==4.9.0.80

Run
    python3 main.py

Notes
- Put an ArUco marker (DICT_4X4_50) of known side length (e.g., 50.0 mm) in view for calibration.
- Keep the webcam fixed at the table side with consistent lighting.
- Encourage single-color stacks when possible for best accuracy.
"""

import sys
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets


# ---------------------------
# Data classes & Utilities
# ---------------------------
@dataclass
class ChipColor:
    name: str
    hsv_low: Tuple[int, int, int]
    hsv_high: Tuple[int, int, int]
    value: float
    thickness_mm: float = 3.3


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ---------------------------
# Config Dialog (chips & players)
# ---------------------------
class ConfigDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Poker Tracker – Setup")
        self.resize(900, 520)

        layout = QtWidgets.QVBoxLayout(self)

        # Tabs: Chips / Players
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)

        # --- Chips tab ---
        chips_page = QtWidgets.QWidget()
        chips_layout = QtWidgets.QVBoxLayout(chips_page)
        self.table = QtWidgets.QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "Color Name", "H_low", "S_low", "V_low", "H_high", "S_high", "V_high", "Value $", "Thickness mm"
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        chips_layout.addWidget(self.table)

        preset_box = QtWidgets.QHBoxLayout()
        chips_layout.addLayout(preset_box)
        preset_btn = QtWidgets.QPushButton("Add Common Presets")
        preset_btn.clicked.connect(self.add_presets)
        preset_box.addWidget(preset_btn)
        add_row_btn = QtWidgets.QPushButton("Add Row")
        add_row_btn.clicked.connect(self.add_row)
        preset_box.addWidget(add_row_btn)
        remove_row_btn = QtWidgets.QPushButton("Remove Selected")
        remove_row_btn.clicked.connect(self.remove_selected)
        preset_box.addWidget(remove_row_btn)
        preset_box.addStretch(1)

        tabs.addTab(chips_page, "Chip Denominations")

        # --- Players tab ---
        players_page = QtWidgets.QWidget()
        players_layout = QtWidgets.QFormLayout(players_page)
        self.spin_players = QtWidgets.QSpinBox()
        self.spin_players.setRange(2, 9)
        self.spin_players.setValue(4)
        players_layout.addRow("Number of players:", self.spin_players)
        tabs.addTab(players_page, "Players")

        # Buttons
        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

        # Start with some practical presets
        self.add_presets()

    def add_row(self, row_data: Optional[List[str]] = None):
        r = self.table.rowCount()
        self.table.insertRow(r)
        defaults = row_data or ["", "0", "0", "0", "179", "255", "255", "1", "3.3"]
        for c, val in enumerate(defaults):
            item = QtWidgets.QTableWidgetItem(str(val))
            if c != 0:
                item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(r, c, item)

    def remove_selected(self):
        for idx in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(idx)

    def add_presets(self):
        # Basic HSV guesses for common chip colors (tune on your lighting)
        presets = [
            ["Red",   "0",   "120", "80",  "10",  "255", "255", "5", "3.3"],
            ["Blue",  "100", "120", "60",  "130", "255", "255", "10", "3.3"],
            ["Green", "40",  "120", "60",  "80",  "255", "255", "25", "3.3"],
            ["Black", "0",   "0",   "0",   "180", "255", "50",  "100", "3.3"],
        ]
        for p in presets:
            self.add_row(p)

    def get_config(self) -> Dict:
        chips: List[ChipColor] = []
        for r in range(self.table.rowCount()):
            def val(col):
                it = self.table.item(r, col)
                return it.text().strip() if it else ""

            name = val(0) or f"Color{r+1}"
            try:
                Hl = clamp(int(val(1)), 0, 180)
                Sl = clamp(int(val(2)), 0, 255)
                Vl = clamp(int(val(3)), 0, 255)
                Hh = clamp(int(val(4)), 0, 180)
                Sh = clamp(int(val(5)), 0, 255)
                Vh = clamp(int(val(6)), 0, 255)
                value = float(val(7))
                thick = float(val(8) or 3.3)
            except Exception:
                continue
            chips.append(ChipColor(name, (Hl, Sl, Vl), (Hh, Sh, Vh), value, thick))
        cfg = {
            "players": int(self.spin_players.value()),
            "chips": [c.__dict__ for c in chips],
        }
        return cfg


# ---------------------------
# Camera feed widget base
# ---------------------------
class CameraWidget(QtWidgets.QWidget):
    frame_ready = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, cam_index=0, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_tick)
        self.label = QtWidgets.QLabel()
        self.label.setMinimumSize(640, 360)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(self.label)

    def start(self):
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.timer.start(33)  # ~30 FPS

    def stop(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _on_tick(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        self.frame_ready.emit(frame)
        # show frame (may be annotated by subclass before display)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))


# ---------------------------
# ArUco Calibrator
# ---------------------------
class Calibrator(CameraWidget):
    calibrated = QtCore.pyqtSignal(float)  # ppmm

    def __init__(self, cam_index=0, parent=None):
        super().__init__(cam_index=cam_index, parent=parent)
        self.marker_mm = 50.0
        self.ppmm = 0.0
        self.info = QtWidgets.QLabel("Place an ArUco (DICT_4X4_50) marker and click Capture.")
        self.btn_capture = QtWidgets.QPushButton("Capture Calibration")
        self.btn_capture.clicked.connect(self.capture)

        lay = self.layout()
        lay.addWidget(self.info)
        lay.addWidget(self.btn_capture)

        self.dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        # OpenCV API compatibility (4.7+ vs 4.10+)
        # Prefer new-style DetectorParameters() + ArucoDetector, fallback to legacy *_create + detectMarkers
        self.detector = None
        try:
            # OpenCV >= 4.7 legacy factory may not exist
            self.params = cv2.aruco.DetectorParameters()  # new API
        except AttributeError:
            # Older API
            self.params = getattr(cv2.aruco, 'DetectorParameters_create')()
        # Try to build a dedicated detector (new API); if it fails we'll use detectMarkers(..., parameters=...)
        try:
            self.detector = cv2.aruco.ArucoDetector(self.dict, self.params)
        except Exception:
            self.detector = None

        # Input for marker size
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("Marker side length (mm):"))
        self.edit_size = QtWidgets.QDoubleSpinBox()
        self.edit_size.setRange(10.0, 200.0)
        self.edit_size.setValue(self.marker_mm)
        self.edit_size.setDecimals(1)
        size_row.addWidget(self.edit_size)
        self.layout().addLayout(size_row)

    def capture(self):
        if self.cap is None:
            return
        # Grab a quick burst and average measurement for robustness
        px_sizes = []
        for _ in range(10):
            ok, frame = self.cap.read()
            if not ok:
                continue
            # Use new API if available
            if self.detector is not None:
                corners, ids, _ = self.detector.detectMarkers(frame)
            else:
                corners, ids, _ = cv2.aruco.detectMarkers(frame, self.dict, parameters=self.params)
            if corners:
                for c in corners:
                    c = c[0]
                    # side length in pixels: average of 4 edges
                    d01 = np.linalg.norm(c[0] - c[1])
                    d12 = np.linalg.norm(c[1] - c[2])
                    d23 = np.linalg.norm(c[2] - c[3])
                    d30 = np.linalg.norm(c[3] - c[0])
                    px_sizes.append(np.mean([d01, d12, d23, d30]))
        if not px_sizes:
            self.info.setText("No marker detected. Ensure DICT_4X4_50 is visible.")
            return
        px = float(np.median(px_sizes))
        self.marker_mm = float(self.edit_size.value())
        self.ppmm = px / self.marker_mm
        self.info.setText(f"Calibrated: {self.ppmm:.3f} px/mm (median of {len(px_sizes)} samples)")
        self.calibrated.emit(self.ppmm)


# ---------------------------
# Zone Editor (draw N ROIs)
# ---------------------------
class ZoneEditor(CameraWidget):
    zones_defined = QtCore.pyqtSignal(list)  # list of (x0,y0,x1,y1) normalized

    def __init__(self, n_players: int, cam_index=0, parent=None):
        super().__init__(cam_index=cam_index, parent=parent)
        self.n_players = n_players
        self._zones_px: List[Tuple[int,int,int,int]] = []
        self._drawing = False
        self._start = None
        self.setWindowTitle("Draw Player Zones (rectangles) – Press D when done")

    def mousePressEvent(self, ev: QtGui.QMouseEvent):
        if not self.label.pixmap():
            return
        self._drawing = True
        self._start = ev.pos()

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent):
        pass

    def mouseReleaseEvent(self, ev: QtGui.QMouseEvent):
        if not self._drawing or self._start is None or self.label.pixmap() is None:
            return
        self._drawing = False
        end = ev.pos()
        # Map widget coords → image coords
        pix = self.label.pixmap()
        w, h = pix.width(), pix.height()
        x0 = clamp(min(self._start.x(), end.x()), 0, w-1)
        y0 = clamp(min(self._start.y(), end.y()), 0, h-1)
        x1 = clamp(max(self._start.x(), end.x()), 0, w-1)
        y1 = clamp(max(self._start.y(), end.y()), 0, h-1)
        self._zones_px.append((x0,y0,x1,y1))
        self.update_overlay()

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        if ev.key() == QtCore.Qt.Key_Backspace and self._zones_px:
            self._zones_px.pop()
            self.update_overlay()
        elif ev.key() == QtCore.Qt.Key_D:
            if len(self._zones_px) != self.n_players:
                QtWidgets.QMessageBox.warning(self, "Zones", f"Define exactly {self.n_players} zones. Use Backspace to undo.")
                return
            # Convert to normalized coords relative to current frame size
            if self.cap is None:
                return
            ok, frame = self.cap.read()
            if not ok:
                return
            H, W = frame.shape[:2]
            zones_norm = []
            for (x0,y0,x1,y1) in self._zones_px:
                zones_norm.append((x0/W, y0/H, x1/W, y1/H))
            self.zones_defined.emit(zones_norm)
            self.close()

    def update_overlay(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        for (x0,y0,x1,y1) in self._zones_px:
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
        self.label.setPixmap(QtGui.QPixmap.fromImage(qimg))


# ---------------------------
# ChipTracker — per-frame processing
# ---------------------------
class ChipTracker:
    def __init__(self, config: Dict, zones_norm: List[Tuple[float,float,float,float]], ppmm: float):
        self.cfg = config
        self.zones_norm = zones_norm
        self.ppmm = ppmm
        self.ema_alpha = 0.3
        # State: per player per color EMA heights (mm)
        self.state: Dict[Tuple[int, str], float] = {}

    def process(self, frame: np.ndarray) -> Tuple[List[float], np.ndarray]:
        H, W = frame.shape[:2]
        overlay = frame.copy()
        players = self.cfg["players"]
        chip_defs = [ChipColor(**c) for c in self.cfg["chips"]]

        totals = [0.0 for _ in range(players)]

        for p_idx, (x0n,y0n,x1n,y1n) in enumerate(self.zones_norm):
            x0, y0, x1, y1 = int(x0n*W), int(y0n*H), int(x1n*W), int(y1n*H)
            roi = frame[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            cv2.rectangle(overlay, (x0,y0), (x1,y1), (0,255,0), 2)

            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            base_y_px = roi.shape[0] - 1  # assume bottom of ROI ~ table baseline

            for chip in chip_defs:
                low = np.array(chip.hsv_low, dtype=np.uint8)
                high = np.array(chip.hsv_high, dtype=np.uint8)
                mask = cv2.inRange(hsv, low, high)
                mask = cv2.medianBlur(mask, 5)

                # Find the top-most pixel (smallest y) that belongs to the chip color in this zone
                ys, xs = np.where(mask > 0)
                if ys.size == 0:
                    key = (p_idx, chip.name)
                    # decay to zero slowly when not seen
                    if key in self.state:
                        self.state[key] = (1 - self.ema_alpha) * self.state[key]
                    continue

                top_y = int(np.min(ys))
                height_px = clamp(base_y_px - top_y, 0, roi.shape[0])
                height_mm = height_px / max(self.ppmm, 1e-6)

                # EMA smoothing
                key = (p_idx, chip.name)
                prev = self.state.get(key, height_mm)
                smoothed = self.ema_alpha * height_mm + (1 - self.ema_alpha) * prev
                self.state[key] = smoothed

                # Convert to chips; conservative rounding
                chips_est = int(round(smoothed / max(chip.thickness_mm, 1e-6)))
                value = chips_est * chip.value
                totals[p_idx] += value

                # Draw diagnostics
                cv2.putText(overlay, f"P{p_idx+1} {chip.name}: {chips_est} (~${value:.2f})",
                            (x0+6, y0+18 + 18*list(self.cfg["chips"]).index(chip.__dict__)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1, cv2.LINE_AA)

        # Draw per-player totals
        for i, tot in enumerate(totals):
            cv2.putText(overlay, f"Player {i+1}: ${tot:.2f}", (10, 24 + 22*i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2, cv2.LINE_AA)

        pot = sum(totals)
        cv2.putText(overlay, f"POT: ${pot:.2f}", (10, 24 + 22*len(totals) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        return totals, overlay


def enumerate_cameras():
    """Find all available camera indices"""
    cameras = []
    # Suppress OpenCV warnings during camera enumeration
    import logging
    logging.getLogger('opencv').setLevel(logging.ERROR)
    
    for i in range(5):  # Reduced range to minimize errors
        try:
            cap = cv2.VideoCapture(i)
            # Try to read a frame to verify camera actually works
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cameras.append(i)
            cap.release()
        except Exception:
            pass
    return cameras


# ---------------------------
# Main Window (run-time loop)
# ---------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poker Chip Tracker – Webcam Prototype")
        self.resize(980, 700)

        self.cfg: Optional[Dict] = None
        self.ppmm: float = 0.0
        self.zones_norm: Optional[List[Tuple[float,float,float,float]]] = None
        self.camera_index: int = 0

        # Central UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        vbox = QtWidgets.QVBoxLayout(central)

        # Video label
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(960, 540)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        vbox.addWidget(self.video_label)

        # Camera selection
        camera_row = QtWidgets.QHBoxLayout()
        vbox.addLayout(camera_row)
        camera_row.addWidget(QtWidgets.QLabel("Camera Source:"))
        self.camera_combo = QtWidgets.QComboBox()
        self.refresh_cameras()
        camera_row.addWidget(self.camera_combo)
        refresh_btn = QtWidgets.QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_cameras)
        camera_row.addWidget(refresh_btn)
        camera_row.addStretch(1)

        # Controls
        btn_row = QtWidgets.QHBoxLayout()
        vbox.addLayout(btn_row)
        self.btn_setup = QtWidgets.QPushButton("1) Configure Chips & Players")
        self.btn_setup.clicked.connect(self.on_setup)
        btn_row.addWidget(self.btn_setup)

        self.btn_calib = QtWidgets.QPushButton("2) Calibrate ArUco")
        self.btn_calib.clicked.connect(self.on_calibrate)
        self.btn_calib.setEnabled(False)
        btn_row.addWidget(self.btn_calib)

        self.btn_zones = QtWidgets.QPushButton("3) Define Player Zones")
        self.btn_zones.clicked.connect(self.on_zones)
        self.btn_zones.setEnabled(False)
        btn_row.addWidget(self.btn_zones)

        self.btn_start = QtWidgets.QPushButton("4) Start Tracking")
        self.btn_start.clicked.connect(self.on_start)
        self.btn_start.setEnabled(False)
        btn_row.addWidget(self.btn_start)

        btn_row.addStretch(1)

        # Timer for runtime
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.tracker: Optional[ChipTracker] = None

        # Status bar
        self.statusBar().showMessage("Ready.")

    def refresh_cameras(self):
        """Refresh the list of available cameras"""
        self.camera_combo.clear()
        cameras = enumerate_cameras()
        if not cameras:
            self.camera_combo.addItem("No cameras found", -1)
            self.camera_combo.setEnabled(False)
        else:
            for cam_idx in cameras:
                self.camera_combo.addItem(f"Camera {cam_idx}", cam_idx)
            self.camera_combo.setEnabled(True)
            if cameras:
                self.camera_index = cameras[0]

    def get_selected_camera(self):
        """Get the currently selected camera index"""
        data = self.camera_combo.currentData()
        return data if data is not None and data >= 0 else 0

    # 1) Setup chips + players
    def on_setup(self):
        dlg = ConfigDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.cfg = dlg.get_config()
            if not self.cfg.get("chips"):
                QtWidgets.QMessageBox.warning(self, "Config", "Please configure at least one chip color.")
                return
            self.statusBar().showMessage("Config saved. Proceed to calibration.")
            self.btn_calib.setEnabled(True)
            # Save to config.json
            with open("config.json", "w") as f:
                json.dump(self.cfg, f, indent=2)

    # 2) Calibrate ArUco → ppmm
    def on_calibrate(self):
        cam_idx = self.get_selected_camera()
        self.calib = Calibrator(cam_idx, self)
        self.calib.calibrated.connect(self.on_calibrated)
        self.calib.show()
        self.calib.start()

    def on_calibrated(self, ppmm: float):
        self.ppmm = ppmm
        self.statusBar().showMessage(f"Calibrated: {ppmm:.3f} px/mm. Define player zones.")
        self.btn_zones.setEnabled(True)
        self.calib.stop()
        self.calib.close()

    # 3) Draw player zones
    def on_zones(self):
        if not self.cfg:
            return
        n = int(self.cfg.get("players", 4))
        cam_idx = self.get_selected_camera()
        self.zone_editor = ZoneEditor(n, cam_idx, self)
        self.zone_editor.zones_defined.connect(self.on_zones_defined)
        self.zone_editor.show()
        self.zone_editor.start()

    def on_zones_defined(self, zones_norm: List[Tuple[float,float,float,float]]):
        self.zones_norm = zones_norm
        self.statusBar().showMessage("Zones saved. You can start tracking.")
        self.btn_start.setEnabled(True)
        if self.cfg is not None:
            self.cfg["zones_norm"] = zones_norm
            with open("config.json", "w") as f:
                json.dump(self.cfg, f, indent=2)

    # 4) Start runtime
    def on_start(self):
        if not (self.cfg and self.ppmm > 0 and self.zones_norm):
            QtWidgets.QMessageBox.warning(self, "Start", "Complete setup, calibration, and zones first.")
            return
        cam_idx = self.get_selected_camera()
        self.cap = cv2.VideoCapture(cam_idx)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.tracker = ChipTracker(self.cfg, self.zones_norm, self.ppmm)
        self.timer.start(33)
        self.statusBar().showMessage("Tracking… Press Esc to stop.")

    def on_tick(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        totals, overlay = self.tracker.process(frame)
        rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QtGui.QImage.Format_RGB888)
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qimg))

    def keyPressEvent(self, ev: QtGui.QKeyEvent):
        if ev.key() == QtCore.Qt.Key_Escape:
            self.shutdown()

    def shutdown(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.close()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

"""
PyQt ROI editor for drawing player zones on live camera feed.
"""
import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .camera import CameraCapture
from .schemas import ROI, NormalizedROI

logger = logging.getLogger(__name__)


class VideoLabel(QtWidgets.QLabel):
    """Custom QLabel for video display with mouse event handling."""
    
    mouse_press = QtCore.pyqtSignal(QtCore.QPoint)
    mouse_move = QtCore.pyqtSignal(QtCore.QPoint)
    mouse_release = QtCore.pyqtSignal(QtCore.QPoint)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setStyleSheet("border: 2px solid black; background-color: #f0f0f0;")
        self.setText("Initializing camera...")
        self.setMouseTracking(False)  # Only track when needed
    
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_press.emit(event.pos())
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):
        self.mouse_move.emit(event.pos())
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.mouse_release.emit(event.pos())
        super().mouseReleaseEvent(event)


class ROIEditor(QtWidgets.QDialog):
    """Interactive ROI editor dialog."""
    
    zones_defined = QtCore.pyqtSignal(list)  # List[NormalizedROI]
    
    def __init__(self, camera_index: int, num_players: int, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.num_players = num_players
        self.rois: List[ROI] = []
        self.drawing = False
        self.start_point: Optional[QtCore.QPoint] = None
        self.current_point: Optional[QtCore.QPoint] = None
        self.current_frame: Optional[np.ndarray] = None
        self.frame_size = (640, 480)  # Default size
        self.last_mouse_move_time = 0  # Throttle mouse moves
        self.frame_skip_counter = 0  # Skip frames when drawing
        
        self.setWindowTitle(f"Draw {num_players} Player Zones - Click and drag rectangles")
        self.setModal(True)
        self.resize(1000, 700)
        
        # Setup UI
        self.setup_ui()
        
        # Setup camera
        self.camera = CameraCapture(camera_index)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        
    def setup_ui(self):
        """Setup the user interface."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Video display with custom mouse handling
        self.video_label = VideoLabel()
        self.video_label.mouse_press.connect(self.on_mouse_press)
        self.video_label.mouse_move.connect(self.on_mouse_move) 
        self.video_label.mouse_release.connect(self.on_mouse_release)
        layout.addWidget(self.video_label)
        
        # Instructions
        self.instructions_label = QtWidgets.QLabel()
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setStyleSheet("padding: 10px; background-color: #e8f4fd; border: 1px solid #bee5eb; border-radius: 4px;")
        self.update_instructions()
        layout.addWidget(self.instructions_label)
        
        # Controls
        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls)
        
        self.clear_button = QtWidgets.QPushButton("🗑️ Clear Last Zone")
        self.clear_button.clicked.connect(self.clear_last_zone)
        self.clear_button.setEnabled(False)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: #212529;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #ffffff;
            }
        """)
        controls.addWidget(self.clear_button)
        
        self.clear_all_button = QtWidgets.QPushButton("🗑️ Clear All")
        self.clear_all_button.clicked.connect(self.clear_all_zones)
        self.clear_all_button.setEnabled(False)
        self.clear_all_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #ffffff;
            }
        """)
        controls.addWidget(self.clear_all_button)
        
        controls.addStretch()
        
        self.done_button = QtWidgets.QPushButton("✅ Done")
        self.done_button.clicked.connect(self.finish_editing)
        self.done_button.setEnabled(False)
        self.done_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                color: white;
                border: none;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #ffffff;
            }
        """)
        controls.addWidget(self.done_button)
        
        self.cancel_button = QtWidgets.QPushButton("❌ Cancel")
        self.cancel_button.clicked.connect(self.close)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
        """)
        controls.addWidget(self.cancel_button)
    
    def update_instructions(self):
        """Update the instructions text."""
        num_zones = len(self.rois)
        self.instructions_label.setText(
            f"📍 Draw {self.num_players} rectangular zones by clicking and dragging\n"
            f"✅ Zones created: {num_zones}/{self.num_players}\n"
            f"💡 Tip: Click 'Clear Last Zone' if you make a mistake"
        )
        
    def update_button_states(self):
        """Update button enabled states."""
        has_zones = len(self.rois) > 0
        is_complete = len(self.rois) == self.num_players
        
        self.clear_button.setEnabled(has_zones)
        self.clear_all_button.setEnabled(has_zones)
        self.done_button.setEnabled(is_complete)
        
        logger.debug(f"Button states updated: zones={len(self.rois)}/{self.num_players}, done_enabled={is_complete}")
    
    def start_editing(self):
        """Start the ROI editing process."""
        if not self.camera.open():
            QtWidgets.QMessageBox.critical(self, "Camera Error", f"Failed to open camera {self.camera_index}")
            return False
        
        self.timer.start(50)  # Reduce to 20 FPS for better performance
        return True
        
    def update_frame(self):
        """Update the video display with current camera frame."""
        # Skip frames when actively drawing to reduce CPU load
        if self.drawing:
            self.frame_skip_counter += 1
            if self.frame_skip_counter < 3:  # Skip 2 out of 3 frames while drawing
                return
            self.frame_skip_counter = 0
        
        frame = self.camera.read()
        if frame is None:
            return
        
        # Only copy frame if it's different from last one (basic change detection)
        if (self.current_frame is None or 
            not np.array_equal(frame[::4, ::4], self.current_frame[::4, ::4])):
            self.current_frame = frame.copy()
        else:
            return  # Skip if frame hasn't changed significantly
            
        # Store frame info
        self.frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
        
        # Use the original frame as base (avoid unnecessary copy)
        display_frame = frame
        
        # Draw existing ROIs
        for i, roi in enumerate(self.rois):
            color = (0, 255, 0)  # Green
            thickness = 2  # Reduced thickness for better performance
            cv2.rectangle(display_frame, (roi.x0, roi.y0), (roi.x1, roi.y1), color, thickness)
            
            # Simplified label drawing
            label = f"P{i+1}"
            cv2.putText(display_frame, label, (roi.x0+5, roi.y0-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw current drawing rectangle if in progress
        if self.drawing and self.start_point and self.current_point:
            start_img = self.widget_to_image_coords(self.start_point)
            current_img = self.widget_to_image_coords(self.current_point)
            if start_img and current_img:
                cv2.rectangle(display_frame, start_img, current_img, (0, 0, 255), 2)  # Red while drawing
        
        # Convert to Qt format and display with optimized settings
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        
        # Use faster scaling
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
    def on_mouse_press(self, pos: QtCore.QPoint):
        """Handle mouse press events for drawing ROIs."""
        if len(self.rois) < self.num_players:
            self.drawing = True
            self.start_point = pos
            self.current_point = pos
            # Enable mouse tracking only while drawing
            self.video_label.setMouseTracking(True)
            
    def on_mouse_move(self, pos: QtCore.QPoint):
        """Handle mouse move events while drawing."""
        if self.drawing:
            # Throttle mouse move events to prevent excessive updates
            import time
            current_time = time.time() * 1000  # Convert to milliseconds
            if current_time - self.last_mouse_move_time < 16:  # Max ~60 FPS for mouse moves
                return
            self.last_mouse_move_time = current_time
            self.current_point = pos
        
    def on_mouse_release(self, pos: QtCore.QPoint):
        """Handle mouse release events to complete ROI drawing."""
        if self.drawing and self.start_point is not None:
            self.drawing = False
            self.current_point = pos
            
            # Convert widget coordinates to image coordinates
            roi = self.widget_coords_to_roi(self.start_point, pos)
            if roi and self.is_valid_roi(roi):
                self.rois.append(roi)
                self.update_ui_state()  # Update UI immediately
                logger.info(f"ROI added: {len(self.rois)}/{self.num_players} zones created")
            
            self.start_point = None
            self.current_point = None
            # Reset frame skip counter and disable mouse tracking
            self.frame_skip_counter = 0
            self.video_label.setMouseTracking(False)
            
    def widget_to_image_coords(self, widget_pos: QtCore.QPoint) -> Optional[tuple]:
        """Convert widget coordinates to image coordinates."""
        if self.current_frame is None or self.frame_size == (640, 480):
            return None
            
        # Get the actual displayed pixmap
        pixmap = self.video_label.pixmap()
        if pixmap is None:
            return None
        
        # Cache label and pixmap sizes to avoid repeated calls
        label_size = self.video_label.size()
        pixmap_size = pixmap.size()
        
        # Account for centering within the label
        x_offset = (label_size.width() - pixmap_size.width()) // 2
        y_offset = (label_size.height() - pixmap_size.height()) // 2
        
        # Check if click is within the image
        rel_x = widget_pos.x() - x_offset
        rel_y = widget_pos.y() - y_offset
        
        if (rel_x < 0 or rel_x >= pixmap_size.width() or 
            rel_y < 0 or rel_y >= pixmap_size.height()):
            return None
        
        # Scale to image coordinates using cached frame_size
        frame_width, frame_height = self.frame_size
        scale_x = frame_width / pixmap_size.width()
        scale_y = frame_height / pixmap_size.height()
        
        img_x = int(rel_x * scale_x)
        img_y = int(rel_y * scale_y)
        
        # Clamp to image bounds
        img_x = max(0, min(img_x, frame_width - 1))
        img_y = max(0, min(img_y, frame_height - 1))
        
        return (img_x, img_y)
    
    def widget_coords_to_roi(self, start: QtCore.QPoint, end: QtCore.QPoint) -> Optional[ROI]:
        """Convert widget coordinates to image ROI coordinates."""
        start_img = self.widget_to_image_coords(start)
        end_img = self.widget_to_image_coords(end)
        
        if not start_img or not end_img:
            logger.warning(f"Failed to convert widget coordinates to image coordinates")
            return None
        
        # Ensure proper order (top-left to bottom-right)
        x0 = min(start_img[0], end_img[0])
        y0 = min(start_img[1], end_img[1])
        x1 = max(start_img[0], end_img[0])
        y1 = max(start_img[1], end_img[1])
        
        # Minimum size check (30x30 pixels - reduced for better UX)
        width = x1 - x0
        height = y1 - y0
        if width < 30 or height < 30:
            logger.warning(f"ROI too small: {width}x{height} pixels (minimum 30x30)")
            return None
            
        logger.debug(f"Created ROI: {width}x{height} pixels at ({x0},{y0})")
        return ROI(x0=x0, y0=y0, x1=x1, y1=y1)
        
    def is_valid_roi(self, roi: ROI) -> bool:
        """Check if ROI is valid (minimum size, within bounds)."""
        is_valid = (roi.x1 > roi.x0 + 20 and 
                   roi.y1 > roi.y0 + 20 and
                   roi.x0 >= 0 and 
                   roi.y0 >= 0)
        
        if not is_valid:
            logger.warning(f"Invalid ROI rejected: ({roi.x0},{roi.y0})-({roi.x1},{roi.y1})")
        else:
            logger.debug(f"Valid ROI accepted: ({roi.x0},{roi.y0})-({roi.x1},{roi.y1})")
            
        return is_valid
                
    def clear_last_zone(self):
        """Remove the last drawn zone."""
        try:
            if self.rois:
                self.rois.pop()
                self.update_ui_state()
                logger.info(f"Cleared last zone, {len(self.rois)} zones remaining")
        except Exception as e:
            logger.error(f"Error clearing last zone: {e}")
            
    def clear_all_zones(self):
        """Clear all drawn zones."""
        try:
            self.rois.clear()
            self.update_ui_state()
            logger.info("Cleared all zones")
        except Exception as e:
            logger.error(f"Error clearing all zones: {e}")
        
    def update_ui_state(self):
        """Update UI elements based on current state."""
        try:
            self.update_instructions()
            self.update_button_states()
            # Force UI update to ensure button state changes are immediately visible
            QtWidgets.QApplication.processEvents()
        except Exception as e:
            logger.error(f"Error updating UI state: {e}")
        
    def finish_editing(self):
        """Complete ROI editing and emit results."""
        try:
            if len(self.rois) != self.num_players:
                QtWidgets.QMessageBox.warning(
                    self, "Incomplete Zones", 
                    f"📍 Please draw exactly {self.num_players} zones.\n"
                    f"✅ Currently have: {len(self.rois)}/{self.num_players}\n\n"
                    f"💡 Click and drag to create rectangular zones for each player."
                )
                return
                
            if not self.frame_size or self.frame_size == (640, 480):  # Default size means no frame
                QtWidgets.QMessageBox.critical(self, "Camera Error", 
                                             "❌ No camera frame available.\n"
                                             "Please ensure the camera is working.")
                return
                
            # Convert to normalized coordinates
            frame_width, frame_height = self.frame_size
            normalized_rois = []
            
            for roi in self.rois:
                normalized_roi = roi.to_normalized(frame_width, frame_height)
                normalized_rois.append(normalized_roi)
            
            logger.info(f"ROI editing completed with {len(normalized_rois)} zones")
            self.zones_defined.emit(normalized_rois)
            self.accept()  # Use accept() instead of close() for dialog
            
        except Exception as e:
            logger.error(f"Error finishing ROI editing: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to complete zone editing: {str(e)}")
        
    def closeEvent(self, event):
        """Handle window close event."""
        try:
            self.timer.stop()
            self.camera.close()
            logger.info("ROI editor closed")
        except Exception as e:
            logger.error(f"Error closing ROI editor: {e}")
        finally:
            event.accept()
    
    def reject(self):
        """Handle dialog rejection (Cancel button)."""
        try:
            self.timer.stop() 
            self.camera.close()
            logger.info("ROI editor cancelled")
        except Exception as e:
            logger.error(f"Error cancelling ROI editor: {e}")
        finally:
            super().reject()
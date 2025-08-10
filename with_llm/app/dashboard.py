"""
Lightweight PyQt dashboard widget for displaying chip counting results.
"""
import logging
from typing import List, Optional
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
import cv2

from .schemas import DashboardState, ColorSpec, NormalizedROI
from .camera import CameraCapture

logger = logging.getLogger(__name__)


class DashboardWidget(QtWidgets.QWidget):
    """Main dashboard widget for displaying poker chip counts."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = DashboardState()
        self.camera: Optional[CameraCapture] = None
        self.rois: List[NormalizedROI] = []
        self.colors: List[ColorSpec] = []
        self.current_frame: Optional[np.ndarray] = None
        
        self.setup_ui()
        
        # Timer for updating display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_display)
        
    def setup_ui(self):
        """Setup the user interface with Apple-inspired design."""
        # Set transparent background for main widget
        self.setStyleSheet("background: transparent; border: none;")
        
        layout = QtWidgets.QHBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Left side - Video feed
        video_container = QtWidgets.QWidget()
        video_container.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 16px;
                border: 1px solid #e0e0e0;
            }
        """)
        video_layout = QtWidgets.QVBoxLayout(video_container)
        video_layout.setContentsMargins(16, 16, 16, 16)
        video_layout.setSpacing(12)
        
        # Video feed with rounded corners
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #f8f9fa, stop: 1 #e9ecef);
                border-radius: 12px;
                border: 2px dashed #d0d7de;
                color: #656d76;
                font-size: 16px;
                font-weight: 500;
            }
        """)
        self.video_label.setText("📹 Camera feed will appear here")
        video_layout.addWidget(self.video_label)
        
        layout.addWidget(video_container, 2)  # 2/3 of space
        
        # Right side - Stats panel
        stats_container = QtWidgets.QWidget()
        stats_container.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 16px;
                border: 1px solid #e0e0e0;
            }
        """)
        stats_layout = QtWidgets.QVBoxLayout(stats_container)
        stats_layout.setContentsMargins(20, 20, 20, 20)
        stats_layout.setSpacing(16)
        
        # Title with icon
        title = QtWidgets.QLabel("📊 Live Analytics")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("""
            QLabel {
                font-size: 20px;
                font-weight: 700;
                color: #1d1d1f;
                background: transparent;
                border: none;
                padding: 8px;
            }
        """)
        stats_layout.addWidget(title)
        
        # Stats cards container
        stats_cards = QtWidgets.QVBoxLayout()
        stats_cards.setSpacing(8)
        
        # Provider card
        self.provider_card = self.create_info_card("🤖 Provider", "None")
        stats_cards.addWidget(self.provider_card)
        
        # Performance card
        self.perf_card = self.create_info_card("⚡ Inference", "-- ms")
        stats_cards.addWidget(self.perf_card)
        
        # Next audit card  
        self.countdown_card = self.create_info_card("⏰ Next Audit", "-- s")
        stats_cards.addWidget(self.countdown_card)
        
        stats_layout.addLayout(stats_cards)
        
        # Player totals section
        players_title = QtWidgets.QLabel("👥 Player Totals")
        players_title.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: 600;
                color: #1d1d1f;
                background: transparent;
                border: none;
                padding: 8px 0px;
            }
        """)
        stats_layout.addWidget(players_title)
        
        # Player cards container
        self.players_container = QtWidgets.QVBoxLayout()
        self.players_container.setSpacing(6)
        stats_layout.addLayout(self.players_container)
        
        self.player_labels: List[QtWidgets.QLabel] = []
        
        # Pot total - prominent display
        self.pot_label = QtWidgets.QLabel("💰 POT: $0.00")
        self.pot_label.setAlignment(QtCore.Qt.AlignCenter)
        self.pot_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: 700;
                color: #ffffff;
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #FF6B35, stop: 1 #F7931E);
                border-radius: 12px;
                padding: 16px;
                margin: 8px 0px;
                border: none;
            }
        """)
        stats_layout.addWidget(self.pot_label)
        
        # Error display
        self.error_label = QtWidgets.QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("""
            QLabel {
                color: #FF3B30;
                background: rgba(255, 59, 48, 0.1);
                border: 1px solid #FF3B30;
                border-radius: 8px;
                padding: 12px;
                font-size: 14px;
                font-weight: 500;
            }
        """)
        self.error_label.hide()
        stats_layout.addWidget(self.error_label)
        
        stats_layout.addStretch()
        layout.addWidget(stats_container, 1)  # 1/3 of space
    
    def create_info_card(self, title: str, value: str) -> QtWidgets.QWidget:
        """Create an information card with title and value."""
        card = QtWidgets.QWidget()
        card.setStyleSheet("""
            QWidget {
                background: rgba(0, 122, 255, 0.05);
                border-radius: 10px;
                border: 1px solid rgba(0, 122, 255, 0.1);
            }
        """)
        
        layout = QtWidgets.QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        
        # Title
        title_label = QtWidgets.QLabel(title)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                font-weight: 500;
                color: #8e8e93;
                background: transparent;
                border: none;
            }
        """)
        layout.addWidget(title_label)
        
        # Value
        value_label = QtWidgets.QLabel(value)
        value_label.setStyleSheet("""
            QLabel {
                font-size: 16px;
                font-weight: 600;
                color: #1d1d1f;
                background: transparent;
                border: none;
            }
        """)
        layout.addWidget(value_label)
        
        # Store value label for updates
        card.value_label = value_label
        
        return card
    
    def create_player_card(self, player_num: int, total: float) -> QtWidgets.QWidget:
        """Create a player total card."""
        card = QtWidgets.QWidget()
        card.setStyleSheet("""
            QWidget {
                background: rgba(52, 199, 89, 0.05);
                border-radius: 10px;
                border: 1px solid rgba(52, 199, 89, 0.1);
            }
        """)
        
        layout = QtWidgets.QHBoxLayout(card)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)
        
        # Player icon and number
        player_label = QtWidgets.QLabel(f"👤 Player {player_num}")
        player_label.setStyleSheet("""
            QLabel {
                font-size: 13px;
                font-weight: 600;
                color: #1d1d1f;
                background: transparent;
                border: none;
            }
        """)
        layout.addWidget(player_label)
        
        layout.addStretch()
        
        # Total value
        total_label = QtWidgets.QLabel(f"${total:.2f}")
        total_label.setStyleSheet("""
            QLabel {
                font-size: 15px;
                font-weight: 700;
                color: #34C759;
                background: transparent;
                border: none;
            }
        """)
        layout.addWidget(total_label)
        
        # Store total label for updates
        card.total_label = total_label
        card.player_num = player_num
        
        return card
    
    def set_players(self, num_players: int):
        """Setup player cards for the given number of players."""
        # Clear existing player cards
        for label in self.player_labels:
            label.deleteLater()
        self.player_labels.clear()
        
        # Clear existing widgets from players container
        while self.players_container.count():
            child = self.players_container.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        
        # Create new player cards
        for i in range(num_players):
            player_card = self.create_player_card(i + 1, 0.00)
            self.player_labels.append(player_card)
            self.players_container.addWidget(player_card)
    
    def start_camera(self, camera_index: int):
        """Start the camera feed."""
        if self.camera:
            self.camera.close()
            
        self.camera = CameraCapture(camera_index)
        if self.camera.open():
            self.timer.start(33)  # ~30 FPS
            logger.info(f"Camera {camera_index} started for dashboard")
            return True
        else:
            logger.error(f"Failed to start camera {camera_index}")
            return False
    
    def stop_camera(self):
        """Stop the camera feed."""
        self.timer.stop()
        if self.camera:
            self.camera.close()
            self.camera = None
    
    def update_display(self):
        """Update the video display and overlays."""
        if not self.camera:
            return
            
        frame = self.camera.read()
        if frame is None:
            return
            
        self.current_frame = frame.copy()
        display_frame = frame.copy()
        
        # Draw ROI overlays
        if self.rois and len(self.rois) > 0:
            frame_height, frame_width = frame.shape[:2]
            
            for i, roi in enumerate(self.rois):
                # Convert normalized to pixel coordinates
                pixel_roi = roi.to_pixel(frame_width, frame_height)
                
                # Draw rectangle
                color = (0, 255, 0)  # Green
                cv2.rectangle(display_frame, 
                             (pixel_roi.x0, pixel_roi.y0), 
                             (pixel_roi.x1, pixel_roi.y1), 
                             color, 2)
                
                # Add player label
                cv2.putText(display_frame, f"P{i+1}", 
                           (pixel_roi.x0, pixel_roi.y0-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Convert to Qt format and display
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QtGui.QImage(rgb_frame.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        
        # Scale to fit label
        pixmap = QtGui.QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(), 
            QtCore.Qt.KeepAspectRatio, 
            QtCore.Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    def update_state(self, state: DashboardState):
        """Update the dashboard with new state information."""
        self.state = state
        
        # Update provider card
        if hasattr(self, 'provider_card') and hasattr(self.provider_card, 'value_label'):
            provider_text = state.provider if state.provider else "None"
            self.provider_card.value_label.setText(f"🤖 {provider_text}")
        
        # Update performance card
        if hasattr(self, 'perf_card') and hasattr(self.perf_card, 'value_label'):
            if state.last_inference_ms is not None:
                self.perf_card.value_label.setText(f"{state.last_inference_ms} ms")
            else:
                self.perf_card.value_label.setText("-- ms")
        
        # Update countdown card
        if hasattr(self, 'countdown_card') and hasattr(self.countdown_card, 'value_label'):
            if state.next_audit_seconds is not None:
                self.countdown_card.value_label.setText(f"{state.next_audit_seconds} s")
            else:
                self.countdown_card.value_label.setText("-- s")
        
        # Update player cards
        for i, card in enumerate(self.player_labels):
            if i < len(state.player_totals):
                total = state.player_totals[i]
                
                # Update total value
                if hasattr(card, 'total_label'):
                    card.total_label.setText(f"${total:.2f}")
                
                # Update card styling based on total
                if total > 0:
                    card.setStyleSheet("""
                        QWidget {
                            background: rgba(52, 199, 89, 0.1);
                            border-radius: 10px;
                            border: 1px solid rgba(52, 199, 89, 0.3);
                        }
                    """)
                    if hasattr(card, 'total_label'):
                        card.total_label.setStyleSheet("""
                            QLabel {
                                font-size: 15px;
                                font-weight: 700;
                                color: #34C759;
                                background: transparent;
                                border: none;
                            }
                        """)
                else:
                    card.setStyleSheet("""
                        QWidget {
                            background: rgba(142, 142, 147, 0.05);
                            border-radius: 10px;
                            border: 1px solid rgba(142, 142, 147, 0.1);
                        }
                    """)
                    if hasattr(card, 'total_label'):
                        card.total_label.setStyleSheet("""
                            QLabel {
                                font-size: 15px;
                                font-weight: 700;
                                color: #8e8e93;
                                background: transparent;
                                border: none;
                            }
                        """)
            else:
                # Reset to default state
                if hasattr(card, 'total_label'):
                    card.total_label.setText("$0.00")
        
        # Update pot total
        if state.pot_total is not None:
            self.pot_label.setText(f"💰 POT: ${state.pot_total:.2f}")
        else:
            self.pot_label.setText("💰 POT: $0.00")
        
        # Update error display
        if state.last_error:
            self.error_label.setText(f"❌ {state.last_error}")
            self.error_label.show()
        else:
            self.error_label.hide()
    
    def set_rois(self, rois: List[NormalizedROI]):
        """Set the ROIs for overlay display."""
        self.rois = rois
        logger.info(f"Dashboard updated with {len(rois)} ROIs")
    
    def set_colors(self, colors: List[ColorSpec]):
        """Set the color specifications."""
        self.colors = colors
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current camera frame for analysis."""
        return self.current_frame
    
    def show_status(self, message: str, timeout: int = 3000):
        """Show a temporary status message."""
        self.status_label.setText(f"Status: {message}")
        if timeout > 0:
            QtCore.QTimer.singleShot(timeout, lambda: self.status_label.setText("Status: Ready"))
    
    def closeEvent(self, event):
        """Handle widget close event."""
        self.stop_camera()
        event.accept()
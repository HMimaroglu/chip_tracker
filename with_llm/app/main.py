"""
Main entry point for VLM-based poker chip tracker.
"""
import sys
import os
import yaml
import asyncio
import logging
from typing import List, Optional
from pathlib import Path
import time
import traceback

from PyQt5 import QtCore, QtGui, QtWidgets

from .schemas import AppConfig, ColorSpec, NormalizedROI, DashboardState, InferenceResult
from .camera import enumerate_cameras
from .roi_editor import ROIEditor
from .dashboard import DashboardWidget
from .providers import get_provider, VLMProvider

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SetupDialog(QtWidgets.QDialog):
    """Initial setup dialog for configuring the application."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Poker Chip Tracker - Setup")
        self.resize(800, 600)
        self.config = AppConfig()
        
        self.setup_ui()
        self.populate_defaults()
        
    def setup_ui(self):
        """Setup the dialog UI."""
        layout = QtWidgets.QVBoxLayout(self)
        
        # Create tabs
        tabs = QtWidgets.QTabWidget()
        layout.addWidget(tabs)
        
        # Camera tab
        camera_tab = QtWidgets.QWidget()
        self.setup_camera_tab(camera_tab)
        tabs.addTab(camera_tab, "Camera")
        
        # Players tab
        players_tab = QtWidgets.QWidget()
        self.setup_players_tab(players_tab)
        tabs.addTab(players_tab, "Players")
        
        # Colors tab
        colors_tab = QtWidgets.QWidget()
        self.setup_colors_tab(colors_tab)
        tabs.addTab(colors_tab, "Chip Colors")
        
        # VLM tab
        vlm_tab = QtWidgets.QWidget()
        self.setup_vlm_tab(vlm_tab)
        tabs.addTab(vlm_tab, "VLM Provider")
        
        # Buttons
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
    def setup_camera_tab(self, tab):
        """Setup camera selection tab."""
        layout = QtWidgets.QFormLayout(tab)
        
        # Camera selection
        camera_layout = QtWidgets.QHBoxLayout()
        self.camera_combo = QtWidgets.QComboBox()
        self.refresh_cameras()
        camera_layout.addWidget(self.camera_combo)
        
        test_button = QtWidgets.QPushButton("Test Camera")
        test_button.clicked.connect(self.test_camera)
        camera_layout.addWidget(test_button)
        
        refresh_button = QtWidgets.QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_cameras)
        camera_layout.addWidget(refresh_button)
        
        layout.addRow("Camera Device:", camera_layout)
        
    def setup_players_tab(self, tab):
        """Setup players configuration tab."""
        layout = QtWidgets.QFormLayout(tab)
        
        self.players_spin = QtWidgets.QSpinBox()
        self.players_spin.setRange(2, 9)
        self.players_spin.setValue(4)
        layout.addRow("Number of Players:", self.players_spin)
        
        info = QtWidgets.QLabel(
            "You will draw rectangular zones for each player after completing setup."
        )
        info.setWordWrap(True)
        layout.addRow("Note:", info)
        
    def setup_colors_tab(self, tab):
        """Setup chip colors configuration tab."""
        layout = QtWidgets.QVBoxLayout(tab)
        
        # Table for colors
        self.colors_table = QtWidgets.QTableWidget(0, 3)
        self.colors_table.setHorizontalHeaderLabels(["Color Name", "Value ($)", "Description"])
        self.colors_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.colors_table)
        
        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        add_button = QtWidgets.QPushButton("Add Row")
        add_button.clicked.connect(self.add_color_row)
        button_layout.addWidget(add_button)
        
        remove_button = QtWidgets.QPushButton("Remove Selected")
        remove_button.clicked.connect(self.remove_color_row)
        button_layout.addWidget(remove_button)
        
        presets_button = QtWidgets.QPushButton("Add Common Presets")
        presets_button.clicked.connect(self.add_color_presets)
        button_layout.addWidget(presets_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
    def setup_vlm_tab(self, tab):
        """Setup VLM provider configuration tab."""
        layout = QtWidgets.QFormLayout(tab)
        
        # Provider selection
        self.provider_combo = QtWidgets.QComboBox()
        self.provider_combo.addItems(["opencv", "ollama", "anthropic", "openai", "google"])
        self.provider_combo.currentTextChanged.connect(self.on_provider_changed)
        layout.addRow("Detection Provider:", self.provider_combo)
        
        # Model name
        self.model_edit = QtWidgets.QLineEdit("qwen2-vl")
        layout.addRow("Model Name:", self.model_edit)
        
        # Cadence
        self.cadence_spin = QtWidgets.QDoubleSpinBox()
        self.cadence_spin.setRange(1.0, 60.0)
        self.cadence_spin.setValue(8.0)
        self.cadence_spin.setSuffix(" seconds")
        layout.addRow("Audit Cadence:", self.cadence_spin)
        
        # Test connection
        test_layout = QtWidgets.QHBoxLayout()
        self.test_button = QtWidgets.QPushButton("Test Connection")
        self.test_button.clicked.connect(self.test_provider)
        test_layout.addWidget(self.test_button)
        
        self.connection_label = QtWidgets.QLabel("")
        test_layout.addWidget(self.connection_label)
        test_layout.addStretch()
        layout.addRow("", test_layout)
        
        # Instructions
        self.provider_info = QtWidgets.QLabel("")
        self.provider_info.setWordWrap(True)
        layout.addRow("Setup:", self.provider_info)
        
        self.on_provider_changed("ollama")
        
    def refresh_cameras(self):
        """Refresh available cameras."""
        self.camera_combo.clear()
        cameras = enumerate_cameras()
        
        if not cameras:
            self.camera_combo.addItem("No cameras found", -1)
        else:
            for cam_idx in cameras:
                self.camera_combo.addItem(f"Camera {cam_idx}", cam_idx)
                
    def test_camera(self):
        """Test the selected camera."""
        cam_idx = self.camera_combo.currentData()
        if cam_idx is None or cam_idx < 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No valid camera selected")
            return
            
        from .camera import CameraCapture
        with CameraCapture(cam_idx) as camera:
            if camera.is_open:
                QtWidgets.QMessageBox.information(self, "Success", f"Camera {cam_idx} is working!")
            else:
                QtWidgets.QMessageBox.warning(self, "Error", f"Failed to open camera {cam_idx}")
                
    def add_color_row(self):
        """Add a new color row."""
        row = self.colors_table.rowCount()
        self.colors_table.insertRow(row)
        
        # Default values
        self.colors_table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
        self.colors_table.setItem(row, 1, QtWidgets.QTableWidgetItem("1.00"))
        self.colors_table.setItem(row, 2, QtWidgets.QTableWidgetItem(""))
        
    def remove_color_row(self):
        """Remove selected color rows."""
        rows = sorted({index.row() for index in self.colors_table.selectedIndexes()}, reverse=True)
        for row in rows:
            self.colors_table.removeRow(row)
            
    def add_color_presets(self):
        """Add common poker chip color presets."""
        presets = [
            ("Red", "5.00", "Standard red poker chip"),
            ("Blue", "10.00", "Standard blue poker chip"), 
            ("Green", "25.00", "Standard green poker chip"),
            ("Black", "100.00", "Standard black poker chip"),
        ]
        
        for name, value, desc in presets:
            self.add_color_row()
            row = self.colors_table.rowCount() - 1
            self.colors_table.setItem(row, 0, QtWidgets.QTableWidgetItem(name))
            self.colors_table.setItem(row, 1, QtWidgets.QTableWidgetItem(value))
            self.colors_table.setItem(row, 2, QtWidgets.QTableWidgetItem(desc))
            
    def on_provider_changed(self, provider_name: str):
        """Update UI when provider changes."""
        model_defaults = {
            "opencv": "opencv-cv",
            "ollama": "llava",
            "anthropic": "claude-3-5-sonnet-20241022", 
            "openai": "gpt-4o",
            "google": "gemini-1.5-pro"
        }
        
        info_text = {
            "opencv": "Fast computer vision detection (~100ms). No internet or special setup required.",
            "ollama": "Install Ollama locally and pull a vision model: 'ollama pull llava'",
            "anthropic": "Set ANTHROPIC_API_KEY environment variable",
            "openai": "Set OPENAI_API_KEY environment variable",
            "google": "Set GOOGLE_API_KEY environment variable"
        }
        
        self.model_edit.setText(model_defaults.get(provider_name, ""))
        self.provider_info.setText(info_text.get(provider_name, ""))
        
    def test_provider(self):
        """Test the selected VLM provider."""
        provider_name = self.provider_combo.currentText()
        model_name = self.model_edit.text()
        
        try:
            provider = get_provider(provider_name, model_name=model_name)
            if provider.test_connection():
                self.connection_label.setText("✓ Connection successful")
                self.connection_label.setStyleSheet("color: green;")
            else:
                self.connection_label.setText("✗ Connection failed")
                self.connection_label.setStyleSheet("color: red;")
        except Exception as e:
            self.connection_label.setText(f"✗ Error: {str(e)}")
            self.connection_label.setStyleSheet("color: red;")
            
    def populate_defaults(self):
        """Populate dialog with default values."""
        # Add default colors if table is empty
        if self.colors_table.rowCount() == 0:
            self.add_color_presets()
            
    def get_config(self) -> AppConfig:
        """Extract configuration from dialog."""
        # Camera
        camera_index = self.camera_combo.currentData() or 0
        
        # Players
        players = self.players_spin.value()
        
        # Colors
        colors = []
        for row in range(self.colors_table.rowCount()):
            name_item = self.colors_table.item(row, 0)
            value_item = self.colors_table.item(row, 1)
            desc_item = self.colors_table.item(row, 2)
            
            if name_item and value_item and name_item.text().strip():
                try:
                    name = name_item.text().strip()
                    value = float(value_item.text())
                    desc = desc_item.text().strip() if desc_item else None
                    colors.append(ColorSpec(name=name, value=value, description=desc))
                except ValueError:
                    continue
                    
        # VLM settings
        provider = self.provider_combo.currentText()
        model_name = self.model_edit.text().strip()
        cadence = self.cadence_spin.value()
        
        return AppConfig(
            camera_index=camera_index,
            players=players,
            colors=colors,
            provider=provider,
            model_name=model_name,
            cadence_seconds=cadence
        )


class MainWindow(QtWidgets.QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VLM Poker Chip Tracker")
        self.resize(1200, 800)
        
        self.config: Optional[AppConfig] = None
        self.provider: Optional[VLMProvider] = None
        self.dashboard: Optional[DashboardWidget] = None
        
        # Inference loop
        self.inference_timer = QtCore.QTimer()
        self.inference_timer.timeout.connect(self.run_inference)
        self.inference_active = False
        
        self.setup_ui()
        self.load_config()
        
    def setup_ui(self):
        """Setup main window UI with Apple-inspired design."""
        # Set window properties
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #f8f9fa, stop: 1 #e9ecef);
            }
        """)
        
        # Central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        central.setStyleSheet("background: transparent;")
        
        layout = QtWidgets.QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Title and toolbar section
        header_container = QtWidgets.QWidget()
        header_container.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 12px;
                border: 1px solid #e0e0e0;
            }
        """)
        header_layout = QtWidgets.QVBoxLayout(header_container)
        header_layout.setContentsMargins(24, 20, 24, 20)
        header_layout.setSpacing(16)
        
        # Title
        title = QtWidgets.QLabel("🃏 VLM Poker Chip Tracker")
        title.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: 700;
                color: #1d1d1f;
                background: transparent;
                border: none;
                padding: 0;
            }
        """)
        title.setAlignment(QtCore.Qt.AlignCenter)
        header_layout.addWidget(title)
        
        # Control buttons in a grid
        controls_grid = QtWidgets.QGridLayout()
        controls_grid.setSpacing(12)
        
        # Configuration button (primary)
        self.config_button = self.create_apple_button(
            "Configure System", 
            primary=True, 
            icon_color="#007AFF"
        )
        self.config_button.clicked.connect(self.show_setup)
        controls_grid.addWidget(self.config_button, 0, 0)
        
        # Define zones button
        self.zones_button = self.create_apple_button(
            "Define Player Zones", 
            primary=False,
            icon_color="#FF9500"
        )
        self.zones_button.clicked.connect(self.define_zones)
        self.zones_button.setEnabled(False)
        controls_grid.addWidget(self.zones_button, 0, 1)
        
        # Start tracking button
        self.start_button = self.create_apple_button(
            "Start Tracking", 
            primary=False,
            icon_color="#34C759"
        )
        self.start_button.clicked.connect(self.start_tracking)
        self.start_button.setEnabled(False)
        controls_grid.addWidget(self.start_button, 1, 0)
        
        # Stop tracking button
        self.stop_button = self.create_apple_button(
            "Stop Tracking", 
            primary=False,
            icon_color="#FF3B30"
        )
        self.stop_button.clicked.connect(self.stop_tracking)
        self.stop_button.setEnabled(False)
        controls_grid.addWidget(self.stop_button, 1, 1)
        
        header_layout.addLayout(controls_grid)
        layout.addWidget(header_container)
        
        # Dashboard widget with modern styling
        self.dashboard = DashboardWidget()
        self.dashboard.setStyleSheet("""
            QWidget {
                background: white;
                border-radius: 12px;
                border: 1px solid #e0e0e0;
            }
        """)
        layout.addWidget(self.dashboard)
        
        # Modern status bar
        self.setup_status_bar()
    
    def create_apple_button(self, text: str, primary: bool = False, icon_color: str = "#007AFF") -> QtWidgets.QPushButton:
        """Create an Apple-style button."""
        button = QtWidgets.QPushButton(text)
        
        if primary:
            # Primary button (blue, filled)
            button.setStyleSheet(f"""
                QPushButton {{
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #007AFF, stop: 1 #0051D0);
                    color: white;
                    border: none;
                    border-radius: 12px;
                    font-size: 16px;
                    font-weight: 600;
                    padding: 16px 24px;
                    min-height: 20px;
                }}
                QPushButton:hover {{
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #0051D0, stop: 1 #003D99);
                }}
                QPushButton:pressed {{
                    background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                        stop: 0 #003D99, stop: 1 #002966);
                }}
                QPushButton:disabled {{
                    background: #f2f2f7;
                    color: #8e8e93;
                }}
            """)
        else:
            # Secondary button (outlined)
            button.setStyleSheet(f"""
                QPushButton {{
                    background: rgba(255, 255, 255, 0.8);
                    color: #1d1d1f;
                    border: 2px solid #e5e5e7;
                    border-radius: 12px;
                    font-size: 15px;
                    font-weight: 500;
                    padding: 14px 20px;
                    min-height: 20px;
                }}
                QPushButton:hover {{
                    background: rgba(0, 122, 255, 0.1);
                    border-color: {icon_color};
                    color: {icon_color};
                }}
                QPushButton:pressed {{
                    background: rgba(0, 122, 255, 0.2);
                    border-color: {icon_color};
                }}
                QPushButton:disabled {{
                    background: #f2f2f7;
                    color: #8e8e93;
                    border-color: #e5e5e7;
                }}
            """)
        
        # Add subtle shadow effect
        shadow = QtWidgets.QGraphicsDropShadowEffect()
        shadow.setBlurRadius(8)
        shadow.setColor(QtGui.QColor(0, 0, 0, 30))
        shadow.setOffset(0, 2)
        button.setGraphicsEffect(shadow)
        
        return button
    
    def setup_status_bar(self):
        """Setup modern status bar."""
        status_widget = QtWidgets.QWidget()
        status_widget.setStyleSheet("""
            QWidget {
                background: rgba(255, 255, 255, 0.95);
                border-top: 1px solid #e0e0e0;
                border-radius: 0;
            }
        """)
        
        status_layout = QtWidgets.QHBoxLayout(status_widget)
        status_layout.setContentsMargins(20, 8, 20, 8)
        
        # Status icon
        self.status_icon = QtWidgets.QLabel("●")
        self.status_icon.setStyleSheet("""
            QLabel {
                color: #8e8e93;
                font-size: 16px;
                background: transparent;
                border: none;
            }
        """)
        status_layout.addWidget(self.status_icon)
        
        # Status text
        self.status_text = QtWidgets.QLabel("Ready - Click 'Configure System' to begin setup")
        self.status_text.setStyleSheet("""
            QLabel {
                color: #1d1d1f;
                font-size: 14px;
                font-weight: 500;
                background: transparent;
                border: none;
                padding-left: 8px;
            }
        """)
        status_layout.addWidget(self.status_text)
        
        status_layout.addStretch()
        
        # Add to status bar
        self.statusBar().addPermanentWidget(status_widget)
        self.statusBar().setStyleSheet("QStatusBar { border: none; }")
        
    def update_status(self, message: str, status_type: str = "ready"):
        """Update status bar with colored indicator."""
        colors = {
            "ready": "#8e8e93",
            "success": "#34C759", 
            "warning": "#FF9500",
            "error": "#FF3B30",
            "active": "#007AFF"
        }
        
        color = colors.get(status_type, "#8e8e93")
        self.status_icon.setStyleSheet(f"""
            QLabel {{
                color: {color};
                font-size: 16px;
                background: transparent;
                border: none;
            }}
        """)
        self.status_text.setText(message)
        
    def create_menu_bar(self):
        """Create application menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        configure_action = QtWidgets.QAction('Configure...', self)
        configure_action.triggered.connect(self.show_setup)
        file_menu.addAction(configure_action)
        
        file_menu.addSeparator()
        
        exit_action = QtWidgets.QAction('Exit', self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        roi_action = QtWidgets.QAction('Define Player Zones...', self)
        roi_action.triggered.connect(self.define_zones)
        tools_menu.addAction(roi_action)
        
        start_action = QtWidgets.QAction('Start Tracking', self)
        start_action.triggered.connect(self.start_tracking)
        tools_menu.addAction(start_action)
        
        stop_action = QtWidgets.QAction('Stop Tracking', self)
        stop_action.triggered.connect(self.stop_tracking)
        tools_menu.addAction(stop_action)
        
    def load_config(self):
        """Load configuration from file."""
        config_path = Path("config.yaml")
        if config_path.exists():
            try:
                with open(config_path) as f:
                    data = yaml.safe_load(f)
                    self.config = AppConfig(**data)
                    self.apply_config()
                    logger.info("Configuration loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load config: {e}")
                self.update_button_states()
                self.update_status("Failed to load config - Click 'Configure System' to set up", "error")
        else:
            self.update_button_states()
            self.update_status("No configuration found - Click 'Configure System' to begin setup", "ready")
            
    def save_config(self):
        """Save configuration to file."""
        if not self.config:
            return
            
        try:
            config_data = self.config.dict()
            with open("config.yaml", "w") as f:
                yaml.dump(config_data, f, default_flow_style=False)
            logger.info("Configuration saved successfully")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            
    def apply_config(self):
        """Apply the current configuration."""
        if not self.config:
            return
            
        # Setup dashboard
        self.dashboard.set_players(self.config.players)
        self.dashboard.set_colors(self.config.colors)
        self.dashboard.set_rois(self.config.rois)
        
        # Start camera
        if not self.dashboard.start_camera(self.config.camera_index):
            QtWidgets.QMessageBox.warning(self, "Camera Error", 
                                        f"Failed to start camera {self.config.camera_index}")
            return
            
        # Initialize provider
        try:
            self.provider = get_provider(
                self.config.provider,
                model_name=self.config.model_name
            )
            logger.info(f"VLM provider initialized: {self.config.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize provider: {e}")
            QtWidgets.QMessageBox.warning(self, "Provider Error", 
                                        f"Failed to initialize {self.config.provider}: {str(e)}")
        
        # Update button states
        self.update_button_states()
        
        status_msg = f"Configured: {self.config.provider} | {len(self.config.colors)} colors | {self.config.players} players"
        if self.config.rois:
            status_msg += f" | {len(self.config.rois)} zones defined"
            self.update_status(status_msg, "success")
        else:
            self.update_status(status_msg, "warning")
    
    def update_button_states(self):
        """Update button enabled/disabled states based on configuration."""
        has_config = self.config is not None
        has_colors = has_config and len(self.config.colors) > 0
        has_zones = has_config and len(self.config.rois) > 0
        has_provider = self.provider is not None
        
        # Enable zone editor if we have basic config
        self.zones_button.setEnabled(has_config and has_colors)
        
        # Enable start tracking if everything is configured
        can_start_tracking = (has_config and has_colors and has_zones and 
                             has_provider and not self.inference_active)
        self.start_button.setEnabled(can_start_tracking)
        
        # Stop button enabled only when tracking
        self.stop_button.setEnabled(self.inference_active)
        
    def show_setup(self):
        """Show setup dialog."""
        dialog = SetupDialog(self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.config = dialog.get_config()
            self.save_config()
            self.apply_config()
            
            # Show success message
            QtWidgets.QMessageBox.information(
                self, "Configuration Saved", 
                f"✅ Configuration saved successfully!\n\n"
                f"• Camera: {self.config.camera_index}\n"
                f"• Players: {self.config.players}\n" 
                f"• Chip Colors: {len(self.config.colors)}\n"
                f"• VLM Provider: {self.config.provider} ({self.config.model_name})\n\n"
                f"Next: Define player zones to start tracking."
            )
            
    def define_zones(self):
        """Open ROI editor to define player zones."""
        if not self.config:
            QtWidgets.QMessageBox.warning(self, "Setup Required", "Please configure the application first.")
            return
            
        # Stop current tracking
        self.stop_tracking()
        
        # Open ROI editor
        editor = ROIEditor(self.config.camera_index, self.config.players, self)
        
        def on_zones_defined(zones):
            self.config.rois = zones
            self.dashboard.set_rois(zones)
            self.save_config()
            self.update_button_states()
            logger.info(f"Player zones defined: {len(zones)} zones")
            
            # Show success message
            QtWidgets.QMessageBox.information(
                self, "Zones Defined",
                f"✅ Player zones defined successfully!\n\n"
                f"• Zones created: {len(zones)}\n"
                f"• Ready to start tracking!\n\n"
                f"Click 'Start Tracking' to begin VLM analysis."
            )
            
        editor.zones_defined.connect(on_zones_defined)
        
        if editor.start_editing():
            editor.exec_()
            
    def start_tracking(self):
        """Start VLM inference tracking."""
        if not self.config or not self.provider:
            QtWidgets.QMessageBox.warning(self, "Setup Required", 
                                        "Please complete configuration first.")
            return
            
        if not self.config.rois:
            QtWidgets.QMessageBox.warning(self, "Zones Required", 
                                        "Please define player zones first.")
            return
            
        if len(self.config.colors) == 0:
            QtWidgets.QMessageBox.warning(self, "Colors Required", 
                                        "Please configure chip colors first.")
            return
            
        self.inference_active = True
        self.inference_timer.start(int(self.config.cadence_seconds * 1000))
        self.update_button_states()
        self.update_status("Tracking active - VLM analyzing chips...", "active")
        logger.info("VLM tracking started")
        
    def stop_tracking(self):
        """Stop VLM inference tracking."""
        self.inference_active = False
        self.inference_timer.stop()
        self.update_button_states()
        self.update_status("Tracking stopped", "ready")
        logger.info("VLM tracking stopped")
        
    def run_inference(self):
        """Run a single VLM inference cycle."""
        if not self.inference_active or not self.provider or not self.dashboard:
            return
            
        frame = self.dashboard.get_current_frame()
        if frame is None:
            logger.warning("No frame available for inference")
            return
            
        # Convert ROIs to pixel coordinates
        frame_height, frame_width = frame.shape[:2]
        pixel_rois = []
        for roi in self.config.rois:
            pixel_rois.append(roi.to_pixel(frame_width, frame_height))
            
        # Run inference asynchronously with proper error handling
        task = asyncio.create_task(self._async_inference(frame, pixel_rois))
        task.add_done_callback(self._handle_inference_task_completion)
        
    async def _async_inference(self, frame, pixel_rois):
        """Run VLM inference asynchronously."""
        start_time = time.time()
        
        try:
            result = await self.provider.infer(frame, pixel_rois, self.config.colors)
            inference_time = int((time.time() - start_time) * 1000)
            
            # Update dashboard state
            state = DashboardState(
                player_totals=[sum(player.counts[color.name] * color.value 
                                 for color in self.config.colors) 
                              for player in result.players],
                pot_total=result.pot,
                last_inference_ms=inference_time,
                provider=self.config.provider,
                next_audit_seconds=int(self.config.cadence_seconds),
                last_error=None
            )
            
            # Update UI from main thread using a safer approach
            QtCore.QTimer.singleShot(0, lambda: self.dashboard.update_state(state) if self.dashboard else None)
            
            logger.info(f"Inference completed in {inference_time}ms, pot: ${result.pot:.2f}")
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Inference failed: {error_msg}")
            
            # Update dashboard with error
            state = DashboardState(
                last_error=error_msg,
                provider=self.config.provider,
                next_audit_seconds=int(self.config.cadence_seconds * 2)  # Back off on error
            )
            
            QtCore.QTimer.singleShot(0, lambda: self.dashboard.update_state(state) if self.dashboard else None)
    
    def _handle_inference_task_completion(self, task):
        """Handle completion of async inference task and catch any unhandled exceptions."""
        try:
            if task.exception():
                error_msg = str(task.exception())
                logger.error(f"Async inference task failed: {error_msg}")
                
                # Update dashboard with error from main thread
                state = DashboardState(
                    last_error=f"Task failed: {error_msg}",
                    provider=self.config.provider if self.config else "unknown",
                    next_audit_seconds=int(self.config.cadence_seconds * 2) if self.config else 16
                )
                
                if self.dashboard:
                    # Use direct call since we're already in main thread
                    self.dashboard.update_state(state)
        except Exception as e:
            logger.error(f"Error handling task completion: {e}")
            
    def closeEvent(self, event):
        """Handle application close."""
        self.stop_tracking()
        if self.dashboard:
            self.dashboard.stop_camera()
        event.accept()


def main():
    """Application entry point."""
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("VLM Poker Chip Tracker")
    
    # Set event loop for asyncio
    import qasync
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    with loop:
        loop.run_forever()


if __name__ == "__main__":
    main()
"""
OpenCV camera capture utility.
"""
import cv2
import numpy as np
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def enumerate_cameras(max_index: int = 5) -> List[int]:
    """Find all available camera indices."""
    cameras = []
    
    for i in range(max_index):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    cameras.append(i)
            cap.release()
        except Exception as e:
            logger.debug(f"Failed to check camera {i}: {e}")
            pass
    
    return cameras


class CameraCapture:
    """OpenCV camera capture with graceful error handling."""
    
    def __init__(self, device_index: int = 0, width: int = 1280, height: int = 720):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None
        self._is_open = False
    
    def open(self) -> bool:
        """Open the camera device."""
        try:
            self.cap = cv2.VideoCapture(self.device_index)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.device_index}")
                return False
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Test read
            ret, _ = self.cap.read()
            if not ret:
                logger.error(f"Camera {self.device_index} opened but cannot read frames")
                self.cap.release()
                return False
            
            self._is_open = True
            logger.info(f"Camera {self.device_index} opened successfully at {self.width}x{self.height}")
            return True
            
        except Exception as e:
            logger.error(f"Exception opening camera {self.device_index}: {e}")
            if self.cap:
                self.cap.release()
            return False
    
    def read(self) -> Optional[np.ndarray]:
        """Read a frame from the camera."""
        if not self._is_open or not self.cap:
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                return None
            return frame
        except Exception as e:
            logger.error(f"Exception reading frame: {e}")
            return None
    
    def close(self):
        """Close the camera device."""
        if self.cap:
            try:
                self.cap.release()
                logger.info(f"Camera {self.device_index} closed")
            except Exception as e:
                logger.error(f"Exception closing camera: {e}")
            finally:
                self.cap = None
                self._is_open = False
    
    def reopen(self) -> bool:
        """Attempt to reopen the camera after a failure."""
        logger.info(f"Attempting to reopen camera {self.device_index}")
        self.close()
        return self.open()
    
    @property
    def is_open(self) -> bool:
        """Check if camera is open and ready."""
        return self._is_open and self.cap is not None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
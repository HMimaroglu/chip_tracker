"""
Computer Vision provider for fast poker chip detection.
Provides sub-second performance using OpenCV.
"""
import logging
import asyncio
from typing import List
import numpy as np

from .base import VLMProvider
from ..schemas import ROI, ColorSpec, InferenceResult, PlayerCounts
from ..cv_detector import FastChipDetector

logger = logging.getLogger(__name__)


class CVProvider(VLMProvider):
    """Computer vision-based chip detection provider."""
    
    def __init__(self, model_name: str = "opencv", **kwargs):
        super().__init__(model_name, **kwargs)
        self.detector = FastChipDetector()
        self.calibrated = False
    
    async def infer(self, image: np.ndarray, rois: List[ROI], colors: List[ColorSpec]) -> InferenceResult:
        """Analyze image using computer vision."""
        try:
            # Run detection in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            chip_counts = await loop.run_in_executor(
                None, self.detector.detect_chips, image, rois, colors
            )
            
            # Convert to InferenceResult format
            players = []
            total_pot = 0.0
            
            for player_id, counts in chip_counts.items():
                # Calculate player total
                player_total = 0.0
                for color in colors:
                    count = counts.get(color.name, 0)
                    player_total += count * color.value
                
                total_pot += player_total
                
                players.append(PlayerCounts(
                    id=player_id,
                    counts=counts,
                    confidence=0.85 if any(counts.values()) else 0.0  # High confidence for CV
                ))
            
            logger.debug(f"CV detection completed: pot=${total_pot:.2f}")
            return InferenceResult(players=players, pot=total_pot)
            
        except Exception as e:
            logger.error(f"CV inference failed: {e}")
            # Return empty result
            players = []
            for i in range(len(rois)):
                counts = {color.name: 0 for color in colors}
                players.append(PlayerCounts(
                    id=i + 1,
                    counts=counts,
                    confidence=0.0
                ))
            
            return InferenceResult(players=players, pot=0.0)
    
    def test_connection(self) -> bool:
        """Test if OpenCV is available."""
        try:
            import cv2
            logger.info(f"OpenCV version: {cv2.__version__}")
            return True
        except ImportError:
            logger.error("OpenCV not available")
            return False
    
    def calibrate(self, image: np.ndarray, color_samples: dict):
        """Calibrate color detection based on sample chips."""
        try:
            self.detector.calibrate_colors(image, color_samples)
            self.calibrated = True
            logger.info("Color calibration completed")
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
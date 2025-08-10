"""
Fast computer vision-based poker chip detector.
Provides sub-second performance for real-time tracking.
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Tuple
from dataclasses import dataclass

from .schemas import ROI, ColorSpec

logger = logging.getLogger(__name__)


@dataclass
class ChipDetection:
    """Represents a detected poker chip."""
    center: Tuple[int, int]
    radius: float
    color_name: str
    confidence: float


class FastChipDetector:
    """Fast OpenCV-based poker chip detector."""
    
    def __init__(self):
        self.color_ranges = {
            # More distinct HSV color ranges to prevent misclassification
            "Red": ([0, 120, 120], [8, 255, 255]),      # Tight red range
            "Blue": ([115, 120, 120], [125, 255, 255]), # Tight blue range
            "Green": ([55, 120, 120], [65, 255, 255]),  # Tight green range
            "Black": ([0, 0, 0], [180, 255, 40]),       # Very dark colors only
            "White": ([0, 0, 220], [180, 30, 255])      # Very light colors only
        }
        
        # Track previous detections for stability
        self.previous_detections = {}
        self.detection_history = {}
        self.stability_threshold = 3  # Require 3 consistent detections
        
        # Morphological kernels
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
    def detect_chips(self, image: np.ndarray, rois: List[ROI], colors: List[ColorSpec]) -> Dict[int, Dict[str, int]]:
        """
        Detect poker chips in image ROIs.
        
        Args:
            image: Input camera frame
            rois: List of player regions of interest
            colors: List of chip color specifications
            
        Returns:
            Dictionary mapping player ID to chip counts by color
        """
        results = {}
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for i, roi in enumerate(rois):
            player_id = i + 1
            
            # Extract ROI
            roi_hsv = hsv[roi.y0:roi.y1, roi.x0:roi.x1]
            roi_bgr = image[roi.y0:roi.y1, roi.x0:roi.x1]
            
            if roi_hsv.size == 0:
                results[player_id] = {color.name: 0 for color in colors}
                continue
                
            # Detect chips for each color with improved validation
            raw_counts = {}
            total_chips_detected = 0
            
            for color in colors:
                count = self._detect_color_chips(roi_hsv, roi_bgr, color.name)
                raw_counts[color.name] = count
                total_chips_detected += count
            
            # Apply stability filter
            stable_counts = self._apply_stability_filter(player_id, raw_counts)
            
            # Additional validation: if too many chips detected, prefer previous stable state
            if total_chips_detected > 20:  # Unrealistic number of chips
                if player_id in self.previous_detections:
                    stable_counts = self.previous_detections[player_id]
                    logger.warning(f"Player {player_id}: Too many chips detected ({total_chips_detected}), using previous state")
            
            results[player_id] = stable_counts
            self.previous_detections[player_id] = stable_counts.copy()
            
        return results
    
    def _detect_color_chips(self, hsv_roi: np.ndarray, bgr_roi: np.ndarray, color_name: str) -> int:
        """Detect chips of a specific color in an ROI."""
        if color_name not in self.color_ranges:
            return 0
            
        lower, upper = self.color_ranges[color_name]
        lower = np.array(lower)
        upper = np.array(upper)
        
        # Create color mask
        mask = cv2.inRange(hsv_roi, lower, upper)
        
        # Clean up mask with morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
        
        # Apply Gaussian blur to reduce noise
        mask = cv2.GaussianBlur(mask, (3, 3), 0)
        
        # Find circles using HoughCircles with better parameters
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=25,  # Increased minimum distance between circles
            param1=40,   # Edge detection threshold
            param2=20,   # Center detection threshold (less sensitive to avoid false positives)
            minRadius=10, # Minimum chip radius in pixels
            maxRadius=40 # Maximum chip radius in pixels
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Filter circles based on color consistency and confidence
            valid_circles = []
            for (x, y, r) in circles:
                confidence = self._validate_chip_circle(bgr_roi, mask, x, y, r, color_name)
                if confidence > 0.3:  # Lower confidence threshold to detect more
                    valid_circles.append((x, y, r, confidence))
            
            # Sort by confidence and take the most confident detections
            valid_circles.sort(key=lambda c: c[3], reverse=True)
            
            logger.debug(f"Detected {len(valid_circles)} {color_name} chips")
            return len(valid_circles)
        
        return 0
    
    def _validate_chip_circle(self, bgr_roi: np.ndarray, mask: np.ndarray, x: int, y: int, r: int, color_name: str) -> float:
        """Validate that a detected circle is actually a chip and return confidence score."""
        h, w = mask.shape
        
        # Check bounds
        if x - r < 0 or x + r >= w or y - r < 0 or y + r >= h:
            return 0.0
        
        # Check circularity by examining the mask within the circle
        circle_mask = np.zeros_like(mask)
        cv2.circle(circle_mask, (x, y), r, 255, -1)
        
        # Calculate the percentage of the circle that matches the color
        overlap = cv2.bitwise_and(mask, circle_mask)
        circle_area = np.pi * r * r
        overlap_area = np.sum(overlap > 0)
        
        # Require at least 25% color match for validation (more lenient)
        color_ratio = overlap_area / circle_area
        
        # Also check for reasonable roundness
        contours, _ = cv2.findContours(overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            # Find the largest contour (should be the chip)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Calculate confidence score based on color match and circularity
                color_score = min(1.0, color_ratio / 0.25)  # Lower threshold
                shape_score = min(1.0, circularity / 0.3)   # Lower threshold
                
                # Combined confidence (both factors must be good)
                confidence = (color_score * shape_score) * 0.8  # Max 0.8 to leave room for improvement
                
                # Bonus for very round and very well-colored chips
                if color_ratio > 0.6 and circularity > 0.7:
                    confidence = min(1.0, confidence * 1.2)
                
                logger.debug(f"{color_name} circle at ({x},{y}) r={r}: color_ratio={color_ratio:.2f}, circularity={circularity:.2f}, confidence={confidence:.2f}")
                return confidence
        
        return 0.0
    
    def _apply_stability_filter(self, player_id: int, current_counts: Dict[str, int]) -> Dict[str, int]:
        """Apply stability filter to reduce flickering between detections."""
        if player_id not in self.detection_history:
            self.detection_history[player_id] = []
        
        # Add current detection to history
        history = self.detection_history[player_id]
        history.append(current_counts.copy())
        
        # Keep only recent history
        if len(history) > 5:
            history.pop(0)
        
        # If we don't have enough history, return current detection
        if len(history) < self.stability_threshold:
            return current_counts
        
        # Find the most stable (consistent) detection in recent history
        stable_counts = {}
        for color_name in current_counts.keys():
            # Get recent values for this color
            recent_values = [h[color_name] for h in history[-self.stability_threshold:]]
            
            # Check if values are consistent (all same or mostly same)
            most_common_value = max(set(recent_values), key=recent_values.count)
            consistent_count = recent_values.count(most_common_value)
            
            if consistent_count >= self.stability_threshold - 1:  # Allow 1 outlier
                stable_counts[color_name] = most_common_value
            else:
                # If not stable, use previous detection if available
                if player_id in self.previous_detections:
                    stable_counts[color_name] = self.previous_detections[player_id].get(color_name, 0)
                else:
                    stable_counts[color_name] = most_common_value
        
        return stable_counts
    
    def calibrate_colors(self, image: np.ndarray, color_samples: Dict[str, List[Tuple[int, int]]]) -> None:
        """
        Calibrate color ranges based on actual chip samples in the image.
        
        Args:
            image: Camera frame containing chips
            color_samples: Dict mapping color names to list of (x, y) sample points
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        for color_name, sample_points in color_samples.items():
            if len(sample_points) == 0:
                continue
                
            # Sample HSV values at the given points
            hsv_samples = []
            for x, y in sample_points:
                if 0 <= x < hsv.shape[1] and 0 <= y < hsv.shape[0]:
                    hsv_samples.append(hsv[y, x])
            
            if len(hsv_samples) > 0:
                # Calculate mean and standard deviation
                hsv_array = np.array(hsv_samples)
                mean_hsv = np.mean(hsv_array, axis=0)
                std_hsv = np.std(hsv_array, axis=0)
                
                # Set ranges as mean ± 2*std, with reasonable bounds
                lower = np.maximum([0, 50, 50], mean_hsv - 2 * std_hsv)
                upper = np.minimum([179, 255, 255], mean_hsv + 2 * std_hsv)
                
                self.color_ranges[color_name] = (lower.astype(int).tolist(), upper.astype(int).tolist())
                logger.info(f"Calibrated {color_name}: lower={lower}, upper={upper}")
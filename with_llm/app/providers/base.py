"""
Abstract base interface for VLM providers.
"""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any
from ..schemas import ROI, ColorSpec, InferenceResult


class VLMProvider(ABC):
    """Abstract base class for Vision-Language Model providers."""
    
    def __init__(self, model_name: str = None, **kwargs):
        self.model_name = model_name
        self.config = kwargs
    
    @abstractmethod
    async def infer(
        self, 
        image: np.ndarray, 
        rois: List[ROI], 
        colors: List[ColorSpec]
    ) -> InferenceResult:
        """
        Analyze image for poker chips in specified regions.
        
        Args:
            image: Full camera frame as numpy array (BGR format)
            rois: List of rectangular regions (pixel coordinates) 
            colors: List of chip color specifications
            
        Returns:
            InferenceResult with per-player chip counts and confidence
        """
        pass
    
    def _build_system_prompt(self) -> str:
        """Build the system message for the VLM."""
        return (
            "You are a poker chip counting expert. Count visible chips by color in each region. "
            "Return ONLY valid JSON with exact counts. Be fast and decisive - don't overthink. "
            "Round chips are poker chips. Stacked chips count each visible chip."
        )
    
    def _build_user_prompt(self, rois: List[ROI], colors: List[ColorSpec]) -> str:
        """Build the user prompt with ROI and color specifications."""
        # Colors section
        colors_text = "Colors and values (USD):\n"
        for color in colors:
            desc = color.description or "chip style"
            colors_text += f"- {color.name}: ${color.value} ({desc})\n"
        
        # ROIs section
        rois_text = "Player zones (pixel coords in the attached image):\n"
        for i, roi in enumerate(rois, start=1):
            rois_text += f"- Player {i}: [{roi.x0},{roi.y0},{roi.x1},{roi.y1}]\n"
        
        # JSON schema
        color_names = [f'"{color.name}": int' for color in colors]
        color_schema = ", ".join(color_names)
        
        schema_text = f"""
Output JSON only, matching this schema exactly:
{{
  "players": [
    {{"id": 1, "counts": {{{color_schema}}}, "confidence": 0.0-1.0}},
    ...
  ],
  "pot": float
}}

Rules:
- Do not include any keys not listed.
- Omit colors that don't appear by setting their count to 0.
- Use integer counts only.
- pot = sum over players of (sum over colors of count[color] * value[color]).
"""
        
        return f"""FAST COUNT NEEDED: Count chips by color in each player zone.

{colors_text}

{rois_text}

Quick rules:
- Round objects = poker chips
- Stack height = number of chips  
- Estimate if partially hidden
- Return JSON immediately

{schema_text}"""
    
    def _encode_image_base64(self, image: np.ndarray, format: str = 'JPEG') -> str:
        """Encode numpy image as base64 string."""
        import cv2
        import base64
        
        # Encode image
        _, buffer = cv2.imencode(f'.{format.lower()}', image)
        encoded = base64.b64encode(buffer).decode('utf-8')
        return encoded
    
    def _extract_json_from_response(self, response_text: str) -> Dict[str, Any]:
        """Extract JSON from potentially markdown-wrapped response."""
        import json
        import re
        
        # Try to find JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON directly
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")
    
    @abstractmethod
    def test_connection(self) -> bool:
        """Test if the provider is available and configured correctly."""
        pass
"""
Pydantic schemas for poker chip counter application.
"""
from typing import Dict, List, Optional
from pydantic import BaseModel, constr, confloat, conint


class ColorSpec(BaseModel):
    """Specification for a chip color and its value."""
    name: constr(strip_whitespace=True, min_length=1)
    value: confloat(ge=0)
    description: Optional[str] = None


class ROI(BaseModel):
    """Region of Interest coordinates (pixel coordinates)."""
    x0: conint(ge=0)
    y0: conint(ge=0) 
    x1: conint(gt=0)
    y1: conint(gt=0)
    
    def to_normalized(self, width: int, height: int) -> 'NormalizedROI':
        """Convert to normalized coordinates (0.0-1.0)."""
        return NormalizedROI(
            x0=self.x0 / width,
            y0=self.y0 / height,
            x1=self.x1 / width,
            y1=self.y1 / height
        )


class NormalizedROI(BaseModel):
    """Region of Interest with normalized coordinates (0.0-1.0)."""
    x0: confloat(ge=0.0, le=1.0)
    y0: confloat(ge=0.0, le=1.0)
    x1: confloat(ge=0.0, le=1.0)
    y1: confloat(ge=0.0, le=1.0)
    
    def to_pixel(self, width: int, height: int) -> ROI:
        """Convert to pixel coordinates."""
        return ROI(
            x0=int(self.x0 * width),
            y0=int(self.y0 * height),
            x1=int(self.x1 * width),
            y1=int(self.y1 * height)
        )


class PlayerCounts(BaseModel):
    """Chip counts for a single player."""
    id: conint(ge=1)
    counts: Dict[str, conint(ge=0)]  # color name -> integer count
    confidence: confloat(ge=0.0, le=1.0)


class InferenceResult(BaseModel):
    """Result from VLM inference."""
    players: List[PlayerCounts]
    pot: confloat(ge=0.0)


class AppConfig(BaseModel):
    """Application configuration."""
    camera_index: int = 0
    players: conint(ge=2, le=9) = 4
    colors: List[ColorSpec] = []
    rois: List[NormalizedROI] = []
    provider: str = "ollama"
    model_name: str = "qwen2-vl"
    cadence_seconds: confloat(ge=1.0) = 8.0
    
    
class DashboardState(BaseModel):
    """Current state for dashboard display."""
    player_totals: List[confloat(ge=0.0)] = []
    pot_total: confloat(ge=0.0) = 0.0
    last_inference_ms: Optional[int] = None
    provider: str = "ollama"
    next_audit_seconds: Optional[int] = None
    last_error: Optional[str] = None
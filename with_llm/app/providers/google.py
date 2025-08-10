"""
Google Gemini Vision provider implementation.
"""
import os
import logging
from typing import List
import numpy as np

from .base import VLMProvider
from ..schemas import ROI, ColorSpec, InferenceResult, PlayerCounts

logger = logging.getLogger(__name__)


class GoogleProvider(VLMProvider):
    """Google Gemini Vision provider."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
        except ImportError:
            raise ImportError("google-generativeai package is required: pip install google-generativeai")
    
    async def infer(self, image: np.ndarray, rois: List[ROI], colors: List[ColorSpec]) -> InferenceResult:
        """Analyze image using Google Gemini Vision."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(rois, colors)
        
        # Combine system and user prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Convert image to PIL Image
        import cv2
        from PIL import Image
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        try:
            import asyncio
            
            def _sync_generate():
                response = self.model.generate_content(
                    [full_prompt, pil_image],
                    generation_config={
                        'temperature': 0.1,
                        'max_output_tokens': 1024,
                    }
                )
                return response
            
            # Run in thread pool since Gemini client is sync
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_generate)
            
            response_text = response.text
            logger.debug(f"Google response: {response_text}")
            
            # Extract JSON from response
            try:
                json_data = self._extract_json_from_response(response_text)
            except ValueError as e:
                # Retry with more forceful instruction
                logger.warning(f"Failed to parse JSON, retrying: {e}")
                return await self._retry_with_strict_prompt(pil_image, rois, colors)
            
            # Validate and convert to InferenceResult
            return self._parse_inference_result(json_data, len(rois), colors)
            
        except Exception as e:
            raise RuntimeError(f"Google inference failed: {e}")
    
    async def _retry_with_strict_prompt(self, pil_image, rois: List[ROI], colors: List[ColorSpec]) -> InferenceResult:
        """Retry inference with a more forceful JSON-only instruction."""
        strict_prompt = f"Return compact JSON only. No explanation, no markdown, just raw JSON.\n\n{self._build_user_prompt(rois, colors)}\n\nOUTPUT JSON NOW:"
        
        try:
            import asyncio
            
            def _sync_generate():
                response = self.model.generate_content(
                    [strict_prompt, pil_image],
                    generation_config={
                        'temperature': 0.0,
                        'max_output_tokens': 512,
                    }
                )
                return response
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, _sync_generate)
            
            response_text = response.text
            json_data = self._extract_json_from_response(response_text)
            return self._parse_inference_result(json_data, len(rois), colors)
            
        except Exception as e:
            # If retry also fails, return empty result
            logger.error(f"Retry also failed: {e}")
            return self._create_empty_result(len(rois), colors)
    
    def _parse_inference_result(self, json_data: dict, num_players: int, colors: List[ColorSpec]) -> InferenceResult:
        """Parse and validate JSON response into InferenceResult."""
        try:
            players_data = json_data.get("players", [])
            
            # Ensure we have data for all players
            players = []
            for i in range(num_players):
                player_id = i + 1
                player_data = None
                
                # Find data for this player
                for p in players_data:
                    if p.get("id") == player_id:
                        player_data = p
                        break
                
                if player_data is None:
                    # Missing player data, create empty
                    counts = {color.name: 0 for color in colors}
                    confidence = 0.0
                else:
                    counts = player_data.get("counts", {})
                    confidence = player_data.get("confidence", 0.5)
                    
                    # Ensure all colors are present
                    for color in colors:
                        if color.name not in counts:
                            counts[color.name] = 0
                
                players.append(PlayerCounts(
                    id=player_id,
                    counts=counts,
                    confidence=max(0.0, min(1.0, confidence))
                ))
            
            # Calculate pot
            pot = json_data.get("pot", 0.0)
            if pot <= 0:
                # Calculate pot from player counts
                pot = 0.0
                for player in players:
                    for color in colors:
                        count = player.counts.get(color.name, 0)
                        pot += count * color.value
            
            return InferenceResult(players=players, pot=pot)
            
        except Exception as e:
            logger.error(f"Failed to parse inference result: {e}")
            return self._create_empty_result(num_players, colors)
    
    def _create_empty_result(self, num_players: int, colors: List[ColorSpec]) -> InferenceResult:
        """Create an empty result when parsing fails."""
        players = []
        for i in range(num_players):
            counts = {color.name: 0 for color in colors}
            players.append(PlayerCounts(
                id=i + 1,
                counts=counts,
                confidence=0.0
            ))
        
        return InferenceResult(players=players, pot=0.0)
    
    def test_connection(self) -> bool:
        """Test if Google API is available."""
        try:
            response = self.model.generate_content(
                "Hello",
                generation_config={
                    'max_output_tokens': 10,
                }
            )
            return bool(response.text)
        except Exception as e:
            logger.error(f"Google connection test failed: {e}")
            return False
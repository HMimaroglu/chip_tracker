"""
Anthropic Claude VLM provider implementation.
"""
import os
import logging
from typing import List
import numpy as np

from .base import VLMProvider
from ..schemas import ROI, ColorSpec, InferenceResult, PlayerCounts

logger = logging.getLogger(__name__)


class AnthropicProvider(VLMProvider):
    """Anthropic Claude vision provider."""
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("anthropic package is required: pip install anthropic")
    
    async def infer(self, image: np.ndarray, rois: List[ROI], colors: List[ColorSpec]) -> InferenceResult:
        """Analyze image using Anthropic Claude Vision."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(rois, colors)
        
        # Encode image as base64
        image_b64 = self._encode_image_base64(image)
        
        try:
            message = await self._call_anthropic(system_prompt, user_prompt, image_b64)
            response_text = message.content[0].text
            
            logger.debug(f"Anthropic response: {response_text}")
            
            # Extract JSON from response
            try:
                json_data = self._extract_json_from_response(response_text)
            except ValueError as e:
                # Retry with more forceful instruction
                logger.warning(f"Failed to parse JSON, retrying: {e}")
                return await self._retry_with_strict_prompt(image, rois, colors, image_b64)
            
            # Validate and convert to InferenceResult
            return self._parse_inference_result(json_data, len(rois), colors)
            
        except Exception as e:
            raise RuntimeError(f"Anthropic inference failed: {e}")
    
    async def _call_anthropic(self, system_prompt: str, user_prompt: str, image_b64: str):
        """Make API call to Anthropic."""
        import asyncio
        
        def _sync_call():
            return self.client.messages.create(
                model=self.model_name,
                max_tokens=1024,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": image_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": user_prompt
                            }
                        ]
                    }
                ]
            )
        
        # Run in thread pool since anthropic client is sync
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_call)
    
    async def _retry_with_strict_prompt(self, image: np.ndarray, rois: List[ROI], colors: List[ColorSpec], image_b64: str) -> InferenceResult:
        """Retry inference with a more forceful JSON-only instruction."""
        strict_system = "Return compact JSON only. No explanation, no markdown, just raw JSON."
        user_prompt = self._build_user_prompt(rois, colors) + "\n\nOUTPUT JSON NOW:"
        
        try:
            message = await self._call_anthropic(strict_system, user_prompt, image_b64)
            response_text = message.content[0].text
            
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
        """Test if Anthropic API is available."""
        try:
            # Simple test call
            test_message = self.client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ]
            )
            return bool(test_message.content)
        except Exception as e:
            logger.error(f"Anthropic connection test failed: {e}")
            return False
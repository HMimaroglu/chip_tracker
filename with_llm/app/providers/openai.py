"""
OpenAI GPT-4 Vision provider implementation.
"""
import os
import logging
from typing import List
import numpy as np

from .base import VLMProvider
from ..schemas import ROI, ColorSpec, InferenceResult, PlayerCounts

logger = logging.getLogger(__name__)


class OpenAIProvider(VLMProvider):
    """OpenAI GPT-4 Vision provider."""
    
    def __init__(self, model_name: str = "gpt-4o", **kwargs):
        super().__init__(model_name, **kwargs)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        try:
            import openai
            self.client = openai.AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("openai package is required: pip install openai")
    
    async def infer(self, image: np.ndarray, rois: List[ROI], colors: List[ColorSpec]) -> InferenceResult:
        """Analyze image using OpenAI GPT-4 Vision."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(rois, colors)
        
        # Encode image as base64
        image_b64 = self._encode_image_base64(image)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user", 
                        "content": [
                            {
                                "type": "text",
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024,
                temperature=0.1
            )
            
            response_text = response.choices[0].message.content
            logger.debug(f"OpenAI response: {response_text}")
            
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
            raise RuntimeError(f"OpenAI inference failed: {e}")
    
    async def _retry_with_strict_prompt(self, image: np.ndarray, rois: List[ROI], colors: List[ColorSpec], image_b64: str) -> InferenceResult:
        """Retry inference with a more forceful JSON-only instruction."""
        strict_system = "Return compact JSON only. No explanation, no markdown, just raw JSON."
        user_prompt = self._build_user_prompt(rois, colors) + "\n\nOUTPUT JSON NOW:"
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": strict_system
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": user_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_b64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=512,
                temperature=0.0
            )
            
            response_text = response.choices[0].message.content
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
        """Test if OpenAI API is available."""
        try:
            import asyncio
            
            async def _test():
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": "Hello"
                        }
                    ],
                    max_tokens=10
                )
                return bool(response.choices)
            
            # Run async test
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(_test())
            loop.close()
            return result
            
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            return False
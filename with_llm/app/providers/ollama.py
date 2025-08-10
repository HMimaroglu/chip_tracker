"""
Ollama VLM provider implementation.
"""
import json
import logging
import httpx
import asyncio
from typing import List
import numpy as np

from .base import VLMProvider
from ..schemas import ROI, ColorSpec, InferenceResult, PlayerCounts

logger = logging.getLogger(__name__)


class OllamaProvider(VLMProvider):
    """Ollama local VLM provider."""
    
    def __init__(self, model_name: str = "qwen2-vl", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model_name, **kwargs)
        self.base_url = base_url.rstrip('/')
        self.timeout = kwargs.get('timeout', 60.0)  # Increased timeout for vision models
    
    async def infer(self, image: np.ndarray, rois: List[ROI], colors: List[ColorSpec]) -> InferenceResult:
        """Analyze image using Ollama API."""
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(rois, colors)
        
        # Combine system and user prompts for Ollama
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        # Encode image as base64
        image_b64 = self._encode_image_base64(image)
        
        # Prepare request
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistency
                "top_p": 0.9,
                "num_ctx": 2048,     # Smaller context for faster processing
                "num_predict": 256,  # Limit response length
                "repeat_penalty": 1.0,
                "top_k": 10          # Reduce sampling for speed
            }
        }
        
        try:
            logger.info(f"Starting Ollama inference with model {self.model_name}, timeout={self.timeout}s")
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                response_text = result.get("response", "")
                logger.info(f"Ollama response received, length: {len(response_text)}")
                
                if not response_text:
                    raise ValueError("Empty response from Ollama")
                
                logger.debug(f"Ollama response: {response_text}")
                
                # Extract JSON from response
                try:
                    json_data = self._extract_json_from_response(response_text)
                except ValueError as e:
                    # Retry with more forceful instruction
                    logger.warning(f"Failed to parse JSON, retrying: {e}")
                    return await self._retry_with_strict_prompt(image, rois, colors, image_b64)
                
                # Validate and convert to InferenceResult
                return self._parse_inference_result(json_data, len(rois), colors)
                
        except httpx.TimeoutException as e:
            logger.error(f"Ollama request timed out after {self.timeout}s: {e}")
            raise RuntimeError(f"Ollama request timed out after {self.timeout}s")
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama API HTTP error: {e.response.status_code} - {e.response.text}")
            raise RuntimeError(f"Ollama API error: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            logger.error(f"Ollama inference exception: {type(e).__name__}: {e}")
            raise RuntimeError(f"Ollama inference failed: {e}")
    
    async def _retry_with_strict_prompt(self, image: np.ndarray, rois: List[ROI], colors: List[ColorSpec], image_b64: str) -> InferenceResult:
        """Retry inference with a more forceful JSON-only instruction."""
        strict_prompt = f"""Return compact JSON only. No explanation, no markdown, just raw JSON.

{self._build_user_prompt(rois, colors)}

OUTPUT JSON NOW:"""
        
        payload = {
            "model": self.model_name,
            "prompt": strict_prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.0,  # Even lower temperature
                "top_p": 0.8,
                "num_ctx": 1024,     # Even smaller context for retry
                "num_predict": 128,  # Shorter response
                "top_k": 5           # Very focused sampling
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                response_text = result.get("response", "")
                
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
        """Test if Ollama is available."""
        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    tags = response.json()
                    models = [model["name"] for model in tags.get("models", [])]
                    logger.info(f"Ollama available with models: {models}")
                    
                    if self.model_name in models or any(self.model_name in model for model in models):
                        return True
                    else:
                        logger.warning(f"Model {self.model_name} not found in Ollama. Available: {models}")
                        return False
                return False
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False
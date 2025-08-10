"""
FastAPI server for web dashboard and API endpoints.
"""
import os
import yaml
import asyncio
import logging
from pathlib import Path
from typing import Optional
import cv2
import base64

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.schemas import AppConfig, DashboardState
from app.camera import CameraCapture

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VLM Poker Chip Tracker API",
    description="Web API for poker chip tracking results",
    version="1.0.0"
)

# Global state
current_state = DashboardState()
current_frame: Optional[bytes] = None
camera: Optional[CameraCapture] = None
config: Optional[AppConfig] = None


async def startup():
    """Initialize the server."""
    global camera, config
    
    # Load config
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
            config = AppConfig(**data)
        
        # Initialize camera
        camera = CameraCapture(config.camera_index)
        if camera.open():
            logger.info(f"Camera {config.camera_index} opened for web server")
        else:
            logger.error(f"Failed to open camera {config.camera_index}")
    
    logger.info("FastAPI server started")


async def shutdown():
    """Cleanup on server shutdown."""
    global camera
    if camera:
        camera.close()
    logger.info("FastAPI server stopped")


@app.on_event("startup")
async def startup_event():
    await startup()


@app.on_event("shutdown") 
async def shutdown_event():
    await shutdown()


@app.get("/")
async def dashboard():
    """Serve the HTML dashboard."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Poker Chip Tracker Dashboard</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { 
                font-family: Arial, sans-serif; 
                margin: 20px; 
                background-color: #f5f5f5; 
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .header { 
                text-align: center; 
                margin-bottom: 20px; 
                color: #333;
            }
            .stats { 
                display: flex; 
                gap: 20px; 
                margin-bottom: 20px; 
                flex-wrap: wrap;
            }
            .stat-box { 
                flex: 1; 
                min-width: 200px;
                padding: 15px; 
                background: #f8f9fa; 
                border-radius: 6px;
                border-left: 4px solid #007bff;
            }
            .stat-label { 
                font-weight: bold; 
                color: #666; 
                font-size: 0.9em;
                margin-bottom: 5px;
            }
            .stat-value { 
                font-size: 1.5em; 
                color: #333; 
            }
            .players { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                gap: 15px; 
                margin-bottom: 20px; 
            }
            .player { 
                padding: 15px; 
                background: #e8f5e8; 
                border-radius: 6px;
                border: 1px solid #4CAF50;
            }
            .player-name { 
                font-weight: bold; 
                margin-bottom: 8px; 
                color: #2E7D32;
            }
            .player-total { 
                font-size: 1.3em; 
                color: #1B5E20; 
            }
            .pot { 
                text-align: center; 
                font-size: 2em; 
                font-weight: bold; 
                color: #d32f2f; 
                padding: 20px;
                background: #ffebee;
                border: 3px solid #d32f2f;
                border-radius: 8px;
                margin: 20px 0;
            }
            .camera-feed {
                text-align: center;
                margin: 20px 0;
            }
            .camera-feed img {
                max-width: 100%;
                height: auto;
                border: 2px solid #ddd;
                border-radius: 6px;
            }
            .error { 
                background: #ffebee; 
                color: #c62828; 
                padding: 10px; 
                border-radius: 4px;
                margin: 10px 0;
            }
            .info {
                background: #e3f2fd;
                color: #1565c0;
                padding: 10px;
                border-radius: 4px;
                margin: 10px 0;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="header">🃏 Poker Chip Tracker Dashboard</h1>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-label">Provider</div>
                    <div class="stat-value" id="provider">--</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Last Inference</div>
                    <div class="stat-value" id="inference-time">--</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Next Audit</div>
                    <div class="stat-value" id="next-audit">--</div>
                </div>
            </div>
            
            <div class="camera-feed">
                <img id="camera-frame" src="/frame.jpg" alt="Camera feed will appear here">
            </div>
            
            <div class="pot" id="pot">POT: $0.00</div>
            
            <div class="players" id="players-container">
                <!-- Players will be populated here -->
            </div>
            
            <div id="error-container"></div>
        </div>

        <script>
            async function updateDashboard() {
                try {
                    const response = await fetch('/state');
                    const data = await response.json();
                    
                    // Update stats
                    document.getElementById('provider').textContent = data.provider || '--';
                    document.getElementById('inference-time').textContent = 
                        data.last_inference_ms ? data.last_inference_ms + 'ms' : '--';
                    document.getElementById('next-audit').textContent = 
                        data.next_audit_seconds ? data.next_audit_seconds + 's' : '--';
                    
                    // Update pot
                    document.getElementById('pot').textContent = `POT: $${data.pot_total.toFixed(2)}`;
                    
                    // Update players
                    const playersContainer = document.getElementById('players-container');
                    playersContainer.innerHTML = '';
                    
                    data.player_totals.forEach((total, index) => {
                        const playerDiv = document.createElement('div');
                        playerDiv.className = 'player';
                        playerDiv.innerHTML = `
                            <div class="player-name">Player ${index + 1}</div>
                            <div class="player-total">$${total.toFixed(2)}</div>
                        `;
                        playersContainer.appendChild(playerDiv);
                    });
                    
                    // Update error display
                    const errorContainer = document.getElementById('error-container');
                    if (data.last_error) {
                        errorContainer.innerHTML = `<div class="error">Error: ${data.last_error}</div>`;
                    } else {
                        errorContainer.innerHTML = '';
                    }
                    
                } catch (error) {
                    console.error('Failed to update dashboard:', error);
                    document.getElementById('error-container').innerHTML = 
                        '<div class="error">Failed to connect to server</div>';
                }
                
                // Update camera frame
                const cameraFrame = document.getElementById('camera-frame');
                cameraFrame.src = '/frame.jpg?' + Date.now(); // Cache busting
            }
            
            // Update every 2 seconds
            setInterval(updateDashboard, 2000);
            
            // Initial update
            updateDashboard();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/state")
async def get_state():
    """Get current dashboard state."""
    return current_state.dict()


@app.get("/frame.jpg")
async def get_frame():
    """Get current camera frame as JPEG."""
    global camera, current_frame
    
    if not camera or not camera.is_open:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    frame = camera.read()
    if frame is None:
        raise HTTPException(status_code=503, detail="Failed to capture frame")
    
    # Encode as JPEG
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    return Response(
        content=buffer.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
    )


@app.post("/state")
async def update_state(state: DashboardState):
    """Update the dashboard state (called by main app)."""
    global current_state
    current_state = state
    return {"status": "updated"}


def start_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run(
        "server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    import sys
    
    host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
    
    start_server(host, port)
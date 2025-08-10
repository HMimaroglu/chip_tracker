# VLM Poker Chip Tracker

A production-quality poker chip counting system that uses Vision-Language Models (VLMs) to count chips by color without requiring geometric calibration. The system supports multiple VLM providers and provides both a desktop GUI and web dashboard.

## Features

- **No Calibration Required**: Uses VLM intelligence instead of geometric calibration
- **Multiple VLM Providers**: Supports Ollama (local), Anthropic Claude, OpenAI GPT-4, and Google Gemini
- **Interactive Setup**: GUI wizard for camera selection, player zones, chip colors, and VLM configuration
- **Real-time Tracking**: Live chip counting at configurable intervals (5-60 seconds)
- **Dual Interface**: PyQt desktop app + FastAPI web dashboard
- **Robust Error Handling**: Automatic retries, connection testing, and graceful degradation

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd with_llm

# Install dependencies
pip install -r requirements.txt

# Create logs directory
mkdir -p logs
```

### 2. Setup VLM Provider

Choose one of the following providers:

#### Option A: Ollama (Local, Recommended)
```bash
# Install Ollama: https://ollama.ai
# Pull a vision model
ollama pull qwen2-vl
# or
ollama pull internvl2
```

#### Option B: Anthropic Claude
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
pip install anthropic
```

#### Option C: OpenAI GPT-4
```bash
export OPENAI_API_KEY="your-api-key-here"
pip install openai
```

#### Option D: Google Gemini
```bash
export GOOGLE_API_KEY="your-api-key-here"
pip install google-generativeai
```

### 3. Run the Application

```bash
# Start the desktop application
python -m app.main

# OR start the web server only
python server.py
```

### 4. Configuration Workflow

1. **Launch Setup**: On first run, the setup dialog appears automatically
2. **Select Camera**: Choose your camera device and test it
3. **Configure Players**: Set number of players (2-9)
4. **Define Chip Colors**: Add chip colors and values (presets available)
5. **Choose VLM Provider**: Select provider and test connection
6. **Define Player Zones**: Draw rectangles around each player's chip area
7. **Start Tracking**: Begin real-time chip counting

## Usage Guide

### Desktop Application

The main window shows:
- **Live camera feed** with player zone overlays
- **Per-player totals** updated in real-time
- **Pot total** calculated from all players
- **Provider status** and inference timing
- **Error messages** when issues occur

Menu options:
- `File → Configure`: Reopen setup dialog
- `Tools → Define Player Zones`: Redraw player areas
- `Tools → Start/Stop Tracking`: Control VLM inference

### Web Dashboard

Access the web interface at `http://localhost:8000` to view:
- Current camera feed
- Live player totals and pot
- Provider performance metrics
- Error status

API endpoints:
- `GET /`: HTML dashboard
- `GET /state`: JSON state data
- `GET /frame.jpg`: Current camera frame
- `POST /state`: Update state (internal use)

### ROI Editor

When defining player zones:
1. **Click and drag** to draw rectangles around chip areas
2. **Minimum size**: 50x50 pixels required
3. **Clear controls**: Remove last zone or clear all
4. **Progress tracking**: Shows current/required zone count
5. **Validation**: Must draw exact number of player zones

## Configuration

Settings are stored in `config.yaml`:

```yaml
camera_index: 0           # Camera device ID
players: 4                # Number of players (2-9)
cadence_seconds: 8.0      # Inference interval
provider: "ollama"        # VLM provider
model_name: "qwen2-vl"    # Model name

colors:                   # Chip specifications
  - name: "Red"
    value: 5.0
    description: "Standard red poker chip"
  # ... more colors

rois: []                  # Player zones (auto-populated)
```

## VLM Provider Details

### Ollama (Local)
- **Pros**: Private, fast, no API costs
- **Cons**: Requires local GPU/RAM
- **Models**: `qwen2-vl`, `internvl2`, `llava`
- **Setup**: Install Ollama, pull model

### Anthropic Claude
- **Pros**: Excellent vision capabilities
- **Cons**: API costs, internet required
- **Model**: `claude-3-5-sonnet-20241022`
- **Setup**: Set `ANTHROPIC_API_KEY`

### OpenAI GPT-4
- **Pros**: Good vision, reliable
- **Cons**: API costs, internet required  
- **Model**: `gpt-4o`
- **Setup**: Set `OPENAI_API_KEY`

### Google Gemini
- **Pros**: Fast, good value
- **Cons**: API costs, internet required
- **Model**: `gemini-1.5-pro`
- **Setup**: Set `GOOGLE_API_KEY`

## Architecture

```
with_llm/
├── app/
│   ├── main.py           # PyQt GUI entry point
│   ├── roi_editor.py     # Zone drawing interface
│   ├── dashboard.py      # Live display widget
│   ├── camera.py         # OpenCV utilities
│   ├── schemas.py        # Pydantic models
│   └── providers/        # VLM implementations
│       ├── base.py       # Abstract interface
│       ├── ollama.py     # Local Ollama
│       ├── anthropic.py  # Claude integration
│       ├── openai.py     # GPT-4 integration
│       └── google.py     # Gemini integration
├── server.py             # FastAPI web server
├── config.yaml           # Configuration file
├── requirements.txt      # Dependencies
└── logs/                 # Application logs
```

## Prompt Engineering

The system uses carefully crafted prompts:

1. **System Message**: Establishes role as vision assistant
2. **User Prompt**: Provides color specifications and ROI coordinates
3. **JSON Schema**: Enforces structured output format
4. **Retry Logic**: Falls back to stricter prompts if parsing fails

Example prompt structure:
```
Task: Count poker chips by color in defined regions.

Colors and values:
- Red: $5.00 (Standard red poker chip)
- Blue: $10.00 (Standard blue poker chip)

Player zones (pixel coordinates):
- Player 1: [100,200,300,400]
- Player 2: [400,200,600,400]

Output JSON only: {"players": [{"id": 1, "counts": {"Red": 5, "Blue": 2}, "confidence": 0.9}], "pot": 35.0}
```

## Performance Optimization

- **Inference Caching**: Configurable cadence prevents excessive API calls
- **Error Backoff**: 2x cadence delay after failures
- **Concurrent Processing**: Async inference doesn't block UI
- **Frame Optimization**: JPEG compression for API efficiency
- **Connection Pooling**: Reused HTTP clients

## Troubleshooting

### Common Issues

**Camera not detected:**
- Check camera permissions
- Try different camera indices
- Ensure no other apps are using camera

**VLM connection failed:**
- Verify API keys are set correctly
- Check internet connection for cloud providers
- For Ollama: ensure service is running (`ollama serve`)

**Poor counting accuracy:**
- Ensure good lighting conditions
- Draw tight ROIs around chip stacks
- Use contrasting chip colors
- Consider lower inference cadence for stability

**High API costs:**
- Use local Ollama provider
- Increase cadence_seconds (10-30s)
- Optimize ROI sizes to focus on chip areas

### Logs

Check `logs/app.log` for detailed error information:
```bash
tail -f logs/app.log
```

## Development

### Adding New Providers

1. Inherit from `VLMProvider` base class
2. Implement `infer()` and `test_connection()` methods
3. Add to provider factory in `providers/__init__.py`
4. Update setup dialog and documentation

### Testing

```bash
# Test individual components
python -m pytest

# Test provider connections
python -c "from app.providers import get_provider; print(get_provider('ollama').test_connection())"
```

## License

This project is provided as-is for educational and development purposes. VLM provider APIs may have their own terms of service and usage costs.

## Support

For issues or questions:
1. Check the logs in `logs/app.log`
2. Verify provider setup and API keys
3. Test camera access independently
4. Review configuration in `config.yaml`
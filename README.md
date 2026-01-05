# GPU Worker

GPU worker for [Gitart](https://gitart.me) SDXL inference.

## Requirements

- NVIDIA GPU with 8GB+ VRAM
- Linux with NVIDIA drivers installed
- Python 3.9+
- [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) server running on port 7860

## Installation

```bash
pip install gpu_worker
```

Or install from source:

```bash
git clone https://github.com/takuya/gpu_worker.git
cd gpu_worker
pip install -e .
```

## Usage

```bash
gpu-worker --api-key YOUR_API_KEY
```

Get your API key from [gitart.me](https://gitart.me).

## Model Setup

Place SDXL model files (`.safetensors`) in `/models/sd/`:

```bash
mkdir -p /models/sd
# Download or copy your SDXL models here
```

The worker automatically detects available models and reports them to the server.

## How It Works

1. Worker polls the Gitart server for jobs
2. When a job is received, it sends the request to the local SD server
3. Generated images are returned to the Gitart server
4. Settings (schedule, enabled models) can be configured via the web UI

## Configuration

Settings are synced from the server and stored in `~/.gitart-worker/settings.json`.

### Schedule

Configure when the worker should be active via the web UI at gitart.me.

## License

MIT License

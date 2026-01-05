#!/usr/bin/env python3
"""
Gitart Worker - Handles SDXL inference jobs
Polls the job server and processes requests using local GPU

Install: pip install gitart-worker
Usage: gitart-worker --api-key YOUR_API_KEY
"""
import json
import time
import requests
import subprocess
import argparse
import hashlib
import glob
from datetime import datetime
from pathlib import Path

# Hardcoded configuration - not user configurable
SERVER_URL = 'https://gitart.me'
SD_SERVER = 'http://localhost:7860'
POLL_INTERVAL = 5

CONFIG_DIR = Path.home() / ".gitart-worker"
SETTINGS_FILE = CONFIG_DIR / "settings.json"

# Files to verify integrity
INTEGRITY_PATHS = [
    '/usr/local/bin/sd',
    '/models/sd/*.safetensors',
]

# Models directory
MODELS_DIR = '/models/sd'


def detect_models():
    """Detect available models from the models directory"""
    models = []
    for filepath in glob.glob(f"{MODELS_DIR}/*.safetensors"):
        name = Path(filepath).stem  # filename without extension
        models.append({"name": name, "service": "sdxl"})
    return models  # Empty if no models found


def compute_file_hash(filepath):
    """Compute SHA256 hash of a file"""
    try:
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()
    except (IOError, OSError):
        return None


def compute_integrity_hashes():
    """Compute hashes of all critical files"""
    hashes = {}

    for pattern in INTEGRITY_PATHS:
        if '*' in pattern:
            for filepath in glob.glob(pattern):
                h = compute_file_hash(filepath)
                if h:
                    hashes[filepath] = h
        else:
            h = compute_file_hash(pattern)
            if h:
                hashes[pattern] = h

    # Hash this script itself
    h = compute_file_hash(Path(__file__).resolve())
    if h:
        hashes['worker_script'] = h

    return hashes


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_settings():
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            pass
    return None


def save_settings(settings):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)


def is_within_schedule(schedule):
    if not schedule or not schedule.get('enabled'):
        return True
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(schedule.get('timezone', 'UTC'))
    except (ImportError, KeyError):
        return True

    now = datetime.now(tz)
    current_day = now.strftime('%a').lower()[:3]
    current_time = now.strftime('%H:%M')

    for rule in schedule.get('rules', []):
        if current_day in rule.get('days', []):
            if rule.get('start_time', '00:00') <= current_time <= rule.get('end_time', '23:59'):
                return True
    return False


def detect_gpus():
    gpus = []
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return gpus

        for line in result.stdout.strip().split('\n'):
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 3:
                try:
                    idx = int(parts[0])
                    name = parts[1]
                    mem = parts[2]

                    if any(k in name.lower() for k in ['llvmpipe', 'software', 'virtual']):
                        continue

                    vram = int(mem.replace('MiB', '').strip())
                    if vram < 2000:
                        continue

                    gpus.append({
                        'index': idx,
                        'name': name,
                        'memory': f"{vram // 1024}GB" if vram >= 1024 else f"{vram}MB"
                    })
                except (ValueError, IndexError):
                    continue
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        pass
    return gpus


def run_sdxl(params):
    try:
        resp = requests.post(f"{SD_SERVER}/txt2img", json={
            "prompt": params.get('prompt', ''),
            "negative_prompt": params.get('negative_prompt', ''),
            "width": params.get('width', 1024),
            "height": params.get('height', 1024),
            "steps": params.get('steps', 20),
            "cfg_scale": params.get('cfg_scale', 7.0),
            "seed": params.get('seed', -1),
            "sample_method": "euler_a"
        }, timeout=300)

        if resp.status_code != 200:
            return {'error': f'Server error {resp.status_code}'}

        r = resp.json()
        if 'images' in r and r['images']:
            return {'image': r['images'][0], 'seed': r.get('seed', -1)}
        elif 'image' in r:
            return {'image': r['image'], 'seed': r.get('seed', -1)}
        return {'error': 'No image'}
    except Exception as e:
        return {'error': str(e)}


def process_job(job):
    service = job.get('service', 'sdxl')
    params = job.get('params', {})

    log(f"Processing {service} job {job['id'][:8]}...")
    start = time.time()

    if service != 'sdxl':
        return {'error': f'Unsupported service: {service}'}

    result = run_sdxl(params)
    elapsed = time.time() - start

    if 'error' in result:
        log(f"Job failed: {result['error']}")
        return {'error': result['error']}

    log(f"Job completed in {elapsed:.1f}s")
    result['inference_time'] = elapsed
    return {'result': result}


def worker_loop(api_key):
    gpus = detect_gpus()
    if gpus:
        log(f"Detected {len(gpus)} GPU(s)")
        for g in gpus:
            log(f"  GPU {g['index']}: {g['name']} ({g['memory']})")
        gpu_info = {'name': gpus[0]['name'], 'memory': gpus[0]['memory']}
    else:
        log("No GPU detected")
        gpu_info = {}

    # Detect available models
    models = detect_models()
    log(f"Available models: {[m['name'] for m in models]}")

    # Compute integrity hashes
    log("Computing integrity hashes...")
    integrity_hashes = compute_integrity_hashes()
    log(f"Verified {len(integrity_hashes)} files")

    settings = load_settings()
    settings_version = settings.get('settings_version', 0) if settings else 0

    headers = {
        'Authorization': f'Bearer {api_key}',
        'X-Worker-Models': json.dumps(models),
        'X-Worker-GPU': json.dumps(gpu_info),
        'X-Worker-GPUs': json.dumps(gpus),
        'X-Worker-Integrity': json.dumps(integrity_hashes)
    }

    log(f"Starting worker, connecting to {SERVER_URL}")
    log("Press Ctrl+C to stop")

    errors = 0
    while True:
        try:
            if settings and not is_within_schedule(settings.get('schedule', {})):
                time.sleep(POLL_INTERVAL)
                continue

            resp = requests.get(f"{SERVER_URL}/worker/jobs/poll", headers=headers, timeout=30)

            if resp.status_code == 401:
                log("ERROR: Invalid API key")
                return
            if resp.status_code != 200:
                errors += 1
                time.sleep(min(POLL_INTERVAL * errors, 60))
                continue

            data = resp.json()

            # Sync settings
            if data.get('config_sync'):
                new_ver = data['config_sync'].get('settings_version', 0)
                if new_ver > settings_version:
                    settings = data['config_sync'].get('settings')
                    if settings:
                        save_settings(settings)
                        settings_version = new_ver
                        log(f"Settings synced (v{new_ver})")

            job = data.get('job')
            if job:
                errors = 0
                result = process_job(job)
                requests.post(
                    f"{SERVER_URL}/worker/jobs/{job['id']}/complete",
                    headers=headers, json=result, timeout=30
                )
            else:
                errors = 0
                time.sleep(POLL_INTERVAL)

        except KeyboardInterrupt:
            log("Shutting down...")
            break
        except Exception as e:
            log(f"Error: {e}")
            errors += 1
            time.sleep(min(POLL_INTERVAL * errors, 60))


def main():
    parser = argparse.ArgumentParser(description='Gitart Worker - SDXL Inference')
    parser.add_argument('--api-key', required=True, help='Your API key')
    parser.add_argument('--version', action='version', version='0.1.0')
    args = parser.parse_args()
    worker_loop(args.api_key)


if __name__ == '__main__':
    main()

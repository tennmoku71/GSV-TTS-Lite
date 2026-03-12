# GSV-TTS-Lite (Build-focused Fork)

This repository is a build-focused fork of the original project.

## What This Project Provides

- macOS runtime compatibility adjustments for GSV-TTS-Lite (including Apple Silicon / MPS environments)
- standalone build artifacts for easier distribution and execution
- a packaging-oriented workflow so users can run the server from extracted deliverables

## Upstream Project

- Original repository: [chinokikiss/GSV-TTS-Lite](https://github.com/chinokikiss/GSV-TTS-Lite)

Please refer to the upstream repository for:

- full feature documentation
- usage/API details
- model/architecture explanations
- original release notes and roadmap

## Why This Fork Exists

This fork is maintained to organize and distribute build-related outputs only, with a focus on practical packaging and runtime compatibility (including macOS).

## Scope of This Repository

- build and packaging adjustments
- runtime compatibility fixes needed for distribution
- minimal docs for running packaged binaries (included below)

## Quick Start (Standalone Package)

This package is prepared so you can run `gsv_tts_http_server.bin` immediately after extraction.

### 1. Start the Server

```bash
cd gsv_tts_http_server_rebuilt_pkg
chmod +x gsv_tts_http_server.bin
./gsv_tts_http_server.bin --host 127.0.0.1 --port 9882
```

Expected startup logs:

- `server started: http://127.0.0.1:9882`
- `endpoints: GET /health, POST /synthesize`

### 2. Health Check

```bash
curl -s http://127.0.0.1:9882/health
```

### 3. Synthesis Test

```bash
curl -s -X POST "http://127.0.0.1:9882/synthesize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello. This is a rebuilt package test."
  }' \
| python3 -c 'import sys, json, base64; d=json.load(sys.stdin); open("test.wav","wb").write(base64.b64decode(d["wav_base64"]))'
```

If `test.wav` is generated, the server is working.

### 4. If Port Is Already In Use

If you see `Address already in use`, change the port:

```bash
./gsv_tts_http_server.bin --host 127.0.0.1 --port 9888
```

### 5. Included Files

- `gsv_tts_http_server.bin`: Server executable
- `models/`: Model files required for inference
- `examples/`: Reference audio files and config

## Respect to Upstream

Core implementation and project design originate from the upstream repository.
If you want to understand or use GSV-TTS-Lite in depth, please start from the upstream documentation first.

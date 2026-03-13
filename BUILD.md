# Build Notes (Nuitka)

This document records a known-good Nuitka build command and common packaging notes for `gsv_tts_http_server.py`.

## Successful Build Command

Run from the repository root:

```bash
.venv/bin/python -m nuitka \
  --follow-imports \
  --nofollow-import-to=torch,torchaudio,numpy,transformers \
  --output-dir="./build" \
  "./gsv_tts_http_server.py"
```

## Build Outputs

- Binary: `build/gsv_tts_http_server.bin`
- Intermediate build directory: `build/gsv_tts_http_server.build/`

## Example: Create a Distributable Package

```bash
pkg_dir="./build/gsv_tts_http_server_rebuilt_pkg"
rm -rf "$pkg_dir"
mkdir -p "$pkg_dir/examples"
cp "./build/gsv_tts_http_server.bin" "$pkg_dir/"
cp -R "./models" "$pkg_dir/"
cp "./examples/models.config" "$pkg_dir/examples/"
cp "./examples/laffey.mp3" "$pkg_dir/examples/"
cp "./examples/AnAn.ogg" "$pkg_dir/examples/"
```

Create ZIP:

```bash
cd ./build
zip -r "gsv_tts_http_server_rebuilt_pkg.zip" "gsv_tts_http_server_rebuilt_pkg" -x "*/.DS_Store"
```

## Notes

- **First build can take time**  
  C compilation can take several minutes.

- **If you see `Address already in use`**  
  Another process is already using the port. Change `--port` or stop the existing process first.

- **Model files are required**  
  The binary alone is not enough; package `models/` and `examples/` together.
  A practical distribution layout is:
  ```text
  gsv_tts_http_server_rebuilt_pkg/
  ├─ gsv_tts_http_server.bin
  ├─ models/
  └─ examples/
  ```
  Put these three items in the same top-level folder, then create a ZIP from that folder.

- **Do not force Flash Attention on non-CUDA backends**  
  On CPU/MPS, the current code falls back to standard attention when Flash Attention is requested.

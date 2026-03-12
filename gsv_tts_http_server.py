#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np

from gsv_tts import TTS


def resolve_default_models_dir() -> str:
    # Priority:
    # 1) Explicit environment override.
    # 2) Bundled models directory next to executable/script.
    # 3) Existing cache location fallback.
    env_dir = (os.getenv("GSV_MODELS_DIR") or "").strip()
    if env_dir:
        return env_dir

    exe_base = Path(sys.argv[0]).resolve().parent
    bundled = exe_base / "models"
    if bundled.exists():
        return str(bundled)

    return str(Path.home() / ".cache" / "gsv")


def resolve_runtime_asset_path(path_str: str) -> str:
    path_str = (path_str or "").strip()
    if not path_str:
        return path_str
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    exe_base = Path(sys.argv[0]).resolve().parent
    exe_rel = exe_base / p
    if exe_rel.exists():
        return str(exe_rel)
    return str((Path.cwd() / p).resolve())


def load_models_config(config_path: str) -> dict[str, Any]:
    path = resolve_runtime_asset_path(config_path)
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as e:
        print(f"[WARN] failed to load models config: {p} ({e})")
    return {}


def float_audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    audio = np.asarray(audio, dtype=np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)

    bio = BytesIO()
    # Minimal WAV encoder
    import wave

    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return bio.getvalue()


def generate_fallback_audio(sample_rate: int = 32000, duration_s: float = 0.35) -> np.ndarray:
    # Short low-volume tone + tiny silence so callers always get playable audio.
    n = max(1, int(sample_rate * duration_s))
    t = np.arange(n, dtype=np.float32) / float(sample_rate)
    tone = 0.08 * np.sin(2.0 * np.pi * 660.0 * t).astype(np.float32)
    tail = np.zeros((int(sample_rate * 0.08),), dtype=np.float32)
    return np.concatenate([tone, tail]).astype(np.float32)


def build_gpt_cache(cache_len: int, batch_size: int) -> list[tuple[int, int]]:
    if batch_size < 1:
        batch_size = 1
    caches = [(1, cache_len)]
    for b in range(4, batch_size - 1, 4):
        caches.append((b, cache_len))
    if (batch_size, cache_len) not in caches:
        caches.append((batch_size, cache_len))
    return caches


class TTSHTTPHandler(BaseHTTPRequestHandler):
    tts: TTS | None = None
    tts_lock = threading.Lock()
    default_spk_audio_path: str = "examples/laffey.mp3"
    default_prompt_audio_path: str = "examples/AnAn.ogg"
    default_prompt_audio_text: str = "ちが……ちがう。レイア、貴様は間違っている。"
    short_text_boost_enabled: bool = False
    short_text_min_chars: int = 6
    short_text_prefix: str = "...."
    short_text_strategy: str = "dot"
    short_text_carrier_prefix: str = "それでは、"
    short_text_carrier_suffix: str = "。"
    punctuation_chars: str = "。．！？!?…"

    def _sanitize_text_for_retry(self, text: str) -> str:
        # Retry-time normalization for texts that can trigger phone/word2ph mismatch.
        normalized = text.translate(
            str.maketrans(
                {
                    "「": "",
                    "」": "",
                    "『": "",
                    "』": "",
                    "【": "",
                    "】": "",
                    "（": "",
                    "）": "",
                    "(": "",
                    ")": "",
                    "[": "",
                    "]": "",
                    "{": "",
                    "}": "",
                    "<": "",
                    ">": "",
                    "\"": "",
                    "'": "",
                    "／": "、",
                    "/": "、",
                    "\\": "、",
                    "|": "、",
                }
            )
        )
        normalized = re.sub(r"、{2,}", "、", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _build_retry_text_candidates(self, text: str) -> list[str]:
        candidates: list[str] = []

        def _add(candidate: str) -> None:
            c = candidate.strip()
            if c and c not in candidates and c != text:
                candidates.append(c)

        c1 = self._sanitize_text_for_retry(text)
        _add(c1)

        # More aggressive normalization for mixed symbols and acronym-heavy strings.
        c2 = c1
        c2 = re.sub(r"\bATC\b", "エーティーシー", c2, flags=re.IGNORECASE)
        c2 = c2.replace("・", "、")
        c2 = re.sub(r"[A-Za-z0-9]+", "", c2)
        c2 = re.sub(r"、{2,}", "、", c2)
        c2 = re.sub(r"\s+", " ", c2).strip(" 、")
        _add(c2)

        # Last-resort filter: keep Japanese chars and common punctuation only.
        c3 = re.sub(r"[^ぁ-んァ-ヶ一-龠々ー、。！？!? ]", "", c2)
        c3 = re.sub(r"、{2,}", "、", c3)
        c3 = re.sub(r"\s+", " ", c3).strip(" 、")
        _add(c3)
        return candidates

    def _apply_short_text_boost(
        self,
        text: str,
        enabled: bool | None = None,
        strategy: str | None = None,
    ) -> tuple[str, str]:
        if strategy is None:
            strategy = self.short_text_strategy
        strategy = str(strategy).lower()
        if strategy not in {"none", "dot", "carrier"}:
            strategy = self.short_text_strategy

        if enabled is None:
            enabled = self.short_text_boost_enabled
            if strategy != "none":
                enabled = True
        if not enabled:
            return text, "none"
        if len(text) >= self.short_text_min_chars:
            return text, "none"

        if strategy == "carrier":
            suffix = self.short_text_carrier_suffix
            if text and text[-1] in self.punctuation_chars:
                suffix = ""
            return f"{self.short_text_carrier_prefix}{text}{suffix}", "carrier"

        pad_len = self.short_text_min_chars - len(text)
        prefix = self.short_text_prefix
        if len(prefix) < pad_len:
            prefix = "." * pad_len
        return prefix[:pad_len] + text, "dot"

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, socket.timeout):
            # Client closed connection before response write finished.
            return

    def log_message(self, format: str, *args: Any) -> None:
        # Keep standard access log noise down.
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"ok": True, "service": "gsv_tts_http_server"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/synthesize":
            self._send_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
            return

        if self.tts is None:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"ok": False, "error": "TTS is not initialized"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length) if content_length > 0 else b"{}"
            body = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "invalid json body"})
            return

        original_text = str(body.get("text", "")).strip()
        if not original_text:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "text is required"})
            return
        text, short_text_strategy_used = self._apply_short_text_boost(
            original_text,
            body.get("short_text_boost"),
            body.get("short_text_strategy"),
        )

        spk_audio_path = body.get("spk_audio_path") or self.default_spk_audio_path
        prompt_audio_path = body.get("prompt_audio_path") or self.default_prompt_audio_path
        prompt_audio_text = body.get("prompt_audio_text") or self.default_prompt_audio_text

        kwargs = {
            "top_k": int(body.get("top_k", 15)),
            "top_p": float(body.get("top_p", 1.0)),
            "temperature": float(body.get("temperature", 1.0)),
            "repetition_penalty": float(body.get("repetition_penalty", 1.35)),
            "min_output_tokens": max(0, int(body.get("min_output_tokens", 0))),
            "noise_scale": float(body.get("noise_scale", 0.5)),
            "speed": float(body.get("speed", 1.0)),
        }

        retry_sanitized = False
        inference_text = text
        try:
            t0 = time.perf_counter()
            # TTS inference is not thread-safe on MPS/CUDA paths.
            # Serialize requests to avoid dtype/cache races and process abort.
            with self.tts_lock:
                try:
                    clip = self.tts.infer(
                        spk_audio_path=spk_audio_path,
                        prompt_audio_path=prompt_audio_path,
                        prompt_audio_text=prompt_audio_text,
                        text=inference_text,
                        **kwargs,
                    )
                except Exception as e:
                    # Some mixed-symbol texts can break phones/word2ph alignment.
                    if "length mismatch" not in str(e):
                        raise
                    last_error = e
                    for retry_text in self._build_retry_text_candidates(inference_text):
                        try:
                            clip = self.tts.infer(
                                spk_audio_path=spk_audio_path,
                                prompt_audio_path=prompt_audio_path,
                                prompt_audio_text=prompt_audio_text,
                                text=retry_text,
                                **kwargs,
                            )
                            inference_text = retry_text
                            retry_sanitized = True
                            break
                        except Exception as retry_error:
                            last_error = retry_error
                    else:
                        raise last_error
            infer_time_s = time.perf_counter() - t0
            wav_bytes = float_audio_to_wav_bytes(clip.audio_data, clip.samplerate)
            wav_b64 = base64.b64encode(wav_bytes).decode("ascii")
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "original_text": original_text,
                    "synthesis_text": inference_text,
                    "short_text_boost_applied": original_text != text,
                    "short_text_strategy_used": short_text_strategy_used,
                    "retry_sanitized": retry_sanitized,
                    "sample_rate": clip.samplerate,
                    "audio_len_s": clip.audio_len_s,
                    "infer_time_s": infer_time_s,
                    "mime_type": "audio/wav",
                    "wav_base64": wav_b64,
                },
            )
        except Exception as e:
            # Return fallback audio instead of HTTP 500 so clients can continue.
            fallback_sr = 32000
            fallback_audio = generate_fallback_audio(sample_rate=fallback_sr)
            fallback_wav = float_audio_to_wav_bytes(fallback_audio, fallback_sr)
            fallback_b64 = base64.b64encode(fallback_wav).decode("ascii")
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "degraded": True,
                    "error": str(e),
                    "original_text": original_text,
                    "synthesis_text": text,
                    "short_text_boost_applied": original_text != text,
                    "short_text_strategy_used": short_text_strategy_used,
                    "retry_sanitized": False,
                    "sample_rate": fallback_sr,
                    "audio_len_s": len(fallback_audio) / fallback_sr,
                    "infer_time_s": 0.0,
                    "mime_type": "audio/wav",
                    "wav_base64": fallback_b64,
                },
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="HTTP server for gsv_tts")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9882)
    parser.add_argument("--models-dir", default=resolve_default_models_dir())
    parser.add_argument("--gpt-cache-len", type=int, default=1024)
    parser.add_argument("--gpt-batch-size", type=int, default=8)
    parser.add_argument("--use-bert", action="store_true")
    parser.add_argument("--use-flash-attn", action="store_true")
    parser.add_argument("--models-config", default="examples/models.config")
    parser.add_argument("--default-spk-audio-path", default=None)
    parser.add_argument("--default-prompt-audio-path", default=None)
    parser.add_argument("--default-prompt-audio-text", default=None)
    parser.add_argument("--short-text-boost", action="store_true")
    parser.add_argument("--short-text-min-chars", type=int, default=6)
    parser.add_argument("--short-text-prefix", default="....")
    parser.add_argument("--short-text-strategy", default="dot", choices=["none", "dot", "carrier"])
    parser.add_argument("--short-text-carrier-prefix", default="それでは、")
    parser.add_argument("--short-text-carrier-suffix", default="。")
    parser.add_argument(
        "--warmup-text",
        default="ウォームアップです。",
        help="Startup warmup text. Set empty string to disable warmup.",
    )
    args = parser.parse_args()
    config = load_models_config(args.models_config)

    default_spk_audio_path = args.default_spk_audio_path or str(config.get("default_spk_audio_path", "examples/laffey.mp3"))
    default_prompt_audio_path = args.default_prompt_audio_path or str(
        config.get("default_prompt_audio_path", default_spk_audio_path)
    )
    default_prompt_audio_text = args.default_prompt_audio_text or str(
        config.get("default_prompt_audio_text", "ちが……ちがう。レイア、貴様は間違っている。")
    )

    default_spk_audio_path = resolve_runtime_asset_path(default_spk_audio_path)
    default_prompt_audio_path = resolve_runtime_asset_path(default_prompt_audio_path)

    tts = TTS(
        gpt_cache=build_gpt_cache(args.gpt_cache_len, args.gpt_batch_size),
        sovits_cache=[],
        use_bert=args.use_bert,
        use_flash_attn=args.use_flash_attn,
        models_dir=args.models_dir,
    )

    TTSHTTPHandler.tts = tts
    TTSHTTPHandler.default_spk_audio_path = default_spk_audio_path
    TTSHTTPHandler.default_prompt_audio_path = default_prompt_audio_path
    TTSHTTPHandler.default_prompt_audio_text = default_prompt_audio_text
    TTSHTTPHandler.short_text_boost_enabled = args.short_text_boost
    TTSHTTPHandler.short_text_min_chars = max(1, args.short_text_min_chars)
    TTSHTTPHandler.short_text_prefix = args.short_text_prefix
    TTSHTTPHandler.short_text_strategy = args.short_text_strategy
    TTSHTTPHandler.short_text_carrier_prefix = args.short_text_carrier_prefix
    TTSHTTPHandler.short_text_carrier_suffix = args.short_text_carrier_suffix

    if args.warmup_text.strip():
        print("[INFO] running startup warmup infer...")
        t0 = time.perf_counter()
        _ = tts.infer(
            spk_audio_path=default_spk_audio_path,
            prompt_audio_path=default_prompt_audio_path,
            prompt_audio_text=default_prompt_audio_text,
            text=args.warmup_text,
        )
        dt = time.perf_counter() - t0
        print(f"[OK] warmup done in {dt:.3f}s")

    server = ThreadingHTTPServer((args.host, args.port), TTSHTTPHandler)
    print(f"[OK] server started: http://{args.host}:{args.port}")
    print("[OK] endpoints: GET /health, POST /synthesize")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

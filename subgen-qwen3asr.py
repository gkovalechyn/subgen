subgen_version = "__SUBGEN_VERSION__"

from datetime import datetime
from threading import Lock
import os
import subprocess
import sys
import tempfile
import time
import logging
import gc
import asyncio
from typing import Union
from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import StreamingResponse
import uvicorn
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def convert_to_bool(in_bool):
    return str(in_bool).lower() in ("true", "on", "1", "y", "yes")


# --- Configuration ---
whisper_model            = os.getenv("WHISPER_MODEL", "Qwen/Qwen3-ASR-1.7B")
transcribe_device        = os.getenv("TRANSCRIBE_DEVICE", "cuda")
webhookport              = int(os.getenv("WEBHOOKPORT", 9000))
debug                    = convert_to_bool(os.getenv("DEBUG", True))
huggingface_token        = os.getenv("HUGGINGFACE_TOKEN", "")
force_detected_language_to = os.getenv("FORCE_DETECTED_LANGUAGE_TO", "").lower()
clear_vram_on_complete   = convert_to_bool(os.getenv("CLEAR_VRAM_ON_COMPLETE", True))
append_credits           = convert_to_bool(os.getenv("APPEND", False))
use_forced_aligner       = convert_to_bool(os.getenv("USE_FORCED_ALIGNER", True))
forced_aligner_model     = os.getenv("FORCED_ALIGNER_MODEL", "Qwen/Qwen3-ForcedAligner-0.6B")
max_new_tokens           = int(os.getenv("MAX_NEW_TOKENS", 1024))
max_inference_batch_size = int(os.getenv("MAX_INFERENCE_BATCH_SIZE", 4))
detect_language_length   = int(os.getenv("DETECT_LANGUAGE_LENGTH", 30))
max_segment_chars        = int(os.getenv("MAX_SEGMENT_CHARS", 40))
max_segment_sec          = float(os.getenv("MAX_SEGMENT_SEC", 7.0))
gap_threshold_sec        = float(os.getenv("GAP_THRESHOLD_SEC", 0.5))

if transcribe_device == "gpu":
    transcribe_device = "cuda"

app = FastAPI()
model = None
model_lock = Lock()
transcribe_lock = Lock()

in_docker = os.path.exists("/.dockerenv")
docker_status = "Docker" if in_docker else "Standalone"

# --- Language mappings ---
code_to_qwen_language = {
    "zh": "Chinese",    "en": "English",    "yue": "Cantonese", "ar": "Arabic",
    "de": "German",     "fr": "French",     "es": "Spanish",    "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian",    "ko": "Korean",     "ru": "Russian",
    "th": "Thai",       "vi": "Vietnamese", "ja": "Japanese",   "tr": "Turkish",
    "hi": "Hindi",      "ms": "Malay",      "nl": "Dutch",      "sv": "Swedish",
    "da": "Danish",     "fi": "Finnish",    "pl": "Polish",     "cs": "Czech",
    "fil": "Filipino",  "fa": "Persian",    "el": "Greek",      "hu": "Hungarian",
    "mk": "Macedonian", "ro": "Romanian",
}
qwen_language_to_code = {v.lower(): k for k, v in code_to_qwen_language.items()}


def resolve_language(lang: str) -> str:
    if not lang:
        return None
    lang_lower = lang.lower()
    if lang_lower in qwen_language_to_code:
        return lang.capitalize()
    return code_to_qwen_language.get(lang_lower)


# --- SRT / VTT / TXT formatters ---
SENTENCE_ENDS = frozenset({".", "!", "?", "\u3002", "\uff01", "\uff1f", "\u2026", ";", "\uff1b"})


def _fmt_srt_time(seconds: float) -> str:
    ms = int(round((seconds % 1) * 1000))
    total_s = int(seconds)
    return f"{total_s // 3600:02d}:{(total_s // 60) % 60:02d}:{total_s % 60:02d},{ms:03d}"


def _fmt_vtt_time(seconds: float) -> str:
    ms = int(round((seconds % 1) * 1000))
    total_s = int(seconds)
    return f"{total_s // 3600:02d}:{(total_s // 60) % 60:02d}:{total_s % 60:02d}.{ms:03d}"


def _group_word_timestamps(time_stamps) -> list:
    segments = []
    current_tokens = []
    current_start = None
    prev_end = None

    for token in time_stamps:
        token_start = float(token.start_time)
        token_end = float(token.end_time)

        if current_tokens:
            joined = "".join(t.text for t in current_tokens).strip()
            last_char = current_tokens[-1].text.rstrip()[-1:] if current_tokens[-1].text.rstrip() else ""
            gap = token_start - prev_end if prev_end is not None else 0.0
            seg_duration = (prev_end - current_start) if prev_end is not None else 0.0

            if (
                gap >= gap_threshold_sec
                or last_char in SENTENCE_ENDS
                or len(joined) >= max_segment_chars
                or seg_duration >= max_segment_sec
            ):
                segments.append((current_start, prev_end, joined))
                current_tokens = []
                current_start = None

        if current_start is None:
            current_start = token_start
        current_tokens.append(token)
        prev_end = token_end

    if current_tokens:
        joined = "".join(t.text for t in current_tokens).strip()
        segments.append((current_start, prev_end, joined))

    return segments


def _estimate_segments(text: str) -> list:
    words = text.split()
    if not words:
        return [(0.0, 3.0, text)]
    segments = []
    current_start = 0.0
    for i in range(0, len(words), 12):
        chunk = words[i:i + 12]
        chunk_end = current_start + len(chunk) * 0.4
        segments.append((current_start, chunk_end, " ".join(chunk)))
        current_start = chunk_end
    return segments


def _build_segments(result) -> list:
    ts = getattr(result, "time_stamps", None)
    if ts:
        return _group_word_timestamps(ts)
    return _estimate_segments(getattr(result, "text", "").strip())


def result_to_srt(result) -> str:
    segments = _build_segments(result)
    if append_credits and segments:
        ts_str = datetime.now().strftime("%d %b %Y - %H:%M:%S")
        last_end = segments[-1][1]
        segments.append((last_end + 5.0, last_end + 8.0, f"Transcribed by Qwen3-ASR via Subgen on {ts_str}"))
    lines = []
    for i, (start, end, text) in enumerate(segments, 1):
        lines += [str(i), f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}", text, ""]
    return "\n".join(lines)


def result_to_vtt(result) -> str:
    segments = _build_segments(result)
    if append_credits and segments:
        ts_str = datetime.now().strftime("%d %b %Y - %H:%M:%S")
        last_end = segments[-1][1]
        segments.append((last_end + 5.0, last_end + 8.0, f"Transcribed by Qwen3-ASR via Subgen on {ts_str}"))
    lines = ["WEBVTT", ""]
    for i, (start, end, text) in enumerate(segments, 1):
        lines += [str(i), f"{_fmt_vtt_time(start)} --> {_fmt_vtt_time(end)}", text, ""]
    return "\n".join(lines)


def result_to_txt(result) -> str:
    text = getattr(result, "text", "").strip()
    if append_credits:
        text += f"\nTranscribed by Qwen3-ASR via Subgen on {datetime.now().strftime('%d %b %Y - %H:%M:%S')}"
    return text + "\n"


def result_to_output(result, output_format: str = "srt") -> str:
    if output_format == "txt":
        return result_to_txt(result)
    elif output_format == "vtt":
        return result_to_vtt(result)
    else:
        return result_to_srt(result)


# --- Model management ---
def start_model():
    global model
    with model_lock:
        if model is None:
            logging.info(f"Loading model: {whisper_model}")
            from qwen_asr import Qwen3ASRModel
            import huggingface_hub

            if huggingface_token:
                huggingface_hub.login(token=huggingface_token, add_to_git_credential=False)

            is_cuda = transcribe_device == "cuda" and torch.cuda.is_available()
            dtype = torch.bfloat16 if is_cuda else torch.float32
            device_map = "cuda:0" if is_cuda else "cpu"

            init_kwargs = dict(
                dtype=dtype,
                device_map=device_map,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
            )
            if use_forced_aligner:
                init_kwargs["forced_aligner"] = forced_aligner_model
                init_kwargs["forced_aligner_kwargs"] = dict(dtype=dtype, device_map=device_map)

            model = Qwen3ASRModel.from_pretrained(whisper_model, **init_kwargs)
            logging.info(f"Model loaded: {whisper_model}")


def unload_model():
    global model
    if not clear_vram_on_complete:
        return
    with model_lock:
        if model is not None:
            del model
            model = None
            logging.info("Model unloaded")
        if transcribe_device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


# --- Audio helpers ---
def write_temp_file(file_content: bytes, suffix: str = ".audio") -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(file_content)
        tmp.flush()
        return tmp.name
    finally:
        tmp.close()


def reencode_to_wav(input_path: str, is_raw_pcm: bool = False) -> str:
    """Re-encode any audio input to 16kHz mono WAV via ffmpeg.
    If is_raw_pcm=True, the input is treated as raw int16 PCM at 16kHz mono.
    """
    out_path = input_path + "_reencoded.wav"
    if is_raw_pcm:
        cmd = ["ffmpeg", "-y", "-f", "s16le", "-ar", "16000", "-ac", "1",
               "-i", input_path, out_path]
    else:
        cmd = ["ffmpeg", "-y", "-i", input_path,
               "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", out_path]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg re-encode failed: {result.stderr.decode()}")
    return out_path


def trim_wav(input_path: str, duration_sec: int) -> str:
    """Trim to first N seconds via ffmpeg."""
    out_path = input_path + "_trimmed.wav"
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", input_path, "-t", str(duration_sec), out_path],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {result.stderr.decode()}")
    return out_path


def cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass


# --- Transcription ---
def transcribe_audio(file_content: bytes, language: str, output_format: str, encode: bool = True) -> str:
    raw_path = wav_path = None
    try:
        raw_path = write_temp_file(file_content)
        wav_path = reencode_to_wav(raw_path, is_raw_pcm=not encode)

        start_model()
        if transcribe_device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

        qwen_language = resolve_language(language) if language else None
        results = model.transcribe(
            audio=wav_path,
            language=qwen_language,
            return_time_stamps=use_forced_aligner,
        )
        result = results[0] if results else None
        return result_to_output(result, output_format) if result else None

    finally:
        cleanup(raw_path, wav_path)
        unload_model()


def detect_language_from_audio(file_content: bytes, encode: bool = True) -> tuple:
    raw_path = wav_path = trimmed_path = None
    try:
        raw_path = write_temp_file(file_content)
        wav_path = reencode_to_wav(raw_path, is_raw_pcm=not encode)
        trimmed_path = trim_wav(wav_path, detect_language_length)

        start_model()
        results = model.transcribe(audio=trimmed_path, language=None, return_time_stamps=False)
        if results:
            raw_lang = getattr(results[0], "language", "") or ""
            detected = raw_lang.lower()
            code = qwen_language_to_code.get(detected, detected[:2])
            return detected, code
        return "", ""

    finally:
        cleanup(raw_path, wav_path, trimmed_path)
        unload_model()


# --- Logging setup ---
class MultiplePatternsFilter(logging.Filter):
    def filter(self, record):
        patterns = [
            "header parsing failed", "timescale not set", "srt was added",
            "doesn't have any audio to transcribe", "Setting `pad_token_id`",
        ]
        return not any(pattern in record.getMessage() for pattern in patterns)


logging.basicConfig(
    stream=sys.stderr,
    level=logging.DEBUG if debug else logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
for handler in logging.getLogger().handlers:
    handler.addFilter(MultiplePatternsFilter())
for noisy in ("multipart", "urllib3", "asyncio", "httpcore", "httpx", "huggingface_hub", "transformers"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# --- API endpoints ---
@app.get("/asr")
@app.get("/detect-language")
def handle_get_request(request: Request):
    return {"error": "Use POST. See https://github.com/McCloudS/subgen for configuration."}


@app.get("/")
def webui():
    return {"info": "Configure via environment variables. See https://github.com/McCloudS/subgen"}


@app.get("/status")
def status():
    return {"version": f"Subgen {subgen_version} (Qwen3-ASR)", "model": whisper_model, "runtime": docker_status}


@app.post("/asr")
async def asr(
        task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
        language: Union[str, None] = Query(default=None),
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True, description="Ignored; audio is always re-encoded via ffmpeg"),
        output: Union[str, None] = Query(default="srt", enum=["txt", "vtt", "srt", "tsv", "json"]),
        word_timestamps: bool = Query(default=False),
):
    try:
        file_content = await audio_file.read()
        if not file_content:
            return {"status": "error", "message": "Audio file is empty"}

        effective_language = force_detected_language_to or language
        if force_detected_language_to:
            logging.info(f"Forcing language to: {force_detected_language_to}")
        if task == "translate":
            logging.warning("Qwen3-ASR does not support translation; transcribing instead.")

        logging.info(f"ASR request: language={effective_language or 'auto'}, output={output}")
        start_time = time.time()

        with transcribe_lock:
            srt_content = await asyncio.to_thread(
                transcribe_audio, file_content, effective_language, output or "srt", encode
            )

        m, s = divmod(int(time.time() - start_time), 60)
        logging.info(f"ASR complete in {m}m {s}s")

        if srt_content:
            return StreamingResponse(
                iter([srt_content]),
                media_type="text/plain",
                headers={"Source": "Transcribed using Qwen3-ASR from Subgen!"},
            )
        return {"status": "error", "message": "Transcription failed or produced no output"}

    except Exception as e:
        logging.error(f"ASR error: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}
    finally:
        await audio_file.close()


@app.post("/detect-language")
async def detect_language(
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True),
        detect_lang_length: int = Query(default=30),
):
    if force_detected_language_to:
        logging.info(f"Forcing language to: {force_detected_language_to}")
        await audio_file.close()
        return {"detected_language": force_detected_language_to, "language_code": force_detected_language_to}

    try:
        file_content = await audio_file.read()
        detected_language, language_code = await asyncio.to_thread(
            detect_language_from_audio, file_content, encode
        )
        return {"detected_language": detected_language, "language_code": language_code}

    except Exception as e:
        logging.error(f"Detect-language error: {e}", exc_info=True)
        return {"detected_language": "", "language_code": ""}
    finally:
        await audio_file.close()


if __name__ == "__main__":
    logging.info(f"Subgen v{subgen_version} (Qwen3-ASR backend)")
    logging.info(f"Device: {transcribe_device}, Model: {whisper_model}")
    logging.info(f"Forced aligner: {'enabled (' + forced_aligner_model + ')' if use_forced_aligner else 'disabled'}")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    uvicorn.run("__main__:app", host="0.0.0.0", port=int(webhookport), use_colors=True)

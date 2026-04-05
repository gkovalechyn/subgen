subgen_version = "__SUBGEN_VERSION__"

from datetime import datetime
from threading import Lock, Timer
import os
import threading
import sys
import time
import queue
import logging
import gc
import hashlib
import asyncio
import tempfile
from typing import Union
from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import StreamingResponse
import uvicorn
import numpy as np
import torch

# Reduce CUDA memory fragmentation before PyTorch allocator initializes
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def convert_to_bool(in_bool):
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')

# --- Configuration ---
whisper_model = os.getenv('WHISPER_MODEL', 'Qwen/Qwen3-ASR-1.7B')
concurrent_transcriptions = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', 1))
transcribe_device = os.getenv('TRANSCRIBE_DEVICE', 'cuda')
webhookport = int(os.getenv('WEBHOOKPORT', 9000))
debug = convert_to_bool(os.getenv('DEBUG', True))
model_location = os.getenv('MODEL_PATH', './models')
huggingface_token = os.getenv('HUGGINGFACE_TOKEN', '')
force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', '').lower()
clear_vram_on_complete = convert_to_bool(os.getenv('CLEAR_VRAM_ON_COMPLETE', True))
append_credits = convert_to_bool(os.getenv('APPEND', False))
use_forced_aligner = convert_to_bool(os.getenv('USE_FORCED_ALIGNER', True))
forced_aligner_model = os.getenv('FORCED_ALIGNER_MODEL', 'Qwen/Qwen3-ForcedAligner-0.6B')
max_new_tokens = int(os.getenv('MAX_NEW_TOKENS', 256))
max_inference_batch_size = int(os.getenv('MAX_INFERENCE_BATCH_SIZE', 4))
detect_language_length = int(os.getenv('DETECT_LANGUAGE_LENGTH', 30))
model_cleanup_delay = int(os.getenv('MODEL_CLEANUP_DELAY', 30))

if transcribe_device == 'gpu':
    transcribe_device = 'cuda'

app = FastAPI()
model = None
model_cleanup_timer = None
model_cleanup_lock = Lock()
model_load_lock = Lock()

in_docker = os.path.exists('/.dockerenv')
docker_status = "Docker" if in_docker else "Standalone"

# --- Language mappings ---
# ISO 639-1/BCP-47 code -> Qwen3-ASR language name
code_to_qwen_language = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "hu": "Hungarian",
    "mk": "Macedonian",
    "ro": "Romanian",
}

# Qwen3-ASR language name (lowercase) -> ISO code
qwen_language_to_code = {v.lower(): k for k, v in code_to_qwen_language.items()}


def resolve_language(lang: str) -> str:
    """Convert an ISO code or full language name to the Qwen3-ASR language name."""
    if not lang:
        return None
    lang_lower = lang.lower()
    # Already a full name recognised by Qwen3-ASR
    if lang_lower in qwen_language_to_code:
        return lang.capitalize()
    # ISO code
    return code_to_qwen_language.get(lang_lower)


# --- SRT / VTT / TXT formatters ---
SENTENCE_ENDS = frozenset({'.', '!', '?', '。', '！', '？', '…', ';', '；'})
MAX_SEGMENT_CHARS = 60


def _fmt_srt_time(seconds: float) -> str:
    ms = int(round((seconds % 1) * 1000))
    total_s = int(seconds)
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _fmt_vtt_time(seconds: float) -> str:
    ms = int(round((seconds % 1) * 1000))
    total_s = int(seconds)
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _group_word_timestamps(time_stamps) -> list:
    """Group word-level ForcedAligner timestamps into subtitle segments."""
    segments = []
    current_tokens = []
    current_start = None

    for token in time_stamps:
        if current_start is None:
            current_start = float(token.start_time)
        current_tokens.append(token)

        joined = ''.join(t.text for t in current_tokens).strip()
        last_char = token.text.rstrip()[-1:] if token.text.rstrip() else ''
        ends_sentence = last_char in SENTENCE_ENDS
        too_long = len(joined) >= MAX_SEGMENT_CHARS

        if ends_sentence or too_long:
            end_time = float(current_tokens[-1].end_time)
            segments.append((current_start, end_time, joined))
            current_tokens = []
            current_start = None

    if current_tokens:
        joined = ''.join(t.text for t in current_tokens).strip()
        end_time = float(current_tokens[-1].end_time)
        segments.append((current_start, end_time, joined))

    return segments


def _estimate_segments(text: str) -> list:
    """Estimate timing for text when no forced-aligner timestamps are available."""
    words = text.split()
    if not words:
        return [(0.0, 3.0, text)]
    segments = []
    current_start = 0.0
    WORDS_PER_SEGMENT = 12
    SEC_PER_WORD = 0.4  # ~150 WPM

    for i in range(0, len(words), WORDS_PER_SEGMENT):
        chunk = words[i:i + WORDS_PER_SEGMENT]
        chunk_end = current_start + len(chunk) * SEC_PER_WORD
        segments.append((current_start, chunk_end, ' '.join(chunk)))
        current_start = chunk_end

    return segments


def _build_segments(result) -> list:
    ts = getattr(result, 'time_stamps', None)
    if ts:
        return _group_word_timestamps(ts)
    return _estimate_segments(getattr(result, 'text', '').strip())


def result_to_srt(result) -> str:
    segments = _build_segments(result)
    if append_credits and segments:
        ts_str = datetime.now().strftime("%d %b %Y - %H:%M:%S")
        credit = f"Transcribed by Qwen3-ASR via Subgen on {ts_str}"
        last_end = segments[-1][1]
        segments.append((last_end + 5.0, last_end + 8.0, credit))

    lines = []
    for i, (start, end, text) in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def result_to_vtt(result) -> str:
    segments = _build_segments(result)
    if append_credits and segments:
        ts_str = datetime.now().strftime("%d %b %Y - %H:%M:%S")
        credit = f"Transcribed by Qwen3-ASR via Subgen on {ts_str}"
        last_end = segments[-1][1]
        segments.append((last_end + 5.0, last_end + 8.0, credit))

    lines = ["WEBVTT", ""]
    for i, (start, end, text) in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_vtt_time(start)} --> {_fmt_vtt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def result_to_txt(result) -> str:
    text = getattr(result, 'text', '').strip()
    if append_credits:
        ts_str = datetime.now().strftime("%d %b %Y - %H:%M:%S")
        text += f"\nTranscribed by Qwen3-ASR via Subgen on {ts_str}"
    return text + "\n"


def result_to_output(result, output_format: str = 'srt') -> str:
    if output_format == 'txt':
        return result_to_txt(result)
    elif output_format == 'vtt':
        return result_to_vtt(result)
    else:
        return result_to_srt(result)


# --- Deduplicated priority queue ---
class DeduplicatedQueue(queue.PriorityQueue):
    def __init__(self):
        super().__init__()
        self._queued = set()
        self._processing = set()
        self._lock = Lock()

    def put(self, item, block=True, timeout=None):
        with self._lock:
            task_id = item["path"]
            if task_id not in self._queued and task_id not in self._processing:
                task_type = item.get("type", "asr")
                priority = 0 if task_type == "detect_language" else 1
                super().put((priority, time.time(), item), block, timeout)
                self._queued.add(task_id)
                return True
            return False

    def get(self, block=True, timeout=None):
        priority, timestamp, item = super().get(block, timeout)
        with self._lock:
            task_id = item["path"]
            self._queued.discard(task_id)
            self._processing.add(task_id)
        return item

    def mark_done(self, item):
        with self._lock:
            self._processing.discard(item["path"])

    def is_idle(self):
        with self._lock:
            return self.empty() and len(self._processing) == 0

    def is_active(self, task_id):
        with self._lock:
            return task_id in self._queued or task_id in self._processing

    def get_queued_count(self):
        with self._lock:
            return len(self._queued)

    def get_processing_count(self):
        with self._lock:
            return len(self._processing)


task_queue = DeduplicatedQueue()


def transcription_worker():
    while True:
        task = None
        try:
            task = task_queue.get(block=True, timeout=1)
            path = task.get("path", "unknown")
            display_name = os.path.basename(path) if ("/" in str(path) or "\\" in str(path)) else path
            proc_count = task_queue.get_processing_count()
            queue_count = task_queue.get_queued_count()
            logging.info(f"WORKER START: {display_name:^40} | Jobs: {proc_count} processing, {queue_count} queued")
            start_time = time.time()

            asr_task_worker(task)

            elapsed = time.time() - start_time
            m, s = divmod(int(elapsed), 60)
            remaining = task_queue.get_queued_count()
            logging.info(f"WORKER FINISH: {display_name:^40} in {m}m {s}s | Remaining: {remaining} queued")
        except queue.Empty:
            continue
        except Exception as e:
            logging.error(f"Error processing task: {e}", exc_info=True)
        finally:
            if task:
                task_queue.task_done()
                task_queue.mark_done(task)
                delete_model()


# --- Logging setup ---
class MultiplePatternsFilter(logging.Filter):
    def filter(self, record):
        patterns = [
            "header parsing failed",
            "timescale not set",
            "srt was added",
            "doesn't have any audio to transcribe",
        ]
        return not any(pattern in record.getMessage() for pattern in patterns)


if debug:
    level = logging.DEBUG
else:
    level = logging.INFO

logging.basicConfig(
    stream=sys.stderr,
    level=level,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger()
logger.setLevel(level)

for handler in logger.handlers:
    handler.addFilter(MultiplePatternsFilter())

logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

for _ in range(concurrent_transcriptions):
    threading.Thread(target=transcription_worker, daemon=True).start()


# --- Model management ---
def start_model():
    global model
    with model_load_lock:
        if model is None:
            logging.info(f"Loading Qwen3-ASR model: {whisper_model}")
            from qwen_asr import Qwen3ASRModel
            import huggingface_hub

            if huggingface_token:
                huggingface_hub.login(token=huggingface_token, add_to_git_credential=False)

            is_cuda = transcribe_device == 'cuda' and torch.cuda.is_available()
            dtype = torch.bfloat16 if is_cuda else torch.float32
            device_map = 'cuda:0' if is_cuda else 'cpu'

            init_kwargs = dict(
                dtype=dtype,
                device_map=device_map,
                max_inference_batch_size=max_inference_batch_size,
                max_new_tokens=max_new_tokens,
            )

            if use_forced_aligner:
                init_kwargs['forced_aligner'] = forced_aligner_model
                init_kwargs['forced_aligner_kwargs'] = dict(
                    dtype=dtype,
                    device_map=device_map,
                )

            model = Qwen3ASRModel.from_pretrained(whisper_model, **init_kwargs)
            logging.info(f"Model loaded: {whisper_model}")


def schedule_model_cleanup():
    global model_cleanup_timer
    with model_cleanup_lock:
        if model_cleanup_timer is not None:
            model_cleanup_timer.cancel()
            model_cleanup_timer.join()
        model_cleanup_timer = Timer(model_cleanup_delay, perform_model_cleanup)
        model_cleanup_timer.daemon = True
        model_cleanup_timer.start()
        logging.debug(f"Model cleanup scheduled in {model_cleanup_delay} seconds")


def perform_model_cleanup():
    global model, model_cleanup_timer
    with model_cleanup_lock:
        if clear_vram_on_complete and task_queue.is_idle():
            logging.debug("Queue idle; clearing model from memory.")
            if model:
                try:
                    del model
                    model = None
                    logging.info("Model unloaded from memory")
                except Exception as e:
                    logging.error(f"Error unloading model: {e}")
            if transcribe_device == 'cuda':
                torch.cuda.empty_cache()
                logging.debug("CUDA cache cleared")
        else:
            logging.debug("Queue not idle or clear_vram disabled; skipping model cleanup")
        gc.collect()
        model_cleanup_timer = None


def delete_model():
    if not clear_vram_on_complete:
        return
    if task_queue.is_idle():
        schedule_model_cleanup()
    else:
        logging.debug("Tasks still in queue; skipping model cleanup scheduling.")


# --- Result storage ---
_asr_results = {}
_asr_results_lock = Lock()


def _prepare_audio(file_content: bytes, encode: bool):
    """Return audio in the format qwen-asr expects."""
    if not encode:
        # Pre-decoded PCM int16 at 16 kHz from Bazarr
        audio_array = np.frombuffer(file_content, np.int16).flatten().astype(np.float32) / 32768.0
        return (audio_array, 16000)
    else:
        # Encoded file bytes (wav/mp3/etc.) — write to a temp file
        suffix = '.audio'
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tmp.write(file_content)
            tmp.flush()
            tmp_path = tmp.name
        finally:
            tmp.close()
        return tmp_path


def asr_task_worker(task_data: dict):
    task_id = task_data.get('path', 'unknown')
    tmp_path = None
    try:
        language = task_data['language']
        file_content = task_data['audio_content']
        encode = task_data['encode']

        start_model()

        audio_input = _prepare_audio(file_content, encode)
        if isinstance(audio_input, str):
            tmp_path = audio_input  # path to clean up later

        qwen_language = resolve_language(language) if language else None

        if transcribe_device == 'cuda':
            gc.collect()
            torch.cuda.empty_cache()

        results = model.transcribe(
            audio=audio_input,
            language=qwen_language,
            return_time_stamps=use_forced_aligner,
        )

        result = results[0] if results else None

        with _asr_results_lock:
            _asr_results[task_id] = result

    except torch.cuda.OutOfMemoryError as e:
        logging.error(f"OOM processing ASR (ID: {task_id}): {e}")
        gc.collect()
        torch.cuda.empty_cache()
        with _asr_results_lock:
            _asr_results[task_id] = None
    except Exception as e:
        logging.error(f"Error processing ASR (ID: {task_id}): {e}", exc_info=True)
        with _asr_results_lock:
            _asr_results[task_id] = None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def asr_wait_for_result(task_id: str, timeout: int = 18000):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _asr_results_lock:
            if task_id in _asr_results:
                return _asr_results.pop(task_id)
        time.sleep(0.5)
    return None


# --- API endpoints ---
@app.get("/asr")
@app.get("/detect-language")
def handle_get_request(request: Request):
    return {"You accessed this request incorrectly via a GET request.  See https://github.com/McCloudS/subgen for proper configuration"}


@app.get("/")
def webui():
    return {"The webui for configuration was removed on 1 October 2024, please configure via environment variables or in your Docker settings."}


@app.get("/status")
def status():
    return {"version": f"Subgen {subgen_version} (Qwen3-ASR), model={whisper_model} ({docker_status})"}


@app.post("/asr")
async def asr(
        task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
        language: Union[str, None] = Query(default=None),
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
        output: Union[str, None] = Query(default="srt", enum=["txt", "vtt", "srt", "tsv", "json"]),
        word_timestamps: bool = Query(default=False, description="Word level timestamps"),
):
    task_id = None
    try:
        logging.info("Transcribing file from Bazarr/ASR webhook")

        file_content = await audio_file.read()
        if not file_content:
            await audio_file.close()
            return {"status": "error", "message": "Audio file is empty"}

        audio_hash = hashlib.sha256(file_content + (language or '').encode()).hexdigest()[:16]
        task_id = f"asr-{audio_hash}"

        effective_language = force_detected_language_to or language
        if force_detected_language_to:
            logging.info(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: forcing language to {force_detected_language_to}")

        if task == "translate":
            logging.warning("Qwen3-ASR does not support translation; transcribing instead.")

        asr_task_data = {
            'path': task_id,
            'type': 'asr',
            'task': 'transcribe',
            'language': effective_language,
            'audio_content': file_content,
            'encode': encode,
            'output': output,
            'word_timestamps': word_timestamps,
        }

        if not task_queue.put(asr_task_data):
            logging.info(f"ASR task {task_id} already queued/processing")

        result = await asyncio.to_thread(asr_wait_for_result, task_id)

    except Exception as e:
        logging.error(f"Error processing Bazarr request {audio_file.filename}: {e}", exc_info=True)
        result = None
    finally:
        await audio_file.close()

    if result:
        srt_content = result_to_output(result, output or 'srt')
        return StreamingResponse(
            iter([srt_content]),
            media_type="text/plain",
            headers={"Source": "Transcribed using Qwen3-ASR from Subgen!"},
        )
    else:
        return {"status": "error", "message": "Transcription failed"}


@app.post("/detect-language")
async def detect_language(
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
        detect_lang_length: int = Query(default=30, description="Detect language on the first X seconds of the file"),
):
    detected_language = ""
    language_code = ""

    if force_detected_language_to:
        logging.info(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: forcing language to {force_detected_language_to}")
        detected_language = force_detected_language_to
        language_code = force_detected_language_to
        await audio_file.close()
        return {"detected_language": detected_language, "language_code": language_code}

    effective_detect_length = detect_lang_length
    if effective_detect_length != 30:
        logging.info(f"Detect language on the first {effective_detect_length} seconds of the audio.")

    try:
        await asyncio.to_thread(start_model)

        audio_file.file.seek(0)
        raw = audio_file.file.read()

        # Use raw PCM path for truncation; otherwise pass full file
        audio_array = np.frombuffer(raw, np.int16).flatten().astype(np.float32) / 32768.0
        target_samples = 16000 * effective_detect_length
        if len(audio_array) > target_samples:
            audio_array = audio_array[:target_samples]
        audio_input = (audio_array, 16000)

        results = await asyncio.to_thread(
            lambda: model.transcribe(audio=audio_input, language=None, return_time_stamps=False)
        )

        if results:
            raw_lang = getattr(results[0], 'language', '') or ''
            detected_language = raw_lang.lower()
            language_code = qwen_language_to_code.get(detected_language, detected_language[:2])

    except Exception as e:
        logging.error(f"Error detecting language for {audio_file.filename}: {e}", exc_info=True)
    finally:
        await audio_file.close()
        delete_model()

    return {"detected_language": detected_language, "language_code": language_code}


if __name__ == "__main__":
    logging.info(f"Subgen v{subgen_version} (Qwen3-ASR backend)")
    logging.info(f"Concurrent transcriptions: {concurrent_transcriptions}")
    logging.info(f"Transcribe device: {transcribe_device}, Model: {whisper_model}")
    logging.info(f"Forced aligner: {'enabled (' + forced_aligner_model + ')' if use_forced_aligner else 'disabled'}")
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    uvicorn.run("__main__:app", host="0.0.0.0", port=int(webhookport), use_colors=True)

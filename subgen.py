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
from typing import Union
from fastapi import FastAPI, File, UploadFile, Query, Request
from fastapi.responses import StreamingResponse
import uvicorn
import numpy as np
import stable_whisper
from stable_whisper import Segment
import ast
import faster_whisper
import huggingface_hub

def get_key_by_value(d, value):
    reverse_dict = {v: k for k, v in d.items()}
    return reverse_dict.get(value)

def convert_to_bool(in_bool):
    # Convert the input to string and lower case, then check against true values
    return str(in_bool).lower() in ('true', 'on', '1', 'y', 'yes')
    
whisper_model = os.getenv('WHISPER_MODEL', 'medium')
whisper_threads = int(os.getenv('WHISPER_THREADS', 4))
concurrent_transcriptions = int(os.getenv('CONCURRENT_TRANSCRIPTIONS', 2))
transcribe_device = os.getenv('TRANSCRIBE_DEVICE', 'cpu')
webhookport = int(os.getenv('WEBHOOKPORT', 9000))
word_level_highlight = convert_to_bool(os.getenv('WORD_LEVEL_HIGHLIGHT', False))
debug = convert_to_bool(os.getenv('DEBUG', True))
model_location = os.getenv('MODEL_PATH', './models')
huggingface_token = os.getenv('HUGGINGFACE_TOKEN', '')
force_detected_language_to = os.getenv('FORCE_DETECTED_LANGUAGE_TO', '').lower()
clear_vram_on_complete = convert_to_bool(os.getenv('CLEAR_VRAM_ON_COMPLETE', True))
compute_type = os.getenv('COMPUTE_TYPE', 'auto')
append = convert_to_bool(os.getenv('APPEND', False))
custom_regroup = os.getenv('CUSTOM_REGROUP', 'cm_sl=84_sl=42++++++1')
detect_language_length = os.getenv('DETECT_LANGUAGE_LENGTH', 30)

try:
    kwargs = ast.literal_eval(os.getenv('SUBGEN_KWARGS', '{}') or '{}')
except ValueError:
    kwargs = {}
    logging.info("kwargs (SUBGEN_KWARGS) is an invalid dictionary, defaulting to empty '{}'")
    
if transcribe_device == "gpu":
    transcribe_device = "cuda"

app = FastAPI()
model = None
model_cleanup_timer = None
model_cleanup_lock = Lock()
model_load_lock = Lock()

in_docker = os.path.exists('/.dockerenv')
docker_status = "Docker" if in_docker else "Standalone"
model_cleanup_delay = int(os.getenv('MODEL_CLEANUP_DELAY', 30))

# Deduplicated priority queue to prevent duplicate tasks
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
            task_id = item["path"]
            self._processing.discard(task_id)

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

# Define a filter class
class MultiplePatternsFilter(logging.Filter):
    def filter(self, record):
        # Define the patterns to search for
        patterns = [
            "Compression ratio threshold is not met",
            "Processing segment at",
            "Log probability threshold is",
            "Reset prompt",
            "Attempting to release",
            "released on ",
            "Attempting to acquire",
            "acquired on",
            "header parsing failed",
            "timescale not set",
            "misdetection possible",
            "srt was added",
            "doesn't have any audio to transcribe",
        ]
        # Return False if any of the patterns are found, True otherwise
        return not any(pattern in record.getMessage() for pattern in patterns)

# Configure logging
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

# Get the root logger
logger = logging.getLogger()
logger.setLevel(level)  # Set the logger level

for handler in logger.handlers:
    handler.addFilter(MultiplePatternsFilter())

logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

for _ in range(concurrent_transcriptions):
    threading.Thread(target=transcription_worker, daemon=True).start()

class ProgressHandler:
    def __init__(self, filename):
        self.filename = filename
        self.start_time = time.time()
        self.last_print_time = 0
        self.interval = 5

    def __call__(self, seek, total):
        if docker_status == 'Docker' or debug:
            current_time = time.time()
            if self.last_print_time == 0 or (current_time - self.last_print_time) >= self.interval:
                self.last_print_time = current_time
                pct = int((seek / total) * 100) if total > 0 else 0
                elapsed = current_time - self.start_time
                speed = seek / elapsed if elapsed > 0 else 0
                eta = (total - seek) / speed if speed > 0 else 0

                def fmt_t(seconds):
                    m, s = divmod(int(seconds), 60)
                    h, m = divmod(m, 60)
                    if h > 0:
                        return f"{h}:{m:02d}:{s:02d}"
                    return f"{m:02d}:{s:02d}"

                proc = task_queue.get_processing_count()
                queued = task_queue.get_queued_count()
                clean_name = (self.filename[:37] + '..') if len(self.filename) > 40 else self.filename

                logging.info(
                    f"[ {clean_name:<40}] {pct:>3}% | "
                    f"{int(seek):>5}/{int(total):<5}s "
                    f"[{fmt_t(elapsed):>5}<{fmt_t(eta):>5}, {speed:>5.2f}s/s] | "
                    f"Jobs: {proc} processing, {queued} queued"
                )

TIME_OFFSET = 5

def appendLine(result):
    if append:
        lastSegment = result.segments[-1]
        date_time_str = datetime.now().strftime("%d %b %Y - %H:%M:%S")
        appended_text = f"Transcribed by whisperAI with faster-whisper ({whisper_model}) on {date_time_str}"
        
        # Create a new segment with the updated information
        newSegment = Segment(
            start=lastSegment.start + TIME_OFFSET,
            end=lastSegment.end + TIME_OFFSET,
            text=appended_text,
            words=[],  # Empty list for words
            id=lastSegment.id + 1
        )
        
        # Append the new segment to the result's segments
        result.segments.append(newSegment)

@app.get("/asr")
@app.get("/detect-language")
def handle_get_request(request: Request):
    return {"You accessed this request incorrectly via a GET request.  See https://github.com/McCloudS/subgen for proper configuration"}

@app.get("/")
def webui():
    return {"The webui for configuration was removed on 1 October 2024, please configure via environment variables or in your Docker settings."}

@app.get("/status")
def status():
    return {"version" : f"Subgen {subgen_version}, stable-ts {stable_whisper.__version__}, faster-whisper {faster_whisper.__version__} ({docker_status})"}

# idea and some code for asr and detect language from https://github.com/ahmetoner/whisper-asr-webservice
@app.post("//asr")
@app.post("/asr")
async def asr(
        task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
        language: Union[str, None] = Query(default=None),
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),  #not used by Bazarr/always False
        output: Union[str, None] = Query(default="srt", enum=["txt", "vtt", "srt", "tsv", "json"]),
        word_timestamps: bool = Query(default=False, description="Word level timestamps") #not used by Bazarr
):
    task_id = None
    try:
        logging.info(f"Transcribing file from Bazarr/ASR webhook")
        result = None

        file_content = await audio_file.read()
        if not file_content:
            await audio_file.close()
            return {"status": "error", "message": "Audio file is empty"}

        audio_hash = hashlib.sha256(file_content + (task or '').encode() + (language or '').encode()).hexdigest()[:16]
        task_id = f"asr-{audio_hash}"

        if force_detected_language_to:
            language = force_detected_language_to
            logging.info(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: Forcing detected language to {force_detected_language_to}")

        asr_task_data = {
            'path': task_id,
            'type': 'asr',
            'task': task,
            'language': language,
            'audio_content': file_content,
            'encode': encode,
            'output': output,
            'word_timestamps': word_timestamps,
        }

        if not task_queue.put(asr_task_data):
            logging.info(f"ASR task {task_id} already queued/processing")

        # Wait for result from worker
        result = await asyncio.to_thread(asr_wait_for_result, task_id)

    except Exception as e:
        logging.error(f"Error processing or transcribing Bazarr {audio_file.filename}: {e}", exc_info=True)
    finally:
        await audio_file.close()

    if result:
        return StreamingResponse(
            iter(result.to_srt_vtt(filepath=None, word_level=word_level_highlight)),
            media_type="text/plain",
            headers={
                'Source': 'Transcribed using stable-ts from Subgen!',
            })
    else:
        return {"status": "error", "message": "Transcription failed"}

@app.post("//detect-language")
@app.post("/detect-language")
async def detect_language(
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
        detect_lang_length: int = Query(default=30, description="Detect language on the first X seconds of the file")
):
    detected_language = ""
    language_code = ""
    if force_detected_language_to:
        logging.info(f"ENV FORCE_DETECTED_LANGUAGE_TO is set: Forcing detected language to {force_detected_language_to}")
    effective_detect_length = int(detect_lang_length) if int(detect_lang_length) != 30 else int(detect_language_length)
    if effective_detect_length != 30:
        logging.info(f"Detect language is set to detect on the first {effective_detect_length} seconds of the audio.")
    try:
        await asyncio.to_thread(start_model)

        args = {}
        audio_file.file.seek(0)
        args['input_sr'] = 16000
        audio_array = np.frombuffer(audio_file.file.read(), np.int16).flatten().astype(np.float32) / 32768.0
        target_length = args['input_sr'] * effective_detect_length
        if len(audio_array) > target_length:
            audio_array = audio_array[:target_length]
        elif len(audio_array) < target_length:
            audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
        args['audio'] = audio_array

        args.update(kwargs)
        args['verbose'] = False
        result = await asyncio.to_thread(lambda: model.transcribe(**args))
        detected_language = result.language
        language_code = get_key_by_value(whisper_languages, detected_language)

    except Exception as e:
        logging.error(f"Error processing or transcribing Bazarr {audio_file.filename}: {e}", exc_info=True)

    finally:
        await audio_file.close()
        delete_model()

        return {"detected_language": detected_language, "language_code": language_code}

# Result storage for ASR tasks processed by the worker
_asr_results = {}
_asr_results_lock = Lock()

def asr_task_worker(task_data: dict):
    """Worker function that processes ASR tasks from the queue."""
    task_id = task_data.get('path', 'unknown')
    try:
        task = task_data['task']
        language = task_data['language']
        file_content = task_data['audio_content']
        encode = task_data['encode']

        start_model()

        args = {}
        display_name = task_id
        if '/' not in whisper_model:
            args['progress_callback'] = ProgressHandler(display_name)

        if not encode:
            args['audio'] = np.frombuffer(file_content, np.int16).flatten().astype(np.float32) / 32768.0
            args['input_sr'] = 16000
        else:
            args['audio'] = file_content

        if custom_regroup and custom_regroup.lower() != 'default':
            args['regroup'] = custom_regroup

        args.update(kwargs)

        result = model.transcribe(task=task, language=language, **args, verbose=None)
        appendLine(result)

        with _asr_results_lock:
            _asr_results[task_id] = result

    except Exception as e:
        logging.error(f"Error processing ASR (ID: {task_id}): {e}", exc_info=True)
        with _asr_results_lock:
            _asr_results[task_id] = None

def asr_wait_for_result(task_id: str, timeout: int = 18000):
    """Block until the ASR result is available."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _asr_results_lock:
            if task_id in _asr_results:
                return _asr_results.pop(task_id)
        time.sleep(0.5)
    return None

def start_model():
    global model
    with model_load_lock:
        if model is None:
            logging.debug("Model was purged, need to re-create")
            if huggingface_token:
                huggingface_hub.login(token=huggingface_token, add_to_git_credential=False)
            if '/' in whisper_model:
                logging.info(f"Loading HuggingFace model: {whisper_model}")
                model = stable_whisper.load_hf_whisper(whisper_model, device=transcribe_device)
            else:
                hf_kwargs = {'huggingface_token': huggingface_token} if huggingface_token else {}
                model = stable_whisper.load_faster_whisper(whisper_model, download_root=model_location, device=transcribe_device, cpu_threads=whisper_threads, num_workers=concurrent_transcriptions, compute_type=compute_type, **hf_kwargs)

def schedule_model_cleanup():
    """Schedule model cleanup with a delay to allow concurrent requests."""
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
    """Actually perform the model cleanup."""
    global model, model_cleanup_timer
    with model_cleanup_lock:
        if clear_vram_on_complete and task_queue.is_idle():
            logging.debug("Queue idle; clearing model from memory.")
            if model:
                try:
                    if hasattr(model, 'model') and hasattr(model.model, 'unload_model'):
                        model.model.unload_model()
                    del model
                    model = None
                    logging.info("Model unloaded from memory")
                except Exception as e:
                    logging.error(f"Error unloading model: {e}")
        else:
            logging.debug("Queue not idle or clear_vram disabled; skipping model cleanup")
        gc.collect()
        model_cleanup_timer = None

def delete_model():
    """Schedule cleanup only when system is actually idle."""
    if not clear_vram_on_complete:
        return
    if task_queue.is_idle():
        schedule_model_cleanup()
    else:
        logging.debug("Tasks still in queue; skipping model cleanup scheduling.")

whisper_languages = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
}

if __name__ == "__main__":
    logging.info(f"Subgen v{subgen_version}")
    logging.info(f"Threads: {str(whisper_threads)}, Concurrent transcriptions: {str(concurrent_transcriptions)}")
    logging.info(f"Transcribe device: {transcribe_device}, Model: {whisper_model}")
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    uvicorn.run("__main__:app", host="0.0.0.0", port=int(webhookport), use_colors=True)

[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/donate/?hosted_button_id=SU4QQP6LH5PF6)
<img src="https://raw.githubusercontent.com/McCloudS/subgen/main/icon.png" width="200">

<details>
<summary>Updates:</summary>

3 Apr 2026: Added support for HuggingFace model IDs in `WHISPER_MODEL` (e.g. `litagin/anime-whisper`). When the model name contains a `/`, subgen will automatically download and load it from HuggingFace using the Transformers backend (`stable-ts[hf]`). Set `HUGGINGFACE_TOKEN` to access gated or private models.

30 Sept 2024: Removed webui

5 Sept 2024: Fixed Emby response to a test message/notification.  Clarified Emby/Plex/Jellyfin instructions for paths.

14 Aug 2024: Cleaned up usage of kwargs across the board a bit.  Added ability for /asr to encode or not, so you don't need to worry about what files/formats you upload.

3 Aug 2024: Added SUBGEN_KWARGS environment variable which allows you to override the model.transcribe with most options you'd like from whisper, faster-whisper, or stable-ts.  This won't be exposed via the webui, it's best to set directly.

21 Apr 2024: Fixed queuing with thanks to https://github.com/xhzhu0628 @ https://github.com/McCloudS/subgen/pull/85.  Bazarr intentionally doesn't follow `CONCURRENT_TRANSCRIPTIONS` because it needs a time sensitive response.

31 Mar 2024: Removed `/subsync` endpoint and general refactoring.  Open an issue if you were using it!

24 Mar 2024: ~~Added a 'webui' to configure environment variables.  You can use this instead of manually editing the script or using Environment Variables in your OS or Docker (if you want).  The config will prioritize OS Env Variables, then the .env file, then the defaults.  You can access it at `http://subgen:9000/`~~

23 Mar 2024: Added `CUSTOM_REGROUP` to try to 'clean up' subtitles a bit.  

22 Mar 2024: Added LRC capability via see: `'LRC_FOR_AUDIO_FILES' | True | Will generate LRC (instead of SRT) files for filetypes: '.mp3', '.flac', '.wav', '.alac', '.ape', '.ogg', '.wma', '.m4a', '.m4b', '.aac', '.aiff' |`

21 Mar 2024: Added a 'wizard' into the launcher that will help standalone users get common Bazarr variables configured.  See below in Launcher section.  Removed 'Transformers' as an option.  While I usually don't like to remove features, I don't think anyone is using this and the results are wildly unpredictable and often cause out of memory errors.  Added two new environment variables called `USE_MODEL_PROMPT` and `CUSTOM_MODEL_PROMPT`.  If `USE_MODEL_PROMPT` is `True` it will use `CUSTOM_MODEL_PROMPT` if set, otherwise will default to using the pre-configured language pairings, such as: `"en": "Hello, welcome to my lecture.",
    "zh": "你好，欢迎来到我的讲座。"`  These pre-configurated translations are geared towards fixing some audio that may not have punctionation.  We can prompt it to try to force the use of punctuation during transcription.

19 Mar 2024: Added a `MONITOR` environment variable.  Will 'watch' or 'monitor' your `TRANSCRIBE_FOLDERS` for changes and run on them.  Useful if you just want to paste files into a folder and get subtitles.   

6 Mar 2024: Added a `/subsync` endpoint that can attempt to align/synchronize subtitles to a file.  Takes audio_file, subtitle_file, language (2 letter code), and outputs an srt.

5 Mar 2024: Cleaned up logging. Added timestamps option (if Debug = True, timestamps will print in logs).

4 Mar 2024: Updated Dockerfile CUDA to 12.2.2 (From CTranslate2).  Added endpoint `/status` to return Subgen version.  Can also use distil models now!  See variables below!

29 Feb 2024: Changed sefault port to align with whisper-asr and deconflict other consumers of the previous port.

11 Feb 2024: Added a 'launcher.py' file for Docker to prevent huge image downloads. Now set UPDATE to True if you want pull the latest version, otherwise it will default to what was in the image on build.  Docker builds will still be auto-built on any commit.  If you don't want to use the auto-update function, no action is needed on your part and continue to update docker images as before.  Fixed bug where detect-langauge could return an empty result.  Reduced useless debug output that was spamming logs and defaulted DEBUG to True.  Added APPEND, which will add f"Transcribed by whisperAI with faster-whisper ({whisper_model}) on {datetime.now()}" at the end of a subtitle.

10 Feb 2024: Added some features from JaiZed's branch such as skipping if SDH subtitles are detected, functions updated to also be able to transcribe audio files, allow individual files to be manually transcribed, and a better implementation of forceLanguage. Added `/batch` endpoint (Thanks JaiZed).  Allows you to navigate in a browser to http://subgen_ip:9000/docs and call the batch endpoint which can take a file or a folder to manually transcribe files.  Added CLEAR_VRAM_ON_COMPLETE, HF_TRANSFORMERS, HF_BATCH_SIZE.  Hugging Face Transformers boast '9x increase', but my limited testing shows it's comparable to faster-whisper or slightly slower.  I also have an older 8gb GPU.  Simplest way to persist HF Transformer models is to set "HF_HUB_CACHE" and set it to "/subgen/models" for Docker (assuming you have the matching volume).

8 Feb 2024: Added FORCE_DETECTED_LANGUAGE_TO to force a wrongly detected language.  Fixed asr to actually use the language passed to it.  

5 Feb 2024: General housekeeping, minor tweaks on the TRANSCRIBE_FOLDERS function.

28 Jan 2024: Fixed issue with ffmpeg python module not importing correctly.  Removed separate GPU/CPU containers.  Also removed the script from installing packages, which should help with odd updates I can't control (from other packages/modules). The image is a couple gigabytes larger, but allows easier maintenance.  

19 Dec 2023: Added the ability for Plex and Jellyfin to automatically update metadata so the subtitles shows up properly on playback. (See https://github.com/McCloudS/subgen/pull/33 from Rikiar73574)  

31 Oct 2023: Added Bazarr support via Whipser provider.

25 Oct 2023: Added Emby (IE http://192.168.1.111:9000/emby) support and TRANSCRIBE_FOLDERS, which will recurse through the provided folders and generate subtitles.  It's geared towards attempting to transcribe existing media without using a webhook.

23 Oct 2023: There are now two docker images, ones for CPU (it's smaller): mccloud/subgen:latest, mccloud/subgen:cpu, the other is for cuda/GPU: mccloud/subgen:cuda.  I also added Jellyfin support and considerable cleanup in the script. I also renamed the webhooks, so they will require new configuration/updates on your end. Instead of /webhook they are now /plex, /tautulli, and /jellyfin.

22 Oct 2023: The script should have backwards compability with previous envirionment settings, but just to be sure, look at the new options below.  If you don't want to manually edit your environment variables, just edit the script manually. While I have added GPU support, I haven't tested it yet.

19 Oct 2023: And we're back!  Uses faster-whisper and stable-ts.  Shouldn't break anything from previous settings, but adds a couple new options that aren't documented at this point in time.  As of now, this is not a docker image on dockerhub.  The potential intent is to move this eventually to a pure python script, primarily to simplify my efforts.  Quick and dirty to meet dependencies: pip or `pip3 install flask requests stable-ts faster-whisper`

This potentially has the ability to use CUDA/Nvidia GPU's, but I don't have one set up yet.  Tesla T4 is in the mail!

2 Feb 2023: Added Tautulli webhooks back in.  Didn't realize Plex webhooks was PlexPass only.  See below for instructions to add it back in.

31 Jan 2023 : Rewrote the script substantially to remove Tautulli and fix some variable handling.  For some reason my implementation requires the container to be in host mode.  My Plex was giving "401 Unauthorized" when attempt to query from docker subnets during API calls. (**Fixed now, it can be in bridge**)

</details>

# Changes made from McCloudS/subgen
* **Stripped down to Bazarr/ASR-only** — removed all Plex, Jellyfin, Emby, and Tautulli webhook endpoints (`/plex`, `/jellyfin`, `/emby`, `/tautulli`, `/batch`). Only `/asr`, `/detect-language`, and `/status` remain.
* Removed all media-server-specific logic: path mapping, `TRANSCRIBE_FOLDERS`, `MONITOR`, metadata refresh calls, `SKIPIFINTERNALSUBLANG`, `SKIPIFEXTERNALSUB`, `NAMESUBLANG`, `SKIP_LANG_CODES`, `LRC_FOR_AUDIO_FILES`, `TRANSCRIBE_OR_TRANSLATE`, `USE_MODEL_PROMPT`, `CUSTOM_MODEL_PROMPT`, `PROCADDEDMEDIA`, `PROCMEDIAONPLAY`.
* Removed `launcher.py` and automatic update logic (`UPDATE` variable).
* Added HuggingFace Transformers backend: set `WHISPER_MODEL` to a HuggingFace model ID (e.g. `litagin/anime-whisper`) and it will be downloaded and loaded via `stable-ts[hf]` automatically.
* Added `HUGGINGFACE_TOKEN` environment variable for gated/private HuggingFace models.
* Downgraded to CUDA 12.2 so it can run on older Linux hosts.
* Pinned ctranslate2.
* Fixed Docker build that was always using the original repo in Dockerfiles.
* Changed to manual versioning and build on tags.

# Examples
Running large-v3 on a GTX 1070 in OpenMediaVault 7 (Debian 12):
![image](https://github.com/user-attachments/assets/7a52f157-37f0-4344-b8c3-88dba58f8204)
![image](https://github.com/user-attachments/assets/0a113383-c4f8-4ea4-a2f3-81964f7b8e3e)


# Running instructions:
## Docker compose
```yaml
  whisper:
    image: gkovalechyn/subgen:v24-11-h3-gpu-cu118
    restart: unless-stopped
    volumes:
      - /data/subgen/models:/subgen/models
    environment:
      - "TRANSCRIBE_DEVICE=gpu"
      - "WHISPER_MODEL=large-v3"
      - "MODEL_PATH=/subgen/models"
    ports:
      - "9000:9000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu, utility, compute]
```

## Qwen3-ASR variant (Docker compose)

An alternative backend using [Qwen/Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) instead of faster-whisper. Requires an Nvidia GPU. Build from `Dockerfile.qwen3asr` / `docker-compose-qwen3asr.yml`.

```yaml
  subgen-qwen3asr:
    container_name: subgen-qwen3asr
    tty: true
    build:
      context: .
      dockerfile: Dockerfile.qwen3asr
    environment:
      - "WHISPER_MODEL=Qwen/Qwen3-ASR-1.7B"
      - "WEBHOOKPORT=9000"
      - "TRANSCRIBE_DEVICE=cuda"
      - "DEBUG=True"
      - "CLEAR_VRAM_ON_COMPLETE=True"
      - "APPEND=False"
      - "USE_FORCED_ALIGNER=True"
      - "FORCED_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B"
      - "MAX_NEW_TOKENS=1024"
      - "MAX_INFERENCE_BATCH_SIZE=1"
      - "DETECT_LANGUAGE_LENGTH=30"
      # - "FORCE_DETECTED_LANGUAGE_TO=ja"
      # - "HUGGINGFACE_TOKEN=hf_..."
    volumes:
      - "${APPDATA}/subgen/models:/subgen/models"
    ports:
      - "9000:9000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Qwen3-ASR variables

| Variable | Default | Description |
|---|---|---|
| WHISPER_MODEL | `Qwen/Qwen3-ASR-1.7B` | HuggingFace model ID. Also supports `Qwen/Qwen3-ASR-8B`. |
| TRANSCRIBE_DEVICE | `cuda` | `cuda` or `cpu`. |
| WEBHOOKPORT | `9000` | Port the HTTP server listens on. |
| DEBUG | `True` | Enable verbose debug logging. |
| HUGGINGFACE_TOKEN | `` | HuggingFace API token for gated/private models. |
| FORCE_DETECTED_LANGUAGE_TO | `` | Force a specific language (2-letter code), overriding auto-detection. |
| CLEAR_VRAM_ON_COMPLETE | `True` | Unload the model from VRAM after each transcription. |
| APPEND | `False` | Append a transcription credit line to the end of each subtitle. |
| USE_FORCED_ALIGNER | `True` | Use `Qwen3-ForcedAligner` for word-level timestamps. Disable to skip and use estimated timing. |
| FORCED_ALIGNER_MODEL | `Qwen/Qwen3-ForcedAligner-0.6B` | HuggingFace model ID for the forced aligner. |
| MAX_NEW_TOKENS | `1024` | Max tokens the decoder may generate per audio chunk. Increase if long sentences are cut off. |
| MAX_INFERENCE_BATCH_SIZE | `4` | Number of audio chunks processed in parallel. Reduce to lower VRAM usage. |
| DETECT_LANGUAGE_LENGTH | `30` | Seconds of audio used for language detection. |
| MAX_SEGMENT_CHARS | `40` | Max characters per subtitle card before forcing a line break. |
| MAX_SEGMENT_SEC | `7.0` | Hard cap (seconds) on a single subtitle's duration. |
| GAP_THRESHOLD_SEC | `0.5` | Silence gap (seconds) that starts a new subtitle segment. |

# What is this?

This is a stripped-down fork of [McCloudS/subgen](https://github.com/McCloudS/subgen) focused exclusively on the **Bazarr Whisper provider** use case. It exposes `/asr` and `/detect-language` endpoints compatible with the [whisper-asr-webservice](https://github.com/ahmetoner/whisper-asr-webservice) API that Bazarr uses. All Plex/Jellyfin/Emby/Tautulli webhook functionality has been removed. It uses stable-ts and faster-whisper (or a HuggingFace Transformers model) and supports both Nvidia GPUs and CPUs.

# Why?

Honestly, I built this for me, but saw the utility in other people maybe using it.  This works well for my use case.  Since having children, I'm either deaf or wanting to have everything quiet.  We watch EVERYTHING with subtitles now, and I feel like I can't even understand the show without them.  I use Bazarr to auto-download, and gap fill with Plex's built-in capability.  This is for everything else.  Some shows just won't have subtitles available for some reason or another, or in some cases on my H265 media, they are wildly out of sync. 

# What can it do?

* Serve as a Whisper provider for Bazarr via the `/asr` and `/detect-language` endpoints.
* Transcribe audio/video files to `.srt` subtitles with accurate timestamps via stable-ts.
* Use standard faster-whisper models or any HuggingFace Transformers Whisper-based model.

# How do I set it up?

## Install/Setup

### Standalone/Without Docker

Install Python 3 and ffmpeg, then install dependencies:
```
pip install numpy stable-ts-whisperless "stable-ts[hf]" huggingface_hub fastapi requests faster-whisper uvicorn python-multipart ffmpeg-python watchdog
```
Then run: `python3 subgen.py`

You will need the appropriate NVIDIA drivers if using GPU.


### Docker

The Dockerfile is in the repo along with an example docker-compose file.

`/subgen/models` is for storage of the language models. This isn't strictly necessary, but without it models will be re-downloaded on every image pull.

If you want to use a GPU, map it as shown in the docker-compose example above.

## Bazarr

Configure the Whisper Provider in Bazarr as shown below: <br>
![bazarr_configuration](https://wiki.bazarr.media/Additional-Configuration/images/whisper_config.png) <br>
The Docker Endpoint is the IP address and port of your subgen container (e.g. `http://192.168.1.111:9000`). See https://wiki.bazarr.media/Additional-Configuration/Whisper-Provider/ for more info.

## Variables

The following environment variables are available. They will default to the values listed below.
| Variable | Default | Description |
|---|---|---|
| TRANSCRIBE_DEVICE | `cpu` | `cpu`, `gpu`, or `cuda` |
| WHISPER_MODEL | `medium` | Standard model name (`tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3`, `large`, `distil-large-v2`, `distil-large-v3`, `distil-medium.en`, `distil-small.en`, etc.) or a HuggingFace model ID containing `/` (e.g. `litagin/anime-whisper`) which uses the Transformers backend. |
| CONCURRENT_TRANSCRIPTIONS | `2` | Number of files to transcribe in parallel. |
| WHISPER_THREADS | `4` | Number of CPU threads to use during computation (faster-whisper only). |
| MODEL_PATH | `./models` | Where model files are stored. |
| WEBHOOKPORT | `9000` | Port the HTTP server listens on. |
| WORD_LEVEL_HIGHLIGHT | `False` | Highlight each word as it is spoken in the subtitle. |
| DEBUG | `True` | Enable verbose debug logging. |
| FORCE_DETECTED_LANGUAGE_TO | `` | Force transcription to a specific language (2-letter code), overriding auto-detection. |
| CLEAR_VRAM_ON_COMPLETE | `True` | Unload the model from memory when the queue is idle. |
| MODEL_CLEANUP_DELAY | `30` | Seconds to wait after the queue empties before unloading the model. |
| COMPUTE_TYPE | `auto` | CTranslate2 compute type. See https://github.com/OpenNMT/CTranslate2/blob/master/docs/quantization.md (faster-whisper only). |
| APPEND | `False` | Append a transcription credit line to the end of each subtitle. |
| CUSTOM_REGROUP | `cm_sl=84_sl=42++++++1` | stable-ts regroup string. Set to blank to use the stable-ts default. |
| DETECT_LANGUAGE_LENGTH | `30` | Detect language on the first X seconds of the audio. |
| SUBGEN_KWARGS | `{}` | Python dict of extra kwargs passed to `model.transcribe()`. Example: `{'vad': 'True'}` |
| HUGGINGFACE_TOKEN | `` | HuggingFace API token for downloading gated or private models. Get one at https://huggingface.co/settings/tokens |


# What are the limitations/problems?

* It's using trained AI models to transcribe, so it WILL make mistakes.

# Audio Languages Supported (via OpenAI)

Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, and Welsh.

# Additional reading:

* https://github.com/openai/whisper (Original OpenAI project)
* https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes (2 letter subtitle codes)

# Credits:  
* Whisper.cpp (https://github.com/ggerganov/whisper.cpp) for original implementation
* Google
* ffmpeg
* https://github.com/jianfch/stable-ts
* https://github.com/guillaumekln/faster-whisper
* Whipser ASR Webservice (https://github.com/ahmetoner/whisper-asr-webservice) for how to implement Bazarr webhooks.

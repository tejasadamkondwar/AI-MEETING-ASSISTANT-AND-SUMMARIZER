"""End-to-end meeting pipeline.

    audio -> [1] Whisper speech-to-text        -> raw transcript
          -> [2] LLM #1: transcript refinement -> refined transcript
          -> [3] LLM #2: documentation         -> minutes, decisions, action items
          -> safety pass (code, not a model)   -> final record

Heavy dependencies (whisper, ollama) are imported lazily so the validation and
post-processing logic can be tested without them.
"""

import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from prompts import DOCUMENTER_PROMPT, REFINER_PROMPT

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".oga", ".opus",
    ".aac", ".wma", ".mp4", ".webm", ".mpeg", ".mpga",
}

DEFAULT_WHISPER_MODEL = "small"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

REFINE_CHUNK_WORDS = 450      # refine long transcripts piece by piece
MIN_WORDS_FOR_SPEECH = 3      # fewer words than this is treated as "no speech"
MISSING = "unspecified"
MISSING_VALUES = {"", "none", "unknown", "n/a", "na", "null", "not specified",
                  "unspecified", "not stated", "tbd", "-"}


class PipelineError(Exception):
    """An error whose message is safe and useful to show directly to the user."""


# --------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------
def validate_audio_file(audio_path):
    """Return a Path to a usable audio file or raise PipelineError."""
    if not audio_path:
        raise PipelineError("No audio file was provided. Upload or record a meeting first.")

    path = Path(str(audio_path)).expanduser()

    if not path.is_file():
        raise PipelineError(f"The audio file could not be found: {path.name}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS))
        raise PipelineError(
            f"Unsupported file type '{path.suffix or 'none'}'. "
            f"Please upload one of: {allowed}."
        )

    if path.stat().st_size == 0:
        raise PipelineError("The audio file is empty (0 bytes). Please upload a valid recording.")

    if shutil.which("ffmpeg") is None:
        raise PipelineError(
            "ffmpeg is not installed or not on PATH. Whisper needs it to read audio. "
            "Install it (see README) and restart the app."
        )

    return path


# --------------------------------------------------------------------------
# Stage 1: speech-to-text
# --------------------------------------------------------------------------
@lru_cache(maxsize=2)
def load_whisper_model(model_name):
    try:
        import whisper
    except ImportError as error:
        raise PipelineError(
            "The 'openai-whisper' package is not installed. Run: pip install -r requirements.txt"
        ) from error
    return whisper.load_model(model_name, device="cpu")


def transcribe_audio(path, model_name, domain_hints=""):
    model = load_whisper_model(model_name)
    try:
        result = model.transcribe(
            str(path),
            task="transcribe",
            language="en",
            initial_prompt=domain_hints or None,
            fp16=False,
            verbose=False,
        )
    except PipelineError:
        raise
    except Exception as error:
        raise PipelineError(
            "The file could not be read as audio. It may be corrupted or not a real "
            f"audio recording. (Details: {str(error).strip()[:200]})"
        ) from error

    text = (result.get("text") or "").strip()
    if len(text.split()) < MIN_WORDS_FOR_SPEECH:
        raise PipelineError(
            "No speech was detected in the recording. Check that the file contains "
            "audible English speech."
        )
    return text


# --------------------------------------------------------------------------
# Language-model helpers (Ollama)
# --------------------------------------------------------------------------
def get_client():
    try:
        from ollama import Client
    except ImportError as error:
        raise PipelineError(
            "The 'ollama' package is not installed. Run: pip install -r requirements.txt"
        ) from error
    return Client(host=OLLAMA_URL, timeout=900)


def check_model_available(client, model):
    try:
        client.show(model)
    except ConnectionError as error:
        raise PipelineError(
            "Cannot connect to Ollama. Open the Ollama app or run 'ollama serve', then try again."
        ) from error
    except Exception as error:  # ollama.ResponseError and friends
        if "connect" in str(error).lower():
            raise PipelineError(
                "Cannot connect to Ollama. Open the Ollama app or run 'ollama serve', then try again."
            ) from error
        raise PipelineError(
            f"The language model '{model}' is not available. Run: ollama pull {model}"
        ) from error


def estimate_tokens(text):
    return int(len(text.split()) * 1.6) + 50


def context_size(*texts, reply_budget=1500):
    """Pick an Ollama context window large enough for the prompt and the reply."""
    needed = sum(estimate_tokens(t) for t in texts) + reply_budget + 300
    return min(32768, max(8192, ((needed + 1023) // 1024) * 1024))


def generate_text(client, model, system_prompt, user_prompt, json_output=False, num_ctx=8192):
    arguments = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.1, "num_ctx": num_ctx},
    }
    if json_output:
        arguments["format"] = "json"

    try:
        response = client.chat(**arguments)
    except ConnectionError as error:
        raise PipelineError(
            "Lost connection to Ollama while generating. Make sure it is still running."
        ) from error
    except Exception as error:
        raise PipelineError(f"The language model '{model}' failed: {str(error).strip()[:200]}") from error

    text = response.message.content
    if not text or not text.strip():
        raise PipelineError(f"The language model '{model}' returned an empty response.")
    return text.strip()


# --------------------------------------------------------------------------
# Stage 2: transcript refinement (language model #1)
# --------------------------------------------------------------------------
def chunk_text(text, max_words=REFINE_CHUNK_WORDS):
    """Split text into chunks of at most ~max_words, breaking on sentence ends."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks, current, count = [], [], 0
    for sentence in sentences:
        words = sentence.split()
        if not words:
            continue
        if len(words) > max_words:  # one huge unpunctuated "sentence"
            if current:
                chunks.append(" ".join(current))
                current, count = [], 0
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
            continue
        if count + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks


def refine_transcript(client, model, raw_transcript, domain_hints, warnings):
    system_prompt = REFINER_PROMPT.format(domain_hints=domain_hints.strip() or "None provided")
    refined_parts = []

    for index, chunk in enumerate(chunk_text(raw_transcript), start=1):
        reply = generate_text(
            client, model, system_prompt, f"Raw transcript:\n{chunk}",
            num_ctx=context_size(system_prompt, chunk, reply_budget=estimate_tokens(chunk)),
        )
        # Guard: a refiner must not summarise or pad. If the length changed a lot,
        # keep the original text for this chunk instead of trusting the model.
        before, after = len(chunk.split()), len(reply.split())
        if before >= 8 and not (0.8 <= after / before <= 1.25):
            warnings.append(
                f"Refinement of transcript part {index} changed its length too much "
                f"({before} -> {after} words); the original wording was kept for that part."
            )
            reply = chunk
        refined_parts.append(reply)

    return " ".join(refined_parts)


# --------------------------------------------------------------------------
# Stage 3: documentation (language model #2)
# --------------------------------------------------------------------------
def parse_json_object(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def generate_documentation(client, model, refined_transcript):
    user_prompt = f"Transcript:\n{refined_transcript}"
    num_ctx = context_size(DOCUMENTER_PROMPT, user_prompt, reply_budget=2500)

    last_error = None
    for _ in range(2):  # one retry if the model returns broken JSON
        reply = generate_text(client, model, DOCUMENTER_PROMPT, user_prompt,
                              json_output=True, num_ctx=num_ctx)
        try:
            data = parse_json_object(reply)
            if isinstance(data, dict):
                return data
            last_error = "the JSON was not an object"
        except json.JSONDecodeError as error:
            last_error = str(error)
    raise PipelineError(f"The documentation model did not return valid JSON ({last_error}).")


# --------------------------------------------------------------------------
# Safety pass (plain code, not a model)
# --------------------------------------------------------------------------
def _squash(value):
    """Lower-case and keep only letters, digits and single spaces."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(value or "").casefold()).split())


def _is_missing(value):
    return str(value or "").strip().casefold() in MISSING_VALUES


def _names_in_text(owner, transcript_words):
    """True if every person named in `owner` appears somewhere in the transcript."""
    parts = [p for p in re.split(r",|&|/|\band\b", owner, flags=re.I) if p.strip()]
    for part in parts:
        tokens = [t for t in _squash(part).split() if len(t) > 1]
        if not tokens or not any(t in transcript_words for t in tokens):
            return False
    return True


def _deadline_in_text(deadline, transcript_squashed):
    tokens = _squash(deadline).split()
    if not tokens:
        return False
    words = set(transcript_squashed.split())
    found = sum(1 for t in tokens if t in words)
    return found / len(tokens) >= 0.6


def normalize_minutes(minutes):
    """Accept strings or {topic, points} objects and return {topic, points} objects."""
    result = []
    for entry in minutes if isinstance(minutes, list) else []:
        if isinstance(entry, str) and entry.strip():
            result.append({"topic": "General", "points": [entry.strip()]})
        elif isinstance(entry, dict):
            points = entry.get("points", [])
            if isinstance(points, str):
                points = [points]
            points = [str(p).strip() for p in points if str(p).strip()]
            if points:
                result.append({"topic": str(entry.get("topic") or "General").strip(), "points": points})
    return result


def enforce_safety_and_verification(data, *transcripts):
    """Validate the model's JSON and remove anything not supported by the transcript.

    - Missing owners/deadlines become "unspecified".
    - An owner whose name never appears in the transcript is replaced by "unspecified".
    - A deadline whose words do not appear in the transcript is replaced by "unspecified".
    - Every quote is text-matched against the transcripts and flagged if not found.
    A text match shows the quote exists; it does not prove the model read it correctly.
    """
    if not isinstance(data, dict):
        raise PipelineError("The documentation model did not return a JSON object.")

    squashed = _squash(" ".join(t for t in transcripts if t))
    words = set(squashed.split())

    clean = {
        "summary": str(data.get("summary") or "").strip(),
        "minutes": normalize_minutes(data.get("minutes")),
        "decisions": [],
        "action_items": [],
    }

    seen = set()
    for item in data.get("decisions") or []:
        if not isinstance(item, dict) or not str(item.get("decision") or "").strip():
            continue
        quote = str(item.get("quote") or "").strip()
        clean["decisions"].append({
            "decision": str(item["decision"]).strip(),
            "quote": quote,
            "verified_in_transcript": bool(_squash(quote) and _squash(quote) in squashed),
        })

    for item in data.get("action_items") or []:
        if not isinstance(item, dict) or not str(item.get("task") or "").strip():
            continue
        task = str(item["task"]).strip()
        if _squash(task) in seen:
            continue
        seen.add(_squash(task))

        owner = MISSING if _is_missing(item.get("owner")) else str(item["owner"]).strip()
        deadline = MISSING if _is_missing(item.get("deadline")) else str(item["deadline"]).strip()
        notes = []

        if owner != MISSING and not _names_in_text(owner, words):
            notes.append(f"Owner '{owner}' was not found in the transcript and was removed.")
            owner = MISSING
        if deadline != MISSING and not _deadline_in_text(deadline, squashed):
            notes.append(f"Deadline '{deadline}' was not found in the transcript and was removed.")
            deadline = MISSING

        quote = str(item.get("quote") or "").strip()
        entry = {
            "task": task,
            "owner": owner,
            "deadline": deadline,
            "quote": quote,
            "verified_in_transcript": bool(_squash(quote) and _squash(quote) in squashed),
        }
        if notes:
            entry["review_note"] = " ".join(notes)
        clean["action_items"].append(entry)

    return clean


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_pipeline(
    audio_path,
    domain_hints="",
    whisper_model_name=None,
    refiner_model=None,
    documenter_model=None,
    progress=None,
):
    """Run all three stages. Returns (raw_transcript, refined_transcript, record).

    `progress(fraction, description)` is called as each stage starts and finishes.
    Raises PipelineError with a user-readable message on any failure.
    """
    def report(fraction, text):
        if progress:
            progress(fraction, text)

    domain_hints = (domain_hints or "").strip()
    whisper_model_name = whisper_model_name or os.environ.get("WHISPER_MODEL") or DEFAULT_WHISPER_MODEL
    base_model = os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
    refiner_model = refiner_model or os.environ.get("OLLAMA_REFINER_MODEL") or base_model
    documenter_model = documenter_model or os.environ.get("OLLAMA_DOCUMENTER_MODEL") or base_model
    warnings, timings = [], {}

    report(0.02, "Checking the audio file...")
    path = validate_audio_file(audio_path)

    report(0.05, "Checking that the language models are available...")
    client = get_client()
    for model in dict.fromkeys([refiner_model, documenter_model]):
        check_model_available(client, model)

    start = time.time()
    report(0.10, f"Stage 1/3: Transcribing audio with Whisper ({whisper_model_name})... this can take a few minutes")
    raw_transcript = transcribe_audio(path, whisper_model_name, domain_hints)
    timings["transcription_s"] = round(time.time() - start, 1)

    start = time.time()
    report(0.45, f"Stage 2/3: Refining the transcript with {refiner_model}...")
    refined_transcript = refine_transcript(client, refiner_model, raw_transcript, domain_hints, warnings)
    timings["refinement_s"] = round(time.time() - start, 1)

    start = time.time()
    report(0.70, f"Stage 3/3: Writing minutes, decisions and action items with {documenter_model}...")
    record = generate_documentation(client, documenter_model, refined_transcript)
    record = enforce_safety_and_verification(record, refined_transcript, raw_transcript)
    timings["documentation_s"] = round(time.time() - start, 1)

    record["meta"] = {
        "speech_to_text_model": f"openai-whisper:{whisper_model_name}",
        "refinement_model": refiner_model,
        "documentation_model": documenter_model,
        "domain_hints": domain_hints or None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timings": timings,
        "warnings": warnings,
    }
    report(0.95, "Preparing outputs...")
    return raw_transcript, refined_transcript, record

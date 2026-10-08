# AI Meeting Assistant

Turns a recorded English meeting into a raw transcript, a domain-corrected
transcript, and a structured meeting record (summary, minutes, key decisions,
action items). Built for Inter IIT Tech Meet 15.0 Bootcamp, Phase 2 (ML PS).

Everything runs **locally**: no API keys, and no audio or text leaves the machine.

## Pipeline

| Stage | Role | Model (default) | Where |
|-------|------|-----------------|-------|
| 1 | Speech-to-text | OpenAI Whisper `small` (CPU) | `pipeline.transcribe_audio` |
| 2 | **Language model #1**: transcript refinement | Ollama `llama3.2:3b` | `REFINER_PROMPT` in `prompts.py` |
| 3 | **Language model #2**: minutes, decisions, action items | Ollama `llama3.2:3b` | `DOCUMENTER_PROMPT` in `prompts.py` |

The two language-model roles are separate stages with separate prompts and
separate calls. Each can use a different model (see Configuration).

```
audio -> Whisper -> raw transcript -> LLM #1 -> refined transcript -> LLM #2 -> JSON record
                                                                              -> safety pass -> outputs
```

Stage 2 receives the raw transcript (in chunks of about 450 words). Stage 3
receives the refined transcript. A final safety pass written in plain Python
then validates the result (see "Safeguards").

## Project layout

```
app.py            Gradio interface: upload, status, results, downloads
pipeline.py       Orchestration, validation, model calls, safety pass
prompts.py        Prompts for the two language-model stages
utilities.py      Word-level diff view, Markdown report builder, file writer
requirements.txt  Python dependencies
tests/            Offline checks (python tests/run_checks.py)
example/          Put the shareable sample recording here (sample_meeting.mp3)
outputs/          Created on first run; holds the latest generated files
```

## Setup

Requirements: Python 3.10+, [ffmpeg](https://ffmpeg.org/) on PATH,
[Ollama](https://ollama.com/).

```bash
# 1. ffmpeg (pick one)
brew install ffmpeg            # macOS
sudo apt install ffmpeg        # Ubuntu/Debian
winget install ffmpeg          # Windows

# 2. Python environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Local language model
ollama pull llama3.2:3b
ollama serve                   # not needed if the Ollama app is already running
```

The first run downloads the Whisper weights (`small` is about 460 MB).

## Run

```bash
python app.py
```

Open the printed URL (default http://127.0.0.1:7860), then:

1. Upload an English audio file, or record from the microphone.
2. Optionally enter domain terms (e.g. `CRISPR, PCR, Dr. Rao`). They bias
   Whisper and guide the refinement stage.
3. Click **Process Meeting**. The status bar shows the current stage
   (Stage 1/3, 2/3, 3/3). On CPU a 10-minute recording can take several minutes.
4. Review the **Meeting Record**, **Transcripts**, **Tracked Changes (Diff)** and
   **JSON** tabs.
5. Download the JSON record, Markdown record, raw transcript and refined transcript.

## Outputs

Each run writes four files to `outputs/` (overwritten on the next run):

| File | Content | Format |
|------|---------|--------|
| `raw_transcript.txt` | Whisper output | text |
| `refined_transcript.txt` | After domain-term correction | text |
| `meeting_notes.md` | Summary, minutes, decisions, action items | human-readable |
| `meeting_notes.json` | Same record | machine-readable |

Both record formats are built from the same data, so decisions and tasks always
match. JSON structure:

```json
{
  "summary": "string",
  "minutes": [{"topic": "string", "points": ["string"]}],
  "decisions": [{"decision": "string", "quote": "string", "verified_in_transcript": true}],
  "action_items": [
    {"task": "string", "owner": "name or unspecified", "deadline": "date or unspecified",
     "quote": "string", "verified_in_transcript": true, "review_note": "optional"}
  ],
  "meta": {"speech_to_text_model": "...", "refinement_model": "...",
           "documentation_model": "...", "timings": {}, "warnings": []}
}
```

## Configuration (environment variables)

| Variable | Meaning | Default |
|----------|---------|---------|
| `WHISPER_MODEL` | `tiny`, `base`, `small`, `medium`, ... | `small` |
| `OLLAMA_MODEL` | Model for both language-model stages | `llama3.2:3b` |
| `OLLAMA_REFINER_MODEL` | Model for stage 2 only | `OLLAMA_MODEL` |
| `OLLAMA_DOCUMENTER_MODEL` | Model for stage 3 only | `OLLAMA_MODEL` |
| `OLLAMA_URL` | Ollama server address | `http://localhost:11434` |

Example using two different models:

```bash
OLLAMA_REFINER_MODEL=llama3.2:3b OLLAMA_DOCUMENTER_MODEL=llama3.1:8b python app.py
```

Larger models give noticeably better minutes and task extraction.

## Error handling

Every failure appears in the Status box and as a notification, with a
plain-language reason:

| Situation | Message (summary) |
|-----------|-------------------|
| Nothing uploaded | No audio file was provided |
| Wrong file type (e.g. `.txt`, `.pdf`) | Unsupported file type, with the list of accepted types |
| Empty file (0 bytes) | The audio file is empty |
| Corrupted or non-audio data | The file could not be read as audio |
| Silent recording | No speech was detected |
| ffmpeg missing | ffmpeg is not installed, with install hint |
| Ollama not running / model missing | Checked **before** transcription; shows `ollama serve` / `ollama pull` |
| Model returns broken JSON | One automatic retry, then a clear error |

Accepted types: mp3, wav, m4a, flac, ogg, oga, opus, aac, wma, mp4, webm, mpeg, mpga.

## Safeguards against invented content

- Prompts forbid inventing facts, owners or deadlines; proposals are not
  decisions; "we should" is not a task; "I'll do it" with no clear speaker has
  no owner.
- The refiner may not summarise or reorder. If a refined chunk changes length by
  more than about 20-25%, the original wording of that chunk is kept and a
  warning is shown.
- Code checks the model's JSON: an owner whose name never appears in the
  transcript, or a deadline whose words do not appear in it, is replaced by
  `unspecified` and annotated with a `review_note`.
- Missing owners and deadlines are always shown as `unspecified`.
- Every decision and task carries a verbatim quote that is text-matched against
  the transcripts and flagged if not found. A match proves the quote exists, not
  that it was interpreted correctly.
- Transcripts are treated as data, not instructions (limits prompt injection
  from spoken content). Temperature is 0.1 for both language-model stages.

## Tests

```bash
python tests/run_checks.py
```

Runs 34 offline checks (input validation, error mapping, chunking, safety pass,
stage order with fake models, output files). It does not need Whisper or Ollama.

## Known limitations

- No speaker diarisation, so first-person commitments are only given an owner
  when the transcript names that person elsewhere.
- Small local models may produce plain minutes and can miss subtle decisions.
  Use a larger `OLLAMA_MODEL`.
- Whisper is run in English-only mode and on CPU; use `WHISPER_MODEL=medium`
  for higher accuracy if time allows.
- Outputs go to a fixed `outputs/` folder; the app processes one recording at
  a time.

## Sample recording and results

Put a shareable recording at `example/sample_meeting.mp3` (the **Load Sample
Audio** button reads it), process it, and copy the four files from `outputs/`
into `example/` alongside it.

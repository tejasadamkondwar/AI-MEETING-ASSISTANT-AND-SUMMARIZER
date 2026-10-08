# Demo script and submission checklist

## A. Record the demonstration video (about 3-4 minutes)

Use a recording that is NOT your sample, ideally 2-5 minutes, so the run is quick.

1. **Intro (15 s).** Say the three models: Whisper for speech-to-text, language
   model 1 for transcript refinement, language model 2 for minutes, decisions and tasks.
2. **Show the error handling (30 s).** Upload a `.txt` file, click Process, and show
   the "Unsupported file type" message. Click Process with nothing uploaded and show
   that message too.
3. **Upload the meeting audio (10 s).** Type 3-5 domain terms from the meeting
   into the hints box. Click **Process Meeting**.
4. **Narrate the status bar (30 s).** Point out Stage 1/3, 2/3 and 3/3 as they change.
5. **Transcripts tab (30 s).** Show the raw transcript, then the refined one.
6. **Tracked Changes tab (30 s).** Show 2-3 corrected terms in red/green and say
   that names, numbers and negations were not changed.
7. **Meeting Record tab (60 s).** Walk through summary, minutes by topic, decisions
   (each with an evidence quote) and action items. Point at one task whose owner or
   deadline shows `unspecified` and explain that the app never guesses.
8. **JSON tab and downloads (30 s).** Show the JSON, then download all four files
   and open one.

Keep the screen recording at 1080p, with your voice on. Do not edit out the wait
during processing; speed it up instead.

## B. Rubric check (100 points)

| Criterion | Points | Where it is covered |
|-----------|--------|---------------------|
| Speech transcription | 20 | Whisper `small`, domain terms as `initial_prompt`, English mode |
| Transcript refinement | 20 | `REFINER_PROMPT`, chunking, length guard, diff view |
| Minutes and decisions | 25 | `DOCUMENTER_PROMPT` (topics, agreement-only decisions), evidence quotes |
| Action items | 15 | Owner/deadline only if stated; code removes names/dates not in transcript |
| End-to-end application | 15 | Gradio app, one workflow, status per stage, clear errors, downloads |
| Submission quality | 5 | README, technical description, tests, sample files, demo video |

## C. Deliverables checklist

- [ ] Source archive: zip this folder (without `.venv`, `__pycache__`, `outputs/`)
- [ ] `requirements.txt` and `README.md` (included)
- [ ] Prompts: `prompts.py` (included)
- [ ] Short technical description (the "AI Meeting Assistant: Technical Description" document)
- [ ] Shareable sample recording at `example/sample_meeting.mp3`
- [ ] The four generated files for that recording copied into `example/`
- [ ] Demonstration video of an end-to-end run
- [ ] You have processed one recording you did not use while developing, to confirm
      the pipeline works on new audio
- [ ] Run `python tests/run_checks.py` once and confirm it prints "All 34 checks passed."

"""Offline checks for the pipeline logic. No Whisper, Ollama or network needed.

Run from the project folder:  python tests/run_checks.py
"""
import json
import os
import sys
import tempfile
import types
import wave

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pipeline
from pipeline import PipelineError
from utilities import build_markdown_report, generate_diff_html, save_outputs

passed = 0


def check(name, condition):
    global passed
    assert condition, f"FAILED: {name}"
    passed += 1
    print(f"  ok  {name}")


def raises(fn, *args, contains=""):
    try:
        fn(*args)
    except PipelineError as error:
        return contains.lower() in str(error).lower()
    return False


tmp = tempfile.mkdtemp()

# ---- 1. input validation -------------------------------------------------
print("Input validation")
empty_mp3 = os.path.join(tmp, "empty.mp3"); open(empty_mp3, "wb").close()
text_file = os.path.join(tmp, "notes.txt"); open(text_file, "w").write("hello")
good_wav = os.path.join(tmp, "ok.wav")
with wave.open(good_wav, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(b"\0\0" * 1600)

check("no file", raises(pipeline.validate_audio_file, None, contains="no audio file"))
check("missing file", raises(pipeline.validate_audio_file, os.path.join(tmp, "x.mp3"), contains="could not be found"))
check("unsupported type", raises(pipeline.validate_audio_file, text_file, contains="unsupported file type"))
check("empty file", raises(pipeline.validate_audio_file, empty_mp3, contains="empty"))
check("valid wav accepted", str(pipeline.validate_audio_file(good_wav)).endswith("ok.wav"))

# ---- 2. transcription error mapping -------------------------------------
print("Transcription errors")


class BadModel:
    def transcribe(self, *a, **k):
        raise RuntimeError("Failed to load audio: invalid data")


class SilentModel:
    def transcribe(self, *a, **k):
        return {"text": "  "}


pipeline.load_whisper_model = lambda name: BadModel()
check("unreadable audio", raises(pipeline.transcribe_audio, good_wav, "small", contains="could not be read as audio"))
pipeline.load_whisper_model = lambda name: SilentModel()
check("silent audio", raises(pipeline.transcribe_audio, good_wav, "small", contains="no speech"))

# ---- 3. chunking and JSON parsing ---------------------------------------
print("Chunking and JSON")
long_text = " ".join(f"Sentence number {i} is here." for i in range(200))
chunks = pipeline.chunk_text(long_text, 100)
check("chunks respect size", all(len(c.split()) <= 100 for c in chunks))
check("chunks lose no words", " ".join(chunks).split() == long_text.split())
check("fenced JSON parsed", pipeline.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1})
check("JSON with preamble parsed", pipeline.parse_json_object('Here: {"a": 2} done') == {"a": 2})

# ---- 4. safety pass ------------------------------------------------------
print("Safety pass")
transcript = ("We agree to host the panel on Friday, October 16th. Alex will push the "
              "deck outline by Thursday, October 8th. I can handle the venue.")
model_output = {
    "summary": "S",
    "minutes": ["old style string point", {"topic": "Event", "points": ["Date fixed"]}],
    "decisions": [{"decision": "Host panel Oct 16", "quote": "we agree to host the panel on friday october 16th"}],
    "action_items": [
        {"task": "Push deck outline", "owner": "Alex", "deadline": "Thursday, October 8th",
         "quote": "Alex will push the deck outline by Thursday, October 8th."},
        {"task": "Handle venue", "owner": "Sarah", "deadline": "2026-10-12",
         "quote": "I can handle the venue."},
        {"task": "Handle venue", "owner": "N/A", "deadline": "none", "quote": "dup"},
        {"task": "Book caterer", "owner": "n/a", "deadline": "", "quote": "made up quote"},
    ],
}
out = pipeline.enforce_safety_and_verification(model_output, transcript)
a = {x["task"]: x for x in out["action_items"]}
check("real owner and deadline kept", a["Push deck outline"]["owner"] == "Alex"
      and a["Push deck outline"]["deadline"] == "Thursday, October 8th")
check("invented owner removed", a["Handle venue"]["owner"] == "unspecified")
check("invented deadline removed", a["Handle venue"]["deadline"] == "unspecified")
check("removal is explained", "review_note" in a["Handle venue"])
check("duplicate task dropped", len(out["action_items"]) == 3)
check("missing values -> unspecified", a["Book caterer"]["owner"] == "unspecified"
      and a["Book caterer"]["deadline"] == "unspecified")
check("good quote verified", out["decisions"][0]["verified_in_transcript"] is True)
check("fake quote flagged", a["Book caterer"]["verified_in_transcript"] is False)
check("minutes normalised", out["minutes"][0] == {"topic": "General", "points": ["old style string point"]}
      and out["minutes"][1]["topic"] == "Event")
check("non-object rejected", raises(pipeline.enforce_safety_and_verification, ["x"], "t", contains="json object"))

# ---- 5. full pipeline with fake models ----------------------------------
print("End-to-end workflow (fake models)")
RAW = ("Welcome everyone. We use cry sper for editing and pee see are for testing. "
       "We agree to host the panel on Friday, October 16th. Alex will push the "
       "deck outline by Thursday, October 8th. I can handle the venue and the stream.")
calls = []


class FakeClient:
    def chat(self, **kwargs):
        system = kwargs["messages"][0]["content"]
        user = kwargs["messages"][1]["content"]
        if "transcript editor" in system:
            calls.append("refine")
            body = user.split("Raw transcript:\n", 1)[1]
            if os.environ.get("FAKE_BAD_REFINER"):
                body = "Short summary."
            else:
                body = body.replace("cry sper", "CRISPR").replace("pee see are", "PCR")
            return types.SimpleNamespace(message=types.SimpleNamespace(content=body))
        calls.append("document")
        data = {
            "summary": "Planning meeting.",
            "minutes": [{"topic": "Event", "points": ["Panel date agreed."]}],
            "decisions": [{"decision": "Host the panel on 16 October",
                           "quote": "We agree to host the panel on Friday, October 16th."}],
            "action_items": [
                {"task": "Push the deck outline", "owner": "Alex",
                 "deadline": "Thursday, October 8th",
                 "quote": "Alex will push the deck outline by Thursday, October 8th."},
                {"task": "Handle the venue and stream", "owner": "Priya", "deadline": "unspecified",
                 "quote": "I can handle the venue and the stream."},
            ],
        }
        return types.SimpleNamespace(message=types.SimpleNamespace(content=json.dumps(data)))


pipeline.transcribe_audio = lambda path, model, hints="": (calls.append("transcribe") or RAW)
pipeline.get_client = lambda: FakeClient()
pipeline.check_model_available = lambda client, model: None
steps = []
raw, refined, record = pipeline.run_pipeline(good_wav, "CRISPR, PCR",
                                             progress=lambda f, t: steps.append((f, t)))
check("stages run in order", calls == ["transcribe", "refine", "document"])
check("raw transcript kept", raw == RAW)
check("refined differs from raw", "CRISPR" in refined and "PCR" in refined and refined != raw)
check("progress is monotonic", [s[0] for s in steps] == sorted(s[0] for s in steps))
check("progress names all 3 stages", all(any(f"Stage {n}/3" in t for _, t in steps) for n in (1, 2, 3)))
owners = {x["task"]: x["owner"] for x in record["action_items"]}
check("hallucinated owner removed", owners["Handle the venue and stream"] == "unspecified")
check("stated owner kept", owners["Push the deck outline"] == "Alex")
check("models identified in record", record["meta"]["speech_to_text_model"].startswith("openai-whisper")
      and record["meta"]["refinement_model"] and record["meta"]["documentation_model"])

os.environ["FAKE_BAD_REFINER"] = "1"
raw, refined, record = pipeline.run_pipeline(good_wav, "")
check("bad refiner output is rejected", refined == RAW and len(record["meta"]["warnings"]) == 1)
del os.environ["FAKE_BAD_REFINER"]

# ---- 6. outputs ----------------------------------------------------------
print("Outputs")
md = build_markdown_report(record)
paths = save_outputs(raw, refined, record, md, os.path.join(tmp, "out"))
check("all four files written", all(os.path.getsize(p) > 0 for p in paths.values()))
saved = json.load(open(paths["json"], encoding="utf-8"))
check("JSON and Markdown agree on tasks",
      all(t["task"] in md for t in saved["action_items"]) and all(d["decision"] in md for d in saved["decisions"]))
check("unspecified shown in Markdown", "Owner: unspecified" in md and "Deadline: unspecified" in md)
check("diff shows correction", "<ins" in generate_diff_html(RAW, RAW.replace("cry sper", "CRISPR")))

print(f"\nAll {passed} checks passed.")

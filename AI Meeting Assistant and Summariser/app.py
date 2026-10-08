import json
import os

import gradio as gr

from pipeline import PipelineError, run_pipeline
from utilities import build_markdown_report, generate_diff_html, save_outputs

SAMPLE_PATH = os.path.join("example", "sample_meeting.mp3")
OUTPUT_FOLDER = "outputs"

custom_css = """
.gradio-container { font-weight: 600 !important; }
textarea { font-weight: 600 !important; }
"""


def failure(message):
    """Return values for every output component when processing fails."""
    gr.Warning(message)
    return ("", "", "<p>No comparison available.</p>", f"**Processing failed:** {message}",
            "", None, None, None, None, f"FAILED: {message}")


def process_audio(audio_file, domain_hints, progress=gr.Progress()):
    if not audio_file:
        return failure("No audio file was provided. Upload or record a meeting first.")

    try:
        raw_text, refined_text, record = run_pipeline(
            audio_file,
            domain_hints,
            progress=lambda fraction, text: progress(fraction, desc=text),
        )
        markdown_doc = build_markdown_report(record)
        paths = save_outputs(raw_text, refined_text, record, markdown_doc, OUTPUT_FOLDER)
    except PipelineError as error:
        return failure(str(error))
    except Exception as error:  # anything unexpected still ends in a readable message
        return failure(f"Unexpected error: {error}")

    for warning in record["meta"]["warnings"]:
        gr.Warning(warning)

    status = (
        f"Completed. {len(record['decisions'])} decision(s), {len(record['action_items'])} action item(s). "
        f"Models: {record['meta']['speech_to_text_model']} -> "
        f"{record['meta']['refinement_model']} -> {record['meta']['documentation_model']}."
    )
    return (
        raw_text,
        refined_text,
        generate_diff_html(raw_text, refined_text),
        markdown_doc,
        json.dumps(record, indent=2, ensure_ascii=False),
        paths["json"],
        paths["markdown"],
        paths["raw"],
        paths["refined"],
        status,
    )


def load_example():
    if not os.path.exists(SAMPLE_PATH):
        raise gr.Error(f"Sample file not found. Place a recording at {SAMPLE_PATH}.")
    return gr.update(value=SAMPLE_PATH), "Biotech, CRISPR, PCR, Dr. Rao"


with gr.Blocks(title="AI Meeting Assistant", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown("# AI Meeting Assistant & Documenter")
    gr.Markdown(
        "Upload an English meeting recording. The app transcribes it (Whisper), corrects "
        "domain-specific terms (language model 1), then writes minutes, decisions and "
        "action items (language model 2)."
    )

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath",
                                   label="Upload Audio Recording")
            hints_input = gr.Textbox(
                label="Domain Terms & Keywords (Optional)",
                placeholder="e.g. CRISPR, PCR, Dr. Rao, Inter-IIT",
                lines=2,
            )
            process_btn = gr.Button("Process Meeting", variant="primary")
            example_btn = gr.Button("Load Sample Audio")
            status_box = gr.Textbox(label="Status", value="Ready", interactive=False, lines=3)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Meeting Record"):
                    md_display = gr.Markdown("Outputs will appear here after processing.")
                with gr.TabItem("Transcripts"):
                    raw_display = gr.Textbox(label="Raw Transcript (Whisper)", lines=8)
                    refined_display = gr.Textbox(label="Refined Transcript (language model 1)", lines=8)
                with gr.TabItem("Tracked Changes (Diff)"):
                    diff_display = gr.HTML("<p>Red = removed from the raw transcript, green = added by refinement.</p>")
                with gr.TabItem("JSON"):
                    json_display = gr.Code(label="Machine-Readable Record", language="json")

            with gr.Row():
                json_download = gr.File(label="Download JSON record")
                md_download = gr.File(label="Download Markdown record")
            with gr.Row():
                raw_download = gr.File(label="Download Raw Transcript")
                refined_download = gr.File(label="Download Refined Transcript")

    process_btn.click(
        fn=process_audio,
        inputs=[audio_input, hints_input],
        outputs=[raw_display, refined_display, diff_display, md_display, json_display,
                 json_download, md_download, raw_download, refined_download, status_box],
        api_name=False,
    )
    example_btn.click(fn=load_example, outputs=[audio_input, hints_input], api_name=False)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()

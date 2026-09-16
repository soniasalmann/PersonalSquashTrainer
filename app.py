"""
Squash Personal Trainer — Interactive Web Application
Built with Gradio for local execution and Hugging Face Spaces deployment.

Supports two user modes:
1. "Sample Video (1-Click Demo)": Instantly analyze the pre-loaded practice session.
2. "Upload Your Own Video": Upload any squash drill footage to analyze.
"""

import os
import sys
import subprocess
import tempfile
import gradio as gr

# Ensure local project path is in sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

SAMPLE_VIDEO_PATH = os.path.join(PROJECT_DIR, "input_videos", "practice.mp4")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_DIR, "court_config.json")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def process_squash_video(
    video_path,
    skip_frames=2,
    scale=0.5,
    gemini_key=None,
    progress=gr.Progress(track_tqdm=True)
):
    """
    Executes the Squash Trainer pipeline and returns:
    (processed_video_path, html_report_content, markdown_report_content, status_message)
    """
    if not video_path or not os.path.exists(video_path):
        return None, "<h3>Error: No valid video file provided.</h3>", "", "❌ Please select or upload a video file."

    progress(0.05, desc="Initializing Squash Trainer Pipeline...")

    # Unique output naming
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    out_video = os.path.join(OUTPUT_DIR, f"web_{base_name}_processed.mp4")
    out_html = os.path.join(OUTPUT_DIR, f"web_{base_name}_processed_report.html")
    out_md = os.path.join(OUTPUT_DIR, f"web_{base_name}_processed_report.md")

    # Command execution
    cmd = [
        sys.executable,
        os.path.join(PROJECT_DIR, "squash_main.py"),
        video_path,
        "--config", DEFAULT_CONFIG_PATH,
        "--output", out_video,
        "--no-preview",
        "--skip-frames", str(int(skip_frames)),
        "--scale", str(float(scale)),
        "--export-preview",
        "--preview-length", "10"
    ]

    if gemini_key and gemini_key.strip():
        cmd.extend(["--gemini-key", gemini_key.strip()])

    progress(0.2, desc="Running Pass 1: YOLOv8 Pose Tracking & Kalman Ball Estimation...")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_DIR)
        if proc.returncode != 0:
            err_msg = proc.stderr[-800:] if proc.stderr else proc.stdout[-800:]
            return None, f"<h3>Analysis Failed</h3><pre>{err_msg}</pre>", "", f"❌ Error during video processing: {err_msg}"
    except Exception as e:
        return None, f"<h3>Error: {str(e)}</h3>", "", f"❌ Execution error: {str(e)}"

    progress(0.9, desc="Compiling AI Coach Insights & Dashboards...")

    # Read generated reports
    html_content = "<p>Report could not be generated.</p>"
    if os.path.exists(out_html):
        with open(out_html, "r", encoding="utf-8") as f:
            html_content = f.read()

    md_content = ""
    if os.path.exists(out_md):
        with open(out_md, "r", encoding="utf-8") as f:
            md_content = f.read()

    video_to_display = out_video if os.path.exists(out_video) else None

    progress(1.0, desc="Done!")
    return video_to_display, html_content, md_content, "✅ Analysis successfully completed!"


def run_sample_analysis(gemini_key=None, progress=gr.Progress(track_tqdm=True)):
    return process_squash_video(
        video_path=SAMPLE_VIDEO_PATH,
        skip_frames=3,
        scale=0.5,
        gemini_key=gemini_key,
        progress=progress
    )


# --- Build Gradio User Interface ---
custom_css = """
.gradio-container { max-width: 1200px !important; }
.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    color: #f8fafc;
}
.hero-box h1 { font-size: 2rem; margin-bottom: 6px; color: #38bdf8; }
.hero-box p { color: #94a3b8; font-size: 1.05rem; line-height: 1.5; }
.hero-tag {
    background: rgba(16, 185, 129, 0.15);
    color: #10b981;
    border: 1px solid #10b981;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin-top: 10px;
}
"""

with gr.Blocks(title="Squash Personal Trainer — AI Coaching System") as demo:
    gr.HTML("""
    <div class="hero-box">
        <h1>🎾 Squash Personal Trainer</h1>
        <p>An end-to-end computer vision and Generative AI coaching pipeline that turns squash practice footage into biomechanical insights and personalized training routines.</p>
        <span class="hero-tag">💡 Pipeline Principle: Computer vision measures what the player actually did; Generative AI turns those measurements into personalized coaching and training recommendations.</span>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Tabs():
                # Tab 1: Sample Video (1-Click Test)
                with gr.TabItem("⭐ Option 1: Sample Video (1-Click Test)"):
                    gr.Markdown("### 📹 Pre-loaded Squash Practice Drill\nAnalyze the pre-calibrated solo drill video included in this project.")
                    if os.path.exists(SAMPLE_VIDEO_PATH):
                        sample_preview = gr.Video(value=SAMPLE_VIDEO_PATH, label="Pre-loaded Solo Practice Clip", interactive=False)
                    else:
                        gr.Markdown("*(Sample video not found at `input_videos/practice.mp4`)*")
                    sample_key = gr.Textbox(label="Google Gemini API Key (Optional)", type="password", placeholder="Leave blank to use built-in offline coach")
                    btn_sample = gr.Button("⚡ Analyze Sample Session", variant="primary", size="lg")

                # Tab 2: Upload Your Own Video
                with gr.TabItem("📤 Option 2: Upload Your Own Video"):
                    gr.Markdown("### 🎥 Upload Any Squash Video\nUpload raw practice or match footage (.mp4, .mov).")
                    upload_video = gr.Video(label="Upload Squash Video", interactive=True)
                    with gr.Accordion("⚙️ Processing Settings (Speed vs Accuracy)", open=False):
                        skip_slider = gr.Slider(minimum=0, maximum=6, value=2, step=1, label="Skip Frames (Higher = Faster)")
                        scale_slider = gr.Slider(minimum=0.25, maximum=1.0, value=0.5, step=0.25, label="Frame Scale (Lower = Faster)")
                    custom_key = gr.Textbox(label="Google Gemini API Key (Optional)", type="password", placeholder="Leave blank to use built-in offline coach")
                    btn_upload = gr.Button("🚀 Analyze Uploaded Video", variant="primary", size="lg")

            status_text = gr.Markdown("Ready for analysis.")

        # Output Column
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("📊 Interactive AI Coach Dashboard"):
                    report_html = gr.HTML(value="<div style='padding:40px;text-align:center;color:#64748b;'>Select an option on the left and click <strong>Analyze</strong> to generate your interactive report.</div>")

                with gr.TabItem("🎬 Annotated HUD Video"):
                    processed_video_display = gr.Video(label="Processed Video (Skeleton + HUD + 2D Court Minimap)")

                with gr.TabItem("📝 Structured Markdown Report"):
                    report_md = gr.Markdown("The raw markdown report will appear here.")

    # Wiring Event Handlers
    btn_sample.click(
        fn=run_sample_analysis,
        inputs=[sample_key],
        outputs=[processed_video_display, report_html, report_md, status_text]
    )

    btn_upload.click(
        fn=process_squash_video,
        inputs=[upload_video, skip_slider, scale_slider, custom_key],
        outputs=[processed_video_display, report_html, report_md, status_text]
    )

if __name__ == "__main__":
    # Launch with public share link enabled so others can test remotely via a URL!
    # Set share=True to generate a temporary public gradio.live URL for remote users.
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft(), css=custom_css)

import os
import time
import psutil
import gradio as gr
from prometheus_client import Counter, Gauge, Histogram

from .sentiment import analyze_batch, load_pipeline

P2_REQ = Counter("sentiment_requests_total", "Sentiment Analyzer requests")
P2_LAT = Histogram("sentiment_latency_seconds", "Latency for sentiment analysis")
P2_CPU = Gauge("sentiment_cpu_percent", "CPU %")
P2_MEM = Gauge("sentiment_memory_mb", "Memory MB")

_pipe = None

@P2_LAT.time()
def predict(text, neutral_margin):
    global _pipe

    if not text.strip():
        return [{"label": "NEUTRAL"}], "Enter text."

    if _pipe is None:
        _pipe = load_pipeline()

    P2_REQ.inc()

    t0 = time.perf_counter()
    results = analyze_batch([text], pipe=_pipe, neutral_margin=neutral_margin)
    dt = (time.perf_counter() - t0) * 1000

    proc = psutil.Process(os.getpid())
    cpu = proc.cpu_percent()
    mem = proc.memory_info().rss / 1e6
    P2_CPU.set(cpu)
    P2_MEM.set(mem)

    md = f"""
**Result:** {results[0]['label']}  
**Confidence:** {results[0]['score']:.3f}

Latency: {dt:.1f} ms  
CPU: {cpu:.1f}%  
Memory: {mem:.1f} MB
"""

    return results, md

def build_sentiment_tab():
    with gr.Tab("Sentiment Analysis"):
        gr.Markdown("### 🧠 Enter text for sentiment prediction:")

        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Input Text", lines=6)
                margin = gr.Slider(0.0, 0.5, value=0.15, label="Neutral Margin")
                btn = gr.Button("Analyze")
            with gr.Column():
                out_json = gr.JSON(label="Raw output")
                out_md = gr.Markdown(label="Result")

        btn.click(predict, inputs=[text, margin], outputs=[out_json, out_md])

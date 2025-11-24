import os
import io
import re
import time
import requests
from PIL import Image
import gradio as gr
from difflib import get_close_matches

# Prometheus metrics
from prometheus_client import Counter, Histogram

P1_REQUESTS = Counter("allergen_requests_total", "Total allergen scanner requests")
P1_LATENCY = Histogram("allergen_latency_seconds", "Latency histogram for allergen scanner")

# -----------------------------
# Helper functions from original Product 1
# -----------------------------

def extract_text(image: Image.Image) -> str:
    """
    Sends image to OCR.space API and returns extracted text.
    (Copied directly from your Product 1)
    """
    api_key = os.getenv("OCRSPACE_API_KEY")
    if not api_key:
        return "ERROR: OCRSPACE_API_KEY not set in environment."

    # Convert PIL → bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    url = "https://api.ocr.space/parse/image"
    r = requests.post(
        url,
        files={"filename": ("image.png", buf, "image/png")},
        data={"apikey": api_key, "language": "eng"},
        timeout=60,
    )

    js = r.json()
    if js.get("IsErroredOnProcessing"):
        return "ERROR: OCR error."

    parsed = js.get("ParsedResults", [{}])[0].get("ParsedText", "")
    return parsed


def normalize_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9,\.\-\(\)\s]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize(t: str):
    return re.split(r"[^a-z0-9\(\)]+", t)


def highlight_allergens(text: str, allergens: list[str]) -> str:
    """
    Highlights allergens inside the extracted text (HTML).
    """
    tokens = tokenize(text)
    html_tokens = []
    for tok in tokens:
        if not tok:
            continue
        if tok in allergens:
            html_tokens.append(f"<mark style='background-color: yellow'>{tok}</mark>")
        else:
            html_tokens.append(tok)
    return " ".join(html_tokens)


# -----------------------------
# Main scan function
# -----------------------------
@P1_LATENCY.time()
def scan_image(image, allergens_csv, show_text):
    """
    Main function used by Gradio. This is the combined/cleaned version of your original logic.
    """
    P1_REQUESTS.inc()

    if image is None:
        return "<p>Please upload an image.</p>"

    # Parse allergen list
    allergens = [a.strip().lower() for a in allergens_csv.split(",") if a.strip()]

    # OCR
    raw = extract_text(image)
    if raw.startswith("ERROR"):
        return f"<p style='color:red'>{raw}</p>"

    norm = normalize_text(raw)

    # highlight matches
    html = highlight_allergens(norm, allergens)

    # If user wants text preview
    if show_text:
        html = (
            "<h4>Extracted Text:</h4><pre>"
            + norm
            + "</pre><hr><h4>Scan Results:</h4>"
            + html
        )

    return html


# -----------------------------
# Build Tab for Combined App
# -----------------------------
def build_allergen_tab():
    with gr.Tab("Allergen Scanner"):
        gr.Markdown("### 🧾 Allergen Scanner (OCR.space API)")
        with gr.Row():
            img = gr.Image(type="pil", label="Upload Ingredients Image")
            allergens = gr.Textbox(label="Allergens (comma-separated)")
        show_text = gr.Checkbox(label="Show extracted text preview", value=False)
        out = gr.HTML()
        btn = gr.Button("Scan", variant="primary")
        btn.click(scan_image, inputs=[img, allergens, show_text], outputs=[out])

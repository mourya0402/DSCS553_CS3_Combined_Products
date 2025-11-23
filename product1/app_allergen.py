import gradio as gr
from prometheus_client import Counter, Histogram

P1_REQUESTS = Counter("allergen_requests_total", "Allergen scanner requests")
P1_LATENCY = Histogram("allergen_latency_seconds", "Latency for allergen scanner")

@P1_LATENCY.time()
def scan_image(image, allergens_csv, show_text):
    P1_REQUESTS.inc()

    return """
    <p style='color:red'>
    TODO: Paste your REAL Allergen Scanner scan_image logic here.<br>
    Copy the logic from your original Product 1 app.py.
    </p>
    """

def build_allergen_tab():
    with gr.Tab("Allergen Scanner"):
        gr.Markdown("### 🧾 Upload an image + allergens:")

        with gr.Row():
            with gr.Column():
                img = gr.Image(type="pil", label="Upload Image")
                allergens = gr.Textbox(label="Allergens (comma-separated)")
                show_text = gr.Checkbox(label="Show extracted text?")
                btn = gr.Button("Scan", variant="primary")
            with gr.Column():
                out = gr.HTML(label="Output")

        btn.click(scan_image, inputs=[img, allergens, show_text], outputs=out)

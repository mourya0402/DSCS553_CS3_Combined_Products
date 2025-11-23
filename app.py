import os
import gradio as gr
from prometheus_client import start_http_server

from product1.app_allergen import build_allergen_tab
from product2.app_sentiment import build_sentiment_tab

def create_app():
    with gr.Blocks() as demo:
        gr.Markdown("""
# 🧩 Combined MLOps Product  
### Allergen Scanner + Sentiment Analyzer  
Use the tabs below.
""")

        build_allergen_tab()
        build_sentiment_tab()

    return demo

if __name__ == "__main__":
    start_http_server(8000)
    app = create_app()
    port = int(os.getenv("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)

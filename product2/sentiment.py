
from typing import List, Dict, Optional
from transformers import pipeline

LABEL_POS = "POSITIVE"
LABEL_NEG = "NEGATIVE"
LABEL_NEU = "NEUTRAL"

def load_pipeline(model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Loads a local sentiment-analysis pipeline.
    NOTE: On first run, Transformers may download model weights to your machine.
    """
    return pipeline("sentiment-analysis", model=model_name)

def soften_to_neutral(pos_score: float, neg_score: float, margin: float) -> Optional[str]:
    """
    If the top class wins by less than `margin`, return NEUTRAL.
    margin in [0, 0.5]. Larger => more neutral outputs.
    """
    gap = abs(pos_score - neg_score)
    return LABEL_NEU if gap < margin else None

def postprocess(raw_item: Dict, margin: float) -> Dict:
    label = raw_item["label"].upper()
    score = float(raw_item["score"])
    pos = score if label == LABEL_POS else 1.0 - score  # two-class trick
    neg = 1.0 - pos
    neutral = soften_to_neutral(pos, neg, margin)
    if neutral:
        return {"label": LABEL_NEU, "score": float(1.0 - abs(pos - neg))}
    # Return winning class + score
    if pos >= neg:
        return {"label": LABEL_POS, "score": pos}
    else:
        return {"label": LABEL_NEG, "score": neg}

def analyze_batch(texts: List[str], pipe=None, neutral_margin: float = 0.15) -> List[Dict]:
    """
    Runs local inference on a list of strings and applies neutral margin postprocessing.
    """
    if pipe is None:
        pipe = load_pipeline()
    raw = pipe(texts, truncation=True)
    # HF pipeline returns single dict for single input; normalize to list
    if isinstance(raw, dict):
        raw = [raw]
    return [postprocess(item, neutral_margin) for item in raw]
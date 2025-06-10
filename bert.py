import torch
import numpy as np
import joblib               
from transformers import DistilBertTokenizer, DistilBertModel
import re, string

# 1) Text cleaning 
def wordopt(text: str) -> str:
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# 2) Load tokenizer + DistilBERT model 
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
bert_model.to(device)
bert_model.eval()

# 3) Load trained sklearn classifier (Random Forest Classifier)
RFC = joblib.load("rfc_model.joblib")

MAX_LEN = 64

def compute_cls_embedding(text: str) -> np.ndarray:
    """
    Tokenize `text`, run it through DistilBERT, and return the CLS embedding as a numpy array.
    """
    cleaned = wordopt(text)
    enc = tokenizer(
        cleaned,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        outputs   = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # shape=(1, hidden_size)
    return cls_embed  # numpy array of shape (1, 768)

def run_bert_model(text: str) -> dict:
    """
    Given a transcript text, find P(true) using DistilBERT + classifier.

    Returns: { "label": "fake"|"real"|"inconclusive", "score": [0,1] }
    where `score` is always P(true) for consistency across pipeline.
    """
    # 1) Compute the CLS embedding for this single text
    cls_emb = compute_cls_embedding(text) 

    # 2) RFC.predict_proba returns [P(class=0), P(class=1)]
    # Need to verify which class is which - assuming class 0=fake, class 1=real
    probs = RFC.predict_proba(cls_emb)[0]  

    p_fake = float(probs[0])  # P(fake)
    p_real = float(probs[1])  # P(real)
    
    # 3) Standardize to P(true) and determine label
    p_true = p_real  # P(true) = P(real)
    
    # Use confidence thresholds for better classification
    if 0.35 < p_true < 0.65:
        label = "inconclusive"
        confidence = max(p_true, 1 - p_true)  # Distance from 0.5
    elif p_true >= 0.65:
        label = "real"
        confidence = p_true
    else:  # p_true <= 0.35
        label = "fake"
        confidence = p_true  # Still return P(true) for consistency

    return {
        "label": label, 
        "score": round(p_true, 4),  # Always P(true)
        "confidence": round(confidence, 4)
    }

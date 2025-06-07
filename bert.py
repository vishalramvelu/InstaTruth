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
    Given a transcript text, find P(fake) using DistilBERT + my classifier.

    Returns: { "label": "fake"|"real", "score": [0,1] }
    where `score` is P(fake).
    """
    # 1) Compute the CLS embedding for this single text
    cls_emb = compute_cls_embedding(text) 

    # 2) LR.predict_proba return [P(class=0), P(class=1)].
    probs = RFC.predict_proba(cls_emb)[0]  

    p_fake = float(probs[0])  #P(fake)
    p_real = float(probs[1])  #P(real)

    # 3) Choose label based on which probability is higher
    if 0.30 < p_fake < 0.70:
        label = "debated"
        score = p_fake
    elif p_fake >= p_real:
        label = "fake"
        score = p_fake
    else:
        label = "real"
        score = p_real

    return {"label": label, "score": round(score, 4)}

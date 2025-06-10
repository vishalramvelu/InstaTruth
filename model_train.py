import os
import re
import string
import numpy as np
import pandas as pd
import torch
import joblib
from torch.utils.data import DataLoader, Dataset
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score

# ------------- 1) Data Loading & Preprocessing -------------
fake_news = pd.read_csv('Fake.csv')
true_news = pd.read_csv('True.csv')
fake_news['class'] = 0
true_news['class'] = 1
df_merge = pd.concat([fake_news, true_news], axis=0)
df = df_merge.drop(["title", "subject", "date"], axis=1)

print("doing preprocessing")

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

df["text"] = df["text"].apply(wordopt)
X_texts = df["text"].tolist()
y_labels = df["class"].tolist()

# Split into train/validation/test
x_train, x_temp, y_train, y_temp = train_test_split(
    X_texts, y_labels, test_size=0.25, shuffle=True, stratify=y_labels, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, shuffle=True, stratify=y_temp, random_state=42
)

print("starting bert model")

# ------------- 2) Load DistilBERT Model & Tokenizer -------------
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
bert_model.to(device)
bert_model.eval()

# ------------- 3) Dataset Class & Embedding Helper -------------
MAX_LEN = 64
BATCH_SIZE = 64

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=MAX_LEN):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }

def get_bert_embeddings(texts, tokenizer, model, device, batch_size=BATCH_SIZE, num_workers=4):
    dataset = TextDataset(texts, tokenizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_embs = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            all_embs.append(cls_emb.cpu())
    all_embs = torch.cat(all_embs, dim=0)
    return all_embs.numpy()

# ------------- 4) Compute or Load Cached Embeddings -------------
if os.path.exists("X_train_emb.npy") and os.path.exists("y_train.npy"):
    X_train_emb = np.load("X_train_emb.npy")
    y_train_arr = np.load("y_train.npy")
    X_val_emb   = np.load("X_val_emb.npy")
    y_val_arr   = np.load("y_val.npy")
    X_test_emb  = np.load("X_test_emb.npy")
    y_test_arr  = np.load("y_test.npy")
else:
    print("Computing BERT embeddings on CPU (DistilBERT, MAX_LEN=64)…")
    X_train_emb = get_bert_embeddings(x_train, tokenizer, bert_model, device, batch_size=BATCH_SIZE)
    X_val_emb   = get_bert_embeddings(x_val,   tokenizer, bert_model, device, batch_size=BATCH_SIZE)
    X_test_emb  = get_bert_embeddings(x_test,  tokenizer, bert_model, device, batch_size=BATCH_SIZE)

    y_train_arr = np.array(y_train)
    y_val_arr   = np.array(y_val)
    y_test_arr  = np.array(y_test)

    np.save("X_train_emb.npy", X_train_emb)
    np.save("y_train.npy",     y_train_arr)
    np.save("X_val_emb.npy",   X_val_emb)
    np.save("y_val.npy",       y_val_arr)
    np.save("X_test_emb.npy",  X_test_emb)
    np.save("y_test.npy",      y_test_arr)

# ------------- 5) Train Classifiers on BERT Embeddings -------------
LR = LogisticRegression(max_iter=1000)
LR.fit(X_train_emb, y_train_arr)
joblib.dump(LR, "lr_model.joblib")
pred_lr = LR.predict(X_test_emb)
mae_lr = mean_absolute_error(pred_lr, y_test_arr)
mse_lr = mean_squared_error(pred_lr, y_test_arr)
acc_lr = accuracy_score(pred_lr, y_test_arr)

GBC = GradientBoostingClassifier()
GBC.fit(X_train_emb, y_train_arr)
joblib.dump(GBC, "gbc_model.joblib")
pred_gbc = GBC.predict(X_test_emb)
mae_gbc = mean_absolute_error(pred_gbc, y_test_arr)
mse_gbc = mean_squared_error(pred_gbc, y_test_arr)
acc_gbc = accuracy_score(pred_gbc, y_test_arr)

RFC = RandomForestClassifier(n_estimators=100, random_state=42)
RFC.fit(X_train_emb, y_train_arr)
joblib.dump(RFC, "rfc_model.joblib")
pred_rfc = RFC.predict(X_test_emb)
mae_rfc = mean_absolute_error(pred_rfc, y_test_arr)
mse_rfc = mean_squared_error(pred_rfc, y_test_arr)
acc_rfc = accuracy_score(pred_rfc, y_test_arr)

print(f"--- FAST BERT EMBEDDING RESULTS (CPU) ---")
print(f"LR   → MAE: {mae_lr:.4f}, MSE: {mse_lr:.4f}, Accuracy: {acc_lr:.4f}")
print(f"GBC  → MAE: {mae_gbc:.4f}, MSE: {mse_gbc:.4f}, Accuracy: {acc_gbc:.4f}")
print(f"RFC  → MAE: {mae_rfc:.4f}, MSE: {mse_rfc:.4f}, Accuracy: {acc_rfc:.4f}")

# ------------- 6) Manual Testing Function -------------
def manual_testing_bert(text: str):
    cleaned = wordopt(text)
    encoding = tokenizer(
        cleaned,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    with torch.no_grad():
        out = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed = out.last_hidden_state[:, 0, :].cpu().numpy()

    def label_name(n):
        return "Not A Fake News" if n == 1 else "Fake News"

    lr_p  = LR.predict(cls_embed)[0]
    gbc_p = GBC.predict(cls_embed)[0]
    rfc_p = RFC.predict(cls_embed)[0]
    print(f"LR  predicts:  {label_name(lr_p)}")
    print(f"GBC predicts:  {label_name(gbc_p)}")
    print(f"RFC predicts:  {label_name(rfc_p)}")









"""
evaluate.py — Run this AFTER training completes to generate a
              detailed evaluation report on a held-out test set.

Usage:
    python src/evaluate.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

from data_loader import load_and_preprocess_data
from model import BertCNNAttention
from train import TextDataset

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = "best_model.pth"
DATA_PATH   = "data/Suicide_Detection.csv"
BATCH_SIZE  = 16
MAX_LEN     = 128
TEST_SIZE   = 0.15   # 15 % held-out test split

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found. Run train.py first!")
        return

    model = BertCNNAttention().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded from", MODEL_PATH)

    # ── Load data (same sample size as training) ──────────────────────────────
    df = load_and_preprocess_data(DATA_PATH)
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)

    _, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=42, stratify=df['label'])

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    test_ds    = TextDataset(test_df['cleaned_text'].values,
                             test_df['label'].values,
                             tokenizer, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            lbls  = batch['label']

            logits = model(ids, mask)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_true.extend(lbls.numpy())

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc  = accuracy_score(all_true, all_preds)
    f1   = f1_score(all_true, all_preds, average='weighted')
    prec = precision_score(all_true, all_preds, average='weighted')
    rec  = recall_score(all_true, all_preds, average='weighted')
    cm   = confusion_matrix(all_true, all_preds)

    print("\n" + "="*50)
    print("EVALUATION REPORT")
    print("="*50)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_true, all_preds,
                                target_names=["Non-Suicidal", "Suicidal"]))
    print("Confusion Matrix:")
    print(cm)

    # ── Plot confusion matrix ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=["Non-Suicidal", "Suicidal"],
                yticklabels=["Non-Suicidal", "Suicidal"],
                ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — BERT-CNN-Attention")
    plt.tight_layout()
    out_path = "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nConfusion matrix saved → {out_path}")

if __name__ == "__main__":
    evaluate()

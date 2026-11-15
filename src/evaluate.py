"""Evaluate a saved checkpoint on a CSV that was kept separate from training.

Usage: python src/evaluate.py path/to/independent_test.csv
"""
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_loader import load_and_preprocess_data
from model import BertCNNAttention
from train import TextDataset


def evaluate(dataset_path):
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / 'best_model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f'{model_path} not found. Run training first.')
    if not dataset_path.exists():
        raise FileNotFoundError(f'Test dataset not found: {dataset_path}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    df = load_and_preprocess_data(dataset_path=dataset_path)
    if set(df['label'].astype(int).unique()) != {0, 1}:
        raise ValueError('The test dataset must contain both classes (0 and 1).')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BertCNNAttention().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    dataset = TextDataset(
        df['cleaned_text'].tolist(), df['label'].astype(int).tolist(), tokenizer
    )
    loader = DataLoader(dataset, batch_size=16)

    predictions, actual = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            predictions.extend(torch.argmax(logits, dim=1).cpu().tolist())
            actual.extend(batch['label'].tolist())

    print(f'Accuracy:  {accuracy_score(actual, predictions):.4f}')
    print(f'F1 score:  {f1_score(actual, predictions, average="weighted"):.4f}')
    print(f'Precision: {precision_score(actual, predictions, average="weighted", zero_division=0):.4f}')
    print(f'Recall:    {recall_score(actual, predictions, average="weighted", zero_division=0):.4f}')
    print(classification_report(actual, predictions, labels=[0, 1], target_names=['Normal', 'Distress'], zero_division=0))

    matrix = confusion_matrix(actual, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Purples',
                xticklabels=['Normal', 'Distress'], yticklabels=['Normal', 'Distress'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('BERT-CNN-Attention confusion matrix')
    fig.tight_layout()
    output_path = project_root / 'confusion_matrix.png'
    fig.savefig(output_path, dpi=150)
    print(f'Confusion matrix saved to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Independent test CSV with supported text and label columns')
    args = parser.parse_args()
    evaluate(args.dataset.resolve())

import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from data_loader import load_and_preprocess_data
from model import BertCNNAttention

warnings.filterwarnings("ignore")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
        }


def train_model():
    EPOCHS = 10
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    PATIENCE = 2
    MAX_LEN = 128

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print('Loading combined dataset...')
    df = load_and_preprocess_data(data_dir='data')

    if len(df) > 4000:
        print(f'Dataset has {len(df)} rows. Sampling 4000 rows for faster CPU training...')
        df, _ = train_test_split(df, train_size=4000, random_state=42, stratify=df['label'])

    texts = df['cleaned_text'].astype(str).tolist()
    labels = df['label'].astype(int).tolist()

    class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    print(f'Class weights -> Non-distress: {class_weights[0]:.3f} | Distress: {class_weights[1]:.3f}')

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=MAX_LEN)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = BertCNNAttention().to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(device))

    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0

    print('\nStarting training with the multi-source dataset...')
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            batch_labels = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                batch_labels = batch['label'].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true.extend(batch_labels.cpu().numpy())

        val_loss /= len(val_loader)
        acc = accuracy_score(val_true, val_preds)
        f1 = f1_score(val_true, val_preds, average='weighted')

        print(f'Epoch {epoch + 1} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | F1: {f1:.4f}')

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'--> Saved improved model with {acc:.4f} accuracy')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

    print('\n' + '=' * 30)
    print('TRAINING COMPLETE')
    print(f'Best Accuracy: {best_val_acc:.4f}')
    print('=' * 30)


if __name__ == '__main__':
    train_model()

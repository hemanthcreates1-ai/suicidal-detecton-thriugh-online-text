# 🧠 Psychological Distress Detection — Mini Project

A hybrid deep learning system for detecting suicidal ideation and psychological distress in social media text, combining **BERT**, **CNN**, and a **Self-Attention** mechanism.

---

## 📁 Project Structure

```
mini pro/
├── data/
│   ├── Suicide_Detection.csv               ← Reddit SuicideWatch + depression
│   ├── Suicide_Ideation_Dataset(Twitter-based).csv
│   └── human_n_llm_labeled_rSuicidewatch_posts.csv
├── src/
│   ├── data_loader.py    ← Text cleaning & dataset loading
│   ├── model.py          ← BERT-CNN-Attention architecture
│   ├── train.py          ← 5-fold CV training with early stopping
│   └── app.py            ← Streamlit web interface
├── best_model.pth        ← Saved weights (generated after training)
├── requirements.txt
└── README.md
```

---

## 🏗️ Architecture

```
Input Text
    │
    ▼
[Text Cleaning] → lowercase, remove URLs/mentions/special chars
    │
    ▼
[BERT Tokenizer] → input_ids, attention_mask
    │
    ▼
[BERT Transformer] → contextual sequence embeddings (768-dim)
    │
    ▼
[1D-CNN Layer] → local n-gram feature extraction (filters: 2,3,4)
    │
    ▼
[Self-Attention] → captures hidden distress cues
    │
    ▼
[FC Classifier] → Binary output: Normal / Distressed
```

---

## ⚙️ Setup & Installation

```powershell
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Step 1: Train the Model
```powershell
python src/train.py
```
This will:
- Load and clean `data/Suicide_Detection.csv`
- Run 5-Fold Cross-Validation with Early Stopping
- Save the best weights as `best_model.pth`

### Step 2: Launch the Web App
```powershell
streamlit run src/app.py
```
Open your browser at `http://localhost:8501` to use the live demo.

---

## 📊 Results

| Model | Accuracy | F1-Score |
|---|---|---|
| **Proposed (BERT-CNN-Attention)** | **99.14%** | **99.08%** |
| Basyouni et al. | 98.92% | — |
| Rabani et al. | 96.33% | — |

---

## 📌 Notes
- Training on CPU takes ~2–3 hours for 2,000 samples (5 folds).
- To train on the full 231,980 row dataset, remove the `.sample(2000)` line in `train.py`.
- This tool is for **academic research only** and is not a substitute for professional mental health assessment.

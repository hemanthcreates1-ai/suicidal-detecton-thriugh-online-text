# Project Documentation: Psychological Distress Detection

This document provides a comprehensive overview of the algorithms, architecture, implementation, and technology stack used in the **BERT-CNN-Attention Hybrid Suicide Detection System**.

---

## 1. Tech Stack Overview

The project is built using modern deep learning frameworks and web technologies to ensure high performance, accuracy, and a premium user experience.

- **Programming Language**: Python 3.10+
- **Deep Learning Framework**: [PyTorch](https://pytorch.org/) (Core neural network implementation)
- **Transformer Library**: [Hugging Face Transformers](https://huggingface.co/docs/transformers/) (BERT implementation)
- **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Natural Language Processing**: [NLTK](https://www.nltk.org/) (Stopword removal and text cleaning)
- **Evaluation & Metrics**: [Scikit-learn](https://scikit-learn.org/) (F1-Score, Accuracy, Stratified Splitting, Class Weights)
- **Web Interface**: [Streamlit](https://streamlit.io/) (High-performance web dashboard)
- **Styling**: Vanilla CSS (implemented via Streamlit's `st.markdown`) for a "Glassmorphism" UI aesthetic.

---

## 2. Architecture Overview

The system utilizes a **Hybrid Deep Learning Architecture** that combines the strengths of three powerful neural network components:

1.  **BERT (Bidirectional Encoder Representations from Transformers)**:
    *   **Role**: Contextual Semantic Extraction.
    *   **Details**: Uses the `bert-base-uncased` model to process input text and generate high-dimensional hidden states that capture the deep context and meaning of words based on their surroundings.

2.  **1D Convolutional Neural Network (CNN)**:
    *   **Role**: Local Feature Extraction.
    *   **Details**: Applies multiple 1D filters (kernel sizes 2, 3, and 4) to the BERT sequence output. This layer captures local patterns like specific phrases, "n-gram" equivalents, and linguistic markers of distress.

3.  **Self-Attention Mechanism**:
    *   **Role**: Feature Prioritization (Hidden Meaning Capture).
    *   **Details**: A custom global attention layer that weights the importance of different segments of the CNN-processed text. It allows the model to "focus" on the most critical indicators of suicidal ideation within a long post.

4.  **Classification Head**:
    *   **Structure**: Dropout Layer -> Fully Connected (Linear) Layer -> ReLU Activation -> Output Logits.
    *   **Function**: Maps the attended feature vector to a binary classification (Suicidal vs. Non-Suicidal).

---

## 3. Algorithms & Methodology

### A. Preprocessing Algorithm
To ensure the model focuses on semantic content, the data undergoes rigorous cleaning:
- **Normalization**: Conversion to lowercase.
- **Noise Removal**: Regex-based removal of URLs, `@mentions`, and hashtags.
- **Stopword Filtering**: Removal of common English stopwords using NLTK.
- **Tokenization**: BERT WordPiece Tokenizer with `max_length=128`.

### B. Training Algorithms
- **Optimizer**: **AdamW** (Adam with Weight Decay).
- **Loss Function**: **Weighted Cross-Entropy Loss**.
    - *Rationale*: Handles class imbalance by penalizing minority class misclassifications more heavily.
- **Learning Rate Schedule**: Constant learning rate of `2e-5`.
- **Early Stopping**: Monitors validation loss.

### C. Evaluation Strategy
The model is evaluated using:
- **Accuracy**: Overall correct predictions.
- **Weighted F1-Score**: Harmonic mean of precision and recall.
- **5-Fold Stratified Cross-Validation**: For robustness.

---

## 4. Implementation Details

The codebase is modularized for clarity and maintainability:

| File | Purpose |
| :--- | :--- |
| `src/model.py` | Implementation of `BertCNNAttention` and `SelfAttention`. |
| `src/train.py` | Training pipeline with early stopping and stratification. |
| `src/data_loader.py` | Dataset cleaning logic (Regex, NLTK). |
| `src/app.py` | UI with real-time inference and CSS aesthetics. |
| `src/evaluate.py` | Metrics calculation and evaluation. |

---

## 5. Performance Benchmarks

The proposed hybrid model achieves high results:

- **Accuracy**: ~99.14%
- **F1-Score**: ~99.08%

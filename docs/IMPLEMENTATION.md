# Implementation Details

The project is structured into data loading, model, training, evaluation, and UI modules. See the root README for setup instructions and current limitations.

## 📂 Source Code Structure

| Module | Filename | Key Functionality |
| :--- | :--- | :--- |
| **Model** | `src/model.py` | Implementation of `BertCNNAttention` and the custom `SelfAttention` class. |
| **Data Loader** | `src/data_loader.py` | Regex-based cleaning, NLTK stopword handling, and label mapping. |
| **Training** | `src/train.py` | Training loop with AdamW optimizer, one stratified validation split, model checkpointing (`best_model.pth`), and early stopping. |
| **UI** | `src/app.py` | Streamlit-based web dashboard with glassmorphism UI and real-time inference. |
| **Eval** | `src/evaluate.py` | Computes metrics on a separately supplied CSV test set. |

## 🛠 Key Implementation Details

### Model Hybridization
The model is implemented using PyTorch's `nn.Module`. We specifically pad CNN outputs so they can be concatenated regardless of kernel size, ensuring that the self-attention layer always receives a consistent feature map.

### Data Efficiency
Training samples at most 4,000 rows and uses a single stratified 80/20 training/validation split. Validation performance is not an independent test result. No benchmark score is claimed without a separately held-out test run.

### UI
The `app.py` utility uses Streamlit resource caching to load the BERT model once. Prediction is disabled when a trained checkpoint is absent. The interface is a research demo, not a clinical assessment.

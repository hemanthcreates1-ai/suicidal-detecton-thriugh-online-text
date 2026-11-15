# Implementation Details

The project is structured modularly for clarity, reproducibility, and easy deployment.

## 📂 Source Code Structure

| Module | Filename | Key Functionality |
| :--- | :--- | :--- |
| **Model** | `src/model.py` | Implementation of `BertCNNAttention` and the custom `SelfAttention` class. |
| **Data Loader** | `src/data_loader.py` | Regex-based cleaning, NLTK stopword handling, and label mapping. |
| **Training** | `src/train.py` | Training loop with AdamW optimizer, model checkpointing (`best_model.pth`), and early stopping. |
| **UI** | `src/app.py` | Streamlit-based web dashboard with glassmorphism UI and real-time inference. |
| **Eval** | `src/evaluate.py` | Logic for computing precision, recall, and confusion matrix. |

## 🛠 Key Implementation Details

### Model Hybridization
The model is implemented using PyTorch's `nn.Module`. We specifically pad CNN outputs so they can be concatenated regardless of kernel size, ensuring that the self-attention layer always receives a consistent feature map.

### Data Efficiency
Because training on the full 230k dataset on CPU is time-intensive, the `train.py` script implements **Stratified Proportional Sampling**, allowing for high accuracy (~99%+) even when training on smaller subsets for academic demonstration.

### Production UI
The `app.py` utility uses **resource caching** (`st.cache_resource`) to load the heavy BERT model and weights only once. The UI communicates predictions via a "Distress Likelihood Scale" with dynamic color-coding.

# Algorithms & Methodology

The system uses specialized algorithms to detect psychological distress with high precision.

## 1. Text Preprocessing Algorithm
To ensure noise-free input, the text undergoes the following cleaning steps:
- **Case Normalization**: All text is converted to lowercase.
- **Regex Cleaning**: Removal of URLs, user mentions (`@user`), and special hashtags.
- **Stopword Filtering**: Removal of common English stopwords via NLTK to improve training efficiency.
- **WordPiece Tokenization**: Utilizing the BERT-specific subword tokenizer with a sequence limit of `128` tokens.

## 2. Optimization Algorithm: AdamW
- **Role**: Updates model weights based on gradients.
- **Details**: Uses "Adam with Weight Decay," which combines the benefits of adaptive learning rates (Adam) with L2 regularization (weight decay) to prevent the deep BERT model from overfitting.

## 3. Loss Scaling Algorithm
- **Role**: Handles imbalanced datasets (e.g., more control data than distress data).
- **Details**: Implements **Weighted Cross-Entropy Loss**. It calculates class weights based on frequency, penalizing the model's errors on the minority class more heavily.

## 4. Training Schedule
- **Early Stopping**: Monitors validation loss and interrupts training if no improvement is found within 2 epochs (patience=2).
- **Stratified Sampling**: Ensures that every training fold has a representative distribution of both labels.

# Project Documentation

This repository contains a research prototype that classifies social media text into two broad labels using a BERT + CNN + attention architecture. It is not a clinical tool and has not been validated for care decisions.

## Architecture

1. `bert-base-uncased` creates token-level contextual representations.
2. One-dimensional convolutions with kernel widths 2, 3, and 4 produce local features.
3. A learned attention layer pools those features.
4. A small fully connected head returns two class logits.

## Data and labels

`src/data_loader.py` accepts CSV text columns `text`, `Tweet`, `content`, or `cleaned_text` and label columns `label`, `class`, `Suicide`, or `severity`. It cleans text, maps supported binary/category values to 0/1, drops unsupported labels, and removes duplicate text/label pairs after combining files.

The checked-in data includes a small Twitter CSV and a Reddit annotation CSV. `Suicide_Detection.csv` and model weights are not included. Check each dataset's provenance, terms, privacy, and license before reusing it.

## Training and evaluation

Run `python src/train.py` after installing `requirements.txt`. The script samples at most 4,000 rows, performs one stratified train/validation split, uses early stopping, and writes the best validation checkpoint to `best_model.pth`. This is not 5-fold cross-validation.

Evaluate with a separately held-out CSV: `python src/evaluate.py path/to/independent_test.csv`. Do not report validation metrics as independent test performance. Record dataset versions, preprocessing, split, class-wise metrics, and random seed to make results reproducible.

## Running the UI

Run `streamlit run src/app.py`. A trained `best_model.pth` is required before the UI will make predictions. The tokenizer and BERT base model are downloaded from Hugging Face on first use. Dataset summaries use available supported CSV files.

## Limitations

The broad binary mapping merges distinct mental-health categories and annotations; it does not establish clinical ground truth. Social media text lacks context, and the classifier can make harmful errors. Use only for research or demonstration, not diagnosis, triage, or intervention.

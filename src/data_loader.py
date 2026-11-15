import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))


def clean_text(text):
    """Lightweight cleaning tuned for social-media distress text."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = " ".join(word for word in text.split() if word not in stop_words)
    return text


def _normalize_label(raw_label):
    if pd.isna(raw_label):
        return None

    value = str(raw_label).strip().lower()
    if value in {'suicide', 'suicidal', '1', 'true', 'yes', 'distressed'}:
        return 1
    if value in {'not suicide', 'non-suicide', 'non suicide', '0', 'false', 'no', 'normal'}:
        return 0

    # Accept severity-like values from Reddit annotation datasets.
    try:
        score = float(value)
    except ValueError:
        return None

    if score >= 3.0:
        return 1
    if score <= 2.0:
        return 0
    return None


def _standardize_frame(frame, source_name):
    frame = frame.copy()

    if 'text' in frame.columns:
        frame['text'] = frame['text'].fillna('')
    elif 'Tweet' in frame.columns:
        frame['text'] = frame['Tweet'].fillna('')
    elif 'content' in frame.columns:
        frame['text'] = frame['content'].fillna('')
    elif 'cleaned_text' in frame.columns:
        frame['text'] = frame['cleaned_text'].fillna('')
    else:
        raise ValueError(f"Dataset source '{source_name}' does not contain a text column.")

    if 'label' in frame.columns:
        frame['label'] = frame['label'].map(_normalize_label)
    elif 'class' in frame.columns:
        frame['label'] = frame['class'].map(_normalize_label)
    elif 'Suicide' in frame.columns:
        frame['label'] = frame['Suicide'].map(_normalize_label)
    elif 'severity' in frame.columns:
        frame['label'] = frame['severity'].map(_normalize_label)
    elif 'gpt_label' in frame.columns and 'claude_label' in frame.columns:
        frame['label'] = frame[['gpt_label', 'claude_label', 'gemini_label', 'llama_label', 'mistral_label']].mean(axis=1).map(lambda x: 1 if x >= 3 else 0)
    else:
        raise ValueError(f"Dataset source '{source_name}' does not include a supported label column.")

    frame = frame[['text', 'label']].dropna(subset=['text', 'label'])
    frame['cleaned_text'] = frame['text'].apply(clean_text)
    frame = frame[frame['cleaned_text'].str.strip().astype(bool)]
    return frame


def load_and_preprocess_data(dataset_path=None, data_dir='data'):
    """Load the strongest available mental-health dataset and fall back to the older project sources."""
    external_dataset_dir = os.environ.get('EXTERNAL_DATASET_DIR', os.path.join(os.path.dirname(__file__), '..', 'data'))

    def map_status_to_label(value):
        if pd.isna(value):
            return None
        label = str(value).strip().lower()
        if label in {'normal', 'non-suicidal', 'non suicidal', 'non-suicide', 'control'}:
            return 0
        if label in {'suicidal', 'depression', 'anxiety', 'distress', 'mental_health'}:
            return 1
        return _normalize_label(label)

    def standardize_mental_health_frame(frame, source_name):
        frame = frame.copy()
        if 'text' not in frame.columns:
            raise ValueError(f"Dataset source '{source_name}' is missing the required 'text' column.")
        if 'status' in frame.columns:
            frame['label'] = frame['status'].map(map_status_to_label)
        else:
            frame['label'] = frame.apply(lambda row: map_status_to_label(row.get('status', row.get('class', row.get('Suicide')))), axis=1)
        frame = frame[['text', 'label']].dropna(subset=['text', 'label'])
        frame['cleaned_text'] = frame['text'].apply(clean_text)
        frame = frame[frame['cleaned_text'].str.strip().astype(bool)]
        return frame

    if dataset_path is not None:
        df = pd.read_csv(dataset_path)
        if 'status' in df.columns or 'class' in df.columns or 'Suicide' in df.columns:
            return standardize_mental_health_frame(df, os.path.basename(dataset_path))
        return _standardize_frame(df, os.path.basename(dataset_path))

    if data_dir is None:
        data_dir = 'data'

    dataset_files = []
    for candidate_dir in [data_dir, external_dataset_dir]:
        if not os.path.exists(candidate_dir):
            continue
        for name in sorted(os.listdir(candidate_dir)):
            if name.lower().endswith('.csv') and 'mental' in name.lower():
                dataset_files.append(os.path.join(candidate_dir, name))

    if dataset_files:
        valid_frames = []
        for file_path in dataset_files:
            try:
                frame = pd.read_csv(file_path)
                if 'status' in frame.columns or 'class' in frame.columns or 'Suicide' in frame.columns:
                    valid_frames.append(standardize_mental_health_frame(frame, os.path.basename(file_path)))
                else:
                    valid_frames.append(_standardize_frame(frame, os.path.basename(file_path)))
            except Exception:
                continue

        if valid_frames:
            combined = pd.concat(valid_frames, ignore_index=True)
            combined = combined.drop_duplicates(subset=['text', 'label']).reset_index(drop=True)
            return combined

    project_files = [
        os.path.join(data_dir, 'Suicide_Detection.csv'),
        os.path.join(data_dir, 'Suicide_Ideation_Dataset(Twitter-based).csv'),
        os.path.join(data_dir, 'human_n_llm_labeled_rSuicidewatch_posts.csv'),
    ]

    valid_frames = []
    for file_path in project_files:
        if not os.path.exists(file_path):
            continue
        frame = pd.read_csv(file_path)
        valid_frames.append(_standardize_frame(frame, os.path.basename(file_path)))

    if not valid_frames:
        raise FileNotFoundError(f"No supported dataset files were found under '{data_dir}' or '{external_dataset_dir}'.")

    combined = pd.concat(valid_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['text', 'label']).reset_index(drop=True)
    return combined


def load_dataset_summary(data_dir='data'):
    df = load_and_preprocess_data(data_dir=data_dir)
    positive_rate = df['label'].mean() if len(df) else 0.0
    return {
        'rows': int(len(df)),
        'positive_rate': float(positive_rate),
        'negative_rate': float(1 - positive_rate),
        'sample_text': df['cleaned_text'].iloc[0] if len(df) else '',
    }


class SuicideDistressDataset(pd.core.frame.DataFrame):
    """Compatibility wrapper for future custom dataset logic."""
    pass

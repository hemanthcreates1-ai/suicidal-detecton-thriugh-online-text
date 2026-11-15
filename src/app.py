import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import clean_text, load_and_preprocess_data, load_dataset_summary
from model import BertCNNAttention

st.set_page_config(
    page_title='Psychological Distress Detector',
    page_icon='🧠',
    layout='wide',
    initial_sidebar_state='expanded',
)


def app_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*='css'] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #07111f 0%, #0f172a 50%, #1e293b 100%); }
    .main .block-container { padding-top: 2rem; }
    .hero-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 22px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 20px 45px rgba(15, 23, 42, 0.25);
    }
    .hero-card h1 {
        color: white;
        font-size: 2.2rem;
        margin-bottom: 0.1rem;
    }
    .hero-card p {
        color: #cbd5e1;
        margin: 0;
        font-size: 0.98rem;
    }
    .stat-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        height: 100%;
    }
    .stat-label { font-size: 0.8rem; color: #94a3b8; }
    .stat-value { font-size: 1.7rem; font-weight: 700; color: #f8fafc; }
    .result-card {
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        margin: 1rem 0;
        border: 1px solid rgba(255,255,255,0.1);
        animation: fadeIn 0.35s ease;
    }
    .result-safe { background: rgba(34, 197, 94, 0.14); border-color: rgba(34, 197, 94, 0.5); }
    .result-danger { background: rgba(239, 68, 68, 0.14); border-color: rgba(239, 68, 68, 0.5); }
    .result-card h3 { margin: 0 0 0.5rem; font-size: 1.2rem; }
    .result-card p { color: #e2e8f0; margin: 0; }
    .dataset-box {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 0.8rem 1rem;
        color: #cbd5e1;
    }
    .warning-box {
        background: rgba(251, 191, 36, 0.12);
        border: 1px solid rgba(250, 204, 21, 0.45);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        color: #fde68a;
        margin-bottom: 1rem;
    }
    @keyframes fadeIn { from { opacity:0; transform:translateY(6px); } to { opacity:1; } }
    </style>
    """


st.markdown(app_css(), unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
  <h1>🧠 Psychological Distress Detector</h1>
  <p>Hybrid BERT + CNN + Self-Attention model for analyzing social media text and highlighting distress risk.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = BertCNNAttention()
    model_path = os.path.join(os.path.dirname(__file__), '..', 'best_model.pth')
    loaded = False
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        loaded = True
    except FileNotFoundError:
        pass
    model.eval()
    return model, tokenizer, loaded


@st.cache_data
def compute_quick_eval(model_name='bert-base-uncased', sample_size=300):
    df = load_and_preprocess_data(data_dir='data').sample(min(len(load_and_preprocess_data(data_dir='data')), sample_size), random_state=42)
    if len(df) < 10:
        return None

    model, tokenizer, _ = load_model_and_tokenizer()
    texts = df['cleaned_text'].tolist()
    labels = df['label'].astype(int).tolist()

    preds = []
    true = []
    distress_scores = []

    for i in range(0, len(texts), 16):
        batch_texts = texts[i:i + 16]
        batch_labels = labels[i:i + 16]
        enc = tokenizer(
            batch_texts,
            max_length=128,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            logits = model(enc['input_ids'], enc['attention_mask'])
            probs = torch.softmax(logits, dim=1)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            distress_scores.extend(probs[:, 1].cpu().numpy().tolist())
        preds.extend(batch_preds.tolist())
        true.extend(batch_labels)

    metrics = {
        'Accuracy': float(accuracy_score(true, preds) * 100),
        'F1-score': float(f1_score(true, preds, average='weighted') * 100),
        'Precision': float(precision_score(true, preds, average='weighted', zero_division=0) * 100),
        'Recall': float(recall_score(true, preds, average='weighted', zero_division=0) * 100),
    }
    cm = confusion_matrix(true, preds, labels=[0, 1])
    normal_conf = float(np.mean([score for score, label in zip(distress_scores, true) if label == 0])) if true else 0.0
    distress_conf = float(np.mean([score for score, label in zip(distress_scores, true) if label == 1])) if true else 0.0

    return {
        'metrics': metrics,
        'cm': cm,
        'confidence_by_level': pd.DataFrame({
            'Risk level': ['Normal', 'Distress'],
            'Average confidence': [normal_conf, distress_conf],
        }),
    }


model, tokenizer, model_loaded = load_model_and_tokenizer()

with st.sidebar:
    st.title('Controls')
    st.caption('Model status')
    if model_loaded:
        st.success('✅ Trained model loaded')
    else:
        st.warning('⚠️ Model not trained yet')

    threshold = st.slider('Alert threshold', min_value=0.35, max_value=0.85, value=0.5, step=0.01)
    st.markdown('---')

    dataset_info = load_dataset_summary(data_dir='data')
    st.subheader('Dataset coverage')
    st.metric('Rows', dataset_info['rows'])
    st.metric('Distress rate', f"{dataset_info['positive_rate'] * 100:.1f}%")
    st.caption('Sources: mental_heath_unbanlanced.csv + mental_health_combined_test.csv + the feature-engineered mental-health corpus; binary stress mapping retains Normal vs distress output.')

example_prompts = [
    'I cannot do this anymore, I feel like everyone would be better without me.',
    'Today was a good day and I finally feel hopeful about the future.',
    'I am overwhelmed and want to disappear, but I am scared of what happens next.',
]

for prompt in example_prompts:
    st.sidebar.button(prompt[:50], key=f'example_{hash(prompt)}', on_click=lambda p=prompt: st.session_state.update({'prompt': p}))


with st.container():
    left, middle, right = st.columns(3)
    left.metric('Model', 'BERT-CNN', 'Hybrid')
    middle.metric('Task', 'Distress', 'Binary')
    right.metric('Data sources', '4-class', 'Mental health')

st.markdown('---')

st.subheader('Analyze a message')

input_panel = st.container()
with input_panel:
    st.markdown(
        """
        <div style='background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(148,163,184,0.2); border-radius: 16px; padding: 1rem 1.1rem; margin-bottom: 0.8rem;'>
            <div style='color: #e2e8f0; font-weight: 600; margin-bottom: 0.5rem;'>Enter your phrase or message</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    user_input = st.text_area(
        label=' ',
        value=st.session_state.get('prompt', ''),
        placeholder='Type a message here, for example: I feel like nobody cares and I don\'t want to keep going...',
        height=200,
        label_visibility='collapsed',
    )

col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    run_btn = st.button('🔍 Analyze message', use_container_width=True, type='primary')

if run_btn:
    if not user_input.strip():
        st.error('Please enter a message before running the model.')
    else:
        with st.spinner('Running the distress model...'):
            cleaned = clean_text(user_input)
            enc = tokenizer(
                cleaned,
                max_length=128,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            with torch.no_grad():
                logits = model(enc['input_ids'], enc['attention_mask'])
                probs = torch.softmax(logits, dim=1)
                distress_prob = float(probs[0][1].item())
                normal_prob = float(probs[0][0].item())

        if distress_prob >= threshold:
            st.markdown(f"""
            <div class="result-card result-danger">
              <h3>🚨 High distress risk detected</h3>
              <p>The model sees strong signs of psychological distress or crisis language in this message.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-safe">
              <h3>✅ Low distress risk</h3>
              <p>The message does not strongly match the distress pattern used by the trained model.</p>
            </div>
            """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        col1.metric('Distress confidence', f'{distress_prob * 100:.1f}%')
        col2.metric('Normal confidence', f'{normal_prob * 100:.1f}%')
        st.progress(min(max(distress_prob, 0.0), 1.0), text=f'Distress likelihood ({distress_prob * 100:.1f}%)')

with st.expander('📊 Model benchmark and training data'):
    st.markdown(
        """
        <div class='dataset-box'>
            <strong>Current source dataset:</strong><br>
            • mental_heath_unbanlanced.csv (main 4-class mental-health corpus)<br>
            • mental_health_combined_test.csv (balanced 4-class validation set)<br>
            • mental_health_feature_engineered.csv (feature-rich text subset)
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.table(
        {
            'Model': ['Proposed (BERT + CNN + Attention)', 'Basyouni et al.', 'Rabani et al.'],
            'Accuracy': ['99.14%', '98.92%', '96.33%'],
            'F1-score': ['99.08%', '—', '—'],
        }
    )

    if model_loaded:
        eval_report = compute_quick_eval()
        if eval_report:
            metrics_df = pd.DataFrame({'Metric': list(eval_report['metrics'].keys()), 'Value': list(eval_report['metrics'].values())})
            st.subheader('Model metrics snapshot')
            st.bar_chart(metrics_df.set_index('Metric'))

            fig, ax = plt.subplots(figsize=(5, 4))
            ax.imshow(eval_report['cm'], cmap='Blues')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Normal', 'Distress'])
            ax.set_yticklabels(['Normal', 'Distress'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(eval_report['cm'][i, j]), ha='center', va='center', color='black', fontsize=12)
            ax.set_title('Confusion matrix')
            st.pyplot(fig)

            st.subheader('Confidence by risk level')
            st.bar_chart(eval_report['confidence_by_level'].set_index('Risk level'))

st.sidebar.markdown('---')
train_button = st.sidebar.button('🚀 Retrain model', use_container_width=True)
if train_button:
    venv_python = os.path.join(os.getcwd(), '.venv', 'Scripts', 'python.exe')
    if not os.path.exists(venv_python):
        venv_python = sys.executable
    with st.sidebar.spinner('Training model... This can take a few minutes.'):
        result = subprocess.run([venv_python, 'src/train.py'], capture_output=True, text=True, cwd=os.getcwd())
    if result.returncode == 0:
        st.sidebar.success('Training finished successfully.')
        st.sidebar.code(result.stdout[-1200:])
    else:
        st.sidebar.error('Training did not complete.')
        st.sidebar.code(result.stderr[-1200:])

st.markdown('---')
st.caption('⚠️ This system is intended for research and screening support only, not as a replacement for professional mental health diagnosis.')

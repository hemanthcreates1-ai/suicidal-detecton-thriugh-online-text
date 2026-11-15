import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import clean_text, load_dataset_summary
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
    loaded = os.path.isfile(model_path)
    if loaded:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, tokenizer, loaded


@st.cache_data
def compute_quick_eval(model_name='bert-base-uncased', sample_size=300):
    # Kept as a placeholder for a future independent-test-set evaluation.
    return None


model, tokenizer, model_loaded = load_model_and_tokenizer()

with st.sidebar:
    st.title('Controls')
    st.caption('Model status')
    if model_loaded:
        st.success('✅ Trained model loaded')
    else:
        st.warning('⚠️ No trained checkpoint found; predictions are disabled.')

    threshold = st.slider('Alert threshold', min_value=0.35, max_value=0.85, value=0.5, step=0.01)
    st.markdown('---')

    try:
        dataset_info = load_dataset_summary(data_dir='data')
    except (FileNotFoundError, ValueError) as exc:
        dataset_info = None
        st.info(f'Dataset summary unavailable: {exc}')
    if dataset_info:
        st.subheader('Dataset coverage')
        st.metric('Rows', dataset_info['rows'])
        st.metric('Distress rate', f"{dataset_info['positive_rate'] * 100:.1f}%")
        st.caption('Label prevalence in the available project CSV files; labels are not clinical ground truth.')

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
    if not model_loaded:
        st.error('A trained best_model.pth checkpoint is required. See the README training instructions.')
    elif not user_input.strip():
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

with st.expander('📊 Model and data notes'):
    st.write('This research prototype has no independently validated benchmark included. See the README for the training split, dataset setup, and limitations.')

st.markdown('---')
st.caption('⚠️ This system is intended for research and screening support only, not as a replacement for professional mental health diagnosis.')

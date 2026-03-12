import pandas as pd
import numpy as np
import torch
import gspread
import gradio as gr
import os
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from google.colab import auth
from google.auth import default

# Constants
model_name = 'roberta-large'
save_path = './trained_roberta_small_fp16' # Using the quantized version for deployment
THEMES = {
    'Lifestyle': 'Lifestyle: everyday life, hobbies, routines, hanging out, personal moments',
    'Travel & Experience': 'Travel & Experience: trips, visiting places, tourism, events, sports games, concerts',
    'Reflection': 'Reflection: deep thoughts, life lessons, gratitude, personal growth, emotional reflection',
    'Brand & Business': 'Brand & Business: marketing, promoting a brand, entrepreneurship, product announcements, advertisements',
    'Education & Value': 'Education & Value: advice, tips, teaching something, sharing knowledge',
    'Relationships': 'Relationships: family, friends, romantic, social connections',
    'Performance & Sport': 'Performance & Sport: competitions, sports achievements, athletic performance'
}

# Logic Init
label_encoder = LabelEncoder()
label_encoder.fit(list(THEMES.keys()))
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(save_path)
model.eval()

# Cloud Connection
auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
SHEET_URL = 'https://docs.google.com/spreadsheets/d/12c1ifGn-U6xhNpJ8v7Q_2uxufQHo6uItzmin9lhNqrw/edit?gid=0#gid=0'
sh = gc.open_by_url(SHEET_URL)

def classify_caption_trained_model(caption):
    inputs = tokenizer(caption, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    label_idx = np.argmax(probs)
    return label_encoder.inverse_transform([label_idx])[0], round(float(probs[label_idx]), 2)

def save_feedback_to_sheets(caption, theme):
    try:
        worksheet = sh.get_worksheet(0)
        worksheet.append_row([caption, theme])
        return f'Success: Feedback for {theme} saved.'
    except Exception as e: return f'Error: {str(e)}'

with gr.Blocks() as deployment_demo:
    gr.Markdown('# Theme Classifier - Inference Portal')
    cap_in = gr.Textbox(label='Caption', lines=3)
    res_out = gr.Textbox(label='AI Prediction', interactive=False)
    conf_out = gr.Textbox(label='Confidence', interactive=False)
    gr.Button('Classify').click(classify_caption_trained_model, inputs=cap_in, outputs=[res_out, conf_out])
    drop_corr = gr.Dropdown(choices=list(THEMES.keys()), label='Correct Theme')
    gr.Button('Submit to Cloud').click(save_feedback_to_sheets, inputs=[cap_in, drop_corr], outputs=gr.Markdown())

deployment_demo.launch()

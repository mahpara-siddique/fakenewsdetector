import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load Model and Tokenizer
model_path = "fake_news_model"  # Update with your file name
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.load_state_dict(torch.load(model_path))
model.eval()

# Streamlit UI
st.title("Fake News Detector")

user_input = st.text_area("Enter news text:")

if st.button("Analyze"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits).item()
    
    result = "Fake News" if prediction == 1 else "Real News"
    st.write(f"Prediction: **{result}**")

import streamlit as st
import torch
import gdown
import os
from transformers import BertTokenizer

# Google Drive file ID (replace with your actual file ID)
file_id = "1nXMWHReXZ25LSuEV95ywut6rcYDb2KkY"
output_model = "fake_news_detector.pth"

# Function to download model
def download_model():
    if not os.path.exists(output_model):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_model, quiet=False)

download_model()

# ✅ Load Pre-trained Tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ✅ Load the Full Model (No need to redefine architecture)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    model = torch.load(output_model, map_location=device)
    model.eval()
    model.to(device)
except RuntimeError as e:
    st.error(f"Model loading error: {str(e)}")
    st.stop()

# ✅ Streamlit UI
st.title("📰 Fake News Detection App")
st.write("Enter a news article to check if it's FAKE or REAL.")

user_input = st.text_area("✍️ Enter text here...")

if st.button("🔍 Predict"):
    if user_input.strip():
        # Tokenize and prepare input
        inputs = tokenizer(user_input, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

        with torch.no_grad():
            prediction = model(inputs["input_ids"], inputs["attention_mask"]).argmax(dim=1).item()

        result = "FAKE NEWS" if prediction == 0 else "REAL NEWS"
        st.success(f"📢 Prediction: **{result}**")
    else:
        st.warning("⚠️ Please enter some text before clicking Predict.")

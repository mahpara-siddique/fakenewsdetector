import streamlit as st
import torch
import gdown
import os

# Google Drive file ID
file_id = "1nXMWHReXZ25LSuEV95ywut6rcYDb2KkY"
output_model = "fake_news_detector.pth"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(output_model):  # Prevent re-downloading
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_model, quiet=False)

# Download and load the model
download_model()

# Define your model class (Must match the architecture used during training)
class FakeNewsModel(torch.nn.Module):
    def __init__(self):
        super(FakeNewsModel, self).__init__()
        self.layer1 = torch.nn.Linear(768, 1)  # Example structure

    def forward(self, x):
        return torch.sigmoid(self.layer1(x))

# Load model
model = FakeNewsModel()
model.load_state_dict(torch.load(output_model, map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
st.title("Fake News Detection App")
st.write("Enter a news article to check if it's fake or real.")

user_input = st.text_area("Enter text here...")

if st.button("Predict"):
    # Dummy feature vector (Replace with actual text preprocessing)
    input_tensor = torch.randn(1, 768)  # Example input shape

    with torch.no_grad():
        prediction = model(input_tensor).item()

    result = "Fake News" if prediction > 0.5 else "Real News"
    st.success(f"Prediction: {result}")

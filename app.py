import streamlit as st
import torch
import torch.nn as nn
import gdown
import os
from transformers import BertTokenizer, BertModel

# Google Drive file ID (replace with your actual file ID)
file_id = "1nXMWHReXZ25LSuEV95ywut6rcYDb2KkY"
output_model = "fake_news_detector.pth"

# Function to download model
def download_model():
    if not os.path.exists(output_model):
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_model, quiet=False)

download_model()

# ✅ Load Pre-trained BERT Model
bert = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# ✅ Define the Correct Model Architecture
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)  
        self.relu = nn.ReLU()  
        self.fc1 = nn.Linear(768, 512)  
        self.fc2 = nn.Linear(512, 2)  
        self.softmax = nn.LogSoftmax(dim=1)  

    def forward(self, sent_id, mask):
        cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
        x = self.fc1(cls_hs)  
        x = self.relu(x)  
        x = self.dropout(x)  
        x = self.fc2(x)  
        x = self.softmax(x)  
        return x

# ✅ Load the Model
model = BERT_Arch(bert)
try:
    model.load_state_dict(torch.load(output_model, map_location=torch.device('cpu')))
    model.eval()
except RuntimeError as e:
    st.error(f"Model loading error: {str(e)}")
    st.stop()

# ✅ Streamlit UI
st.title("Fake News Detection App")
st.write("Enter a news article to check if it's fake or real.")

user_input = st.text_area("Enter text here...")

if st.button("Predict"):
    if user_input.strip():
        # Tokenize and prepare input
        inputs = tokenizer(user_input, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            prediction = model(input_ids, mask=attention_mask).argmax(dim=1).item()

        result = "FAKE NEWS" if prediction == 1 else "REAL NEWS"
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text before clicking Predict.")

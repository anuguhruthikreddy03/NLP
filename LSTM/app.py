import streamlit as st
import pickle
import re
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model
model = load_model("model1.h5")

# Load tokenizer
with open("tokenizers.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 50

# Cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# UI
st.title("🔥 Hate Speech Detection (RNN)")
st.write("Enter a tweet to check if it's hate speech")

user_input = st.text_area("Enter Tweet")

if st.button("Predict"):
    cleaned = clean_text(user_input)
    
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len)
    
    pred = model.predict(padded)[0][0]
    
    if pred > 0.5:
        st.error(f"Hate Speech ❌ ({pred:.2f})")
    else:
        st.success(f"Not Hate Speech ✅ ({pred:.2f})")
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

# Download required resources
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = word_tokenize(text)
    filtered = [ps.stem(i) for i in text if i.isalnum() and i not in stopwords.words('english')]
    return " ".join(filtered)

# ✅ Load fitted TfidfVectorizer and trained model
try:
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    st.error(f"Failed to load model/vectorizer: {e}")

# 🌟 App Title and Input
st.title("📩 Email/SMS Spam Classifier")
input_sms = st.text_area("✉️ Enter your message below:")

if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        prediction = model.predict(vector_input)[0]
        confidence = model.predict_proba(vector_input)[0]

        if prediction == 1:
            st.header("🚫 Spam")
            st.write(f"Confidence: {confidence[1]:.2%}")
        else:
            st.header("✅ Not Spam")
            st.write(f"Confidence: {confidence[0]:.2%}")

        # Debug info (optional)
        st.caption(f"Vector shape: {vector_input.shape}")

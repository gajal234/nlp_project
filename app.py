import re 
import docx
import PyPDF2
import nltk
import pandas as pd
import pickle 
import streamlit as st
from PyPDF2 import PdfReader
from contractions import fix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from io import BytesIO

rf_classifier = pickle.load(open("Day11_nlp\\model\\rf_nlpmodel.pkl","rb"))
tfidf_vectorizer = pickle.load(open("Day11_nlp\\model\\tfidf_nlpmodel.pkl","rb"))

def clean_text(text):
    text = fix(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text) 
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word not in stop_words]
    lemmitizer = WordNetLemmatizer()
    stem_words = [lemmitizer.lemmatize(word) for word in filtered_words]
    final_word = " ".join(stem_words)
    return final_word

def detect_sarcasm(text):
    cln_fun = clean_text(text)
    features = tfidf_vectorizer.transform([cln_fun])
    result = rf_classifier.predict(features)
    if result[0] == 1:
        return "Sarcastic"
    else:
        return "Not Sarcastic"
    
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text.strip()


def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8").strip()



st.sidebar.header("About the Sarcasm Detection App ")
st.sidebar.info("""
**Sarcasm Detection App** allows you to detect sarcasm in text through three methods:
1. Upload a PDF, TXT, or DOCX document, and the app will extract the text and analyze it to check for sarcasm.
2. Enter text directly into the search bar, and the app will detect whether it contains sarcasm.

**Output:** The app provides an easy-to-read prediction of whether the text is 'Sarcastic' or 'Not Sarcastic'. 
Whether you're analyzing customer reviews, social media comments, or any other text, this app simplifies sarcasm detection.
""")

# Streamlit app interface
st.title("Sarcasm Detection App")
st.markdown("This app detects **sarcasm** in text using advanced **machine learning models**. "
            "You can upload **PDF, TXT, DOCX** files or input text directly. Let's see if the text has a sarcastic tone!")
# Footer at the bottom of the sidebar
st.sidebar.markdown("---")

uploaded_file = st.file_uploader("  Upload a file, pdf, or text",type=["docx","pdf","txt"])

search_query = st.text_input("or, directly input the tsxt")

if st.button("Detect Sarcasm"):
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            headline = extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith(".txt"):
            headline = extract_text_from_txt(uploaded_file)
        elif uploaded_file.name.endswith(".docx"):
            headline = extract_text_from_docx(uploaded_file)

        else :
            headline = ""
        if headline:
            result = detect_sarcasm(headline)
            st.success(f"prediction from input text:  ***{result}***")
        else:
            st.warning("No text could be extracted from the uploaded file. Please try another file.")

    elif search_query:
        result = detect_sarcasm(search_query)
        st.success(f"prediction from input text:  ***{result}***")
    else:
        st.error("Please upload a file or enter some text to get a prediction.")
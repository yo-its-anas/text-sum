import streamlit as st
from transformers import pipeline

# Initialize Hugging Face pipeline for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Streamlit interface
st.title("Text Summarization Application")
st.write("Enter some text to summarize:")

text_input = st.text_area("Text to summarize")

if text_input:
    try:
        # Summarize the text using Hugging Face model
        summary = summarizer(text_input, max_length=150, min_length=50, do_sample=False)
        st.subheader("Summary")
        st.write(summary[0]['summary_text'])
    except Exception as e:
        st.error(f"An error occurred: {e}")

           

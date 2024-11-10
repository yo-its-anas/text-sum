# app.py

import streamlit as st
from transformers import pipeline

# Set up the Streamlit app title and introduction text
st.title("Text Summarization App")
st.write("Enter some text below, and the app will summarize it!")

# Initialize the summarization pipeline
@st.cache_resource  # Use caching to avoid reloading the model on each interaction
def load_summarization_pipeline():
    # Using a lightweight model for faster loading and summarization
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)  # Use CPU

# Load the model and pipeline for summarization
summarizer = load_summarization_pipeline()

# Create an input text area for users to enter text
input_text = st.text_area("Input Text", "Enter the text you want to summarize here...")

# Add a button to start summarization
if st.button("Summarize Text"):
    # Check if input text is provided
    if input_text.strip():
        # Use the summarizer to generate a summary
        with st.spinner("Summarizing..."):
            summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
            st.subheader("Summary")
            st.write(summary[0]['summary_text'])
    else:
        st.warning("Please enter some text to summarize.")

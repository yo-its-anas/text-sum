import streamlit as st
from transformers import pipeline

# Cache the Hugging Face pipelines to avoid reloading models every time
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_sentiment_analyzer():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# Load models
summarizer = load_summarizer()
sentiment_analyzer = load_sentiment_analyzer()

# Step 2: Define the summarization function
def summarize_text_huggingface(text):
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

# Step 3: Define the sentiment analysis function
def analyze_sentiment_huggingface(text):
    sentiment = sentiment_analyzer(text)
    return sentiment[0]['label']  # 'POSITIVE' or 'NEGATIVE'

# Step 4: Streamlit App Layout
st.title("Text Summarization and Sentiment Analysis App")

st.markdown("""
    This app uses Hugging Face models to provide text summarization and sentiment analysis. 
    You can input a paragraph of text to get a summary and sentiment analysis.
""")

# User input for text
text_input = st.text_area("Enter the text you want to analyze:")

# Process if user enters some text
if text_input.strip():
    st.subheader("Processing...")
    
    # Get summary
    summary = summarize_text_huggingface(text_input)
    st.subheader("Summary:")
    st.write(summary)
    
    # Get sentiment
    sentiment = analyze_sentiment_huggingface(text_input)
    st.subheader("Sentiment Analysis:")
    st.write(f"Sentiment: {sentiment}")
else:
    st.write("Please enter some text to analyze.")

import streamlit as st
from transformers import pipeline

try:
  # Attempt to load the model
  summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
except RuntimeError as e:
  # Handle model loading error
  st.error(f"Error loading summarization model: {e}")
  # Consider providing alternative options or disabling summarization functionality

def summarize_text(text):
    """Summarizes the input text (if model loaded successfully)"""
    if summarizer:  # Check if model is available
        summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return summary[0]['summary_text']
    else:
        return "Summarization unavailable: Please check model availability."  # Informative message

def main():
    st.title("Text Summarizer and Sentiment Analyzer")

    with st.expander("Enter your text here"):
        text_input = st.text_area("", height=200)

    if st.button("Analyze"):
        if text_input.strip():
            with st.spinner("Processing..."):
                summary = summarize_text(text_input)
                sentiment = analyze_sentiment(text_input)

            st.subheader("Summary")
            st.write(summary)

            st.subheader("Sentiment Analysis")
            st.write(f"Sentiment: {sentiment}")
        else:
            st.error("Please enter some text to analyze.")

if __name__ == "__main__":
    main()
          

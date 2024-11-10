import streamlit as st
from transformers import pipeline

# Initialize models with smaller, faster options
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
sentiment_analyzer = pipeline("sentiment-analysis")

def summarize_text(text):
    """Summarizes the input text"""
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def analyze_sentiment(text):
    """Analyzes the sentiment of the input text"""
    sentiment = sentiment_analyzer(text)
    return sentiment[0]['label']

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

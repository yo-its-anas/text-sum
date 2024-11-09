import streamlit as st
from transformers import pipeline

# Initialize Hugging Face models
summarizer = pipeline("summarization")
sentiment_analyzer = pipeline("sentiment-analysis")

# Set up Streamlit app
st.title("Text Summarizer and Sentiment Analyzer")

# Create a text input box for user input
text_input = st.text_area("Enter the text you want to analyze:", height=300)

# Display buttons and logic to process input
if st.button("Summarize and Analyze Sentiment"):

    if text_input.strip():
        try:
            # Summarize the text
            summary = summarizer(text_input, max_length=150, min_length=50, do_sample=False)
            sentiment = sentiment_analyzer(text_input)

            # Display the results
            st.subheader("Summary:")
            st.write(summary[0]['summary_text'])

            st.subheader("Sentiment Analysis:")
            st.write(f"Sentiment: {sentiment[0]['label']}, with a confidence score of {sentiment[0]['score']:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter some text to analyze.")

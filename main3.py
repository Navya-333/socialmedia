import pickle
from os import path
import numpy as np
import streamlit as st
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st
import pandas as pd
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords', quiet=True)  # Download stopwords if not already present
nltk.download('punkt', quiet=True) # Download punkt for sentence tokenization

def clean_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'@[^\s]+', '', text) # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtags
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation and special characters
    text = text.lower() # Lowercasing
    
    #Tokenization
    words = text.split()
    
    #Stop word removal
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    #Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return " ".join(words)


def analyze_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

st.title("Social Media Sentiment Analysis App")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        # Assuming your CSV has a column named 'text' containing the tweets/posts.
        # Adjust 'text' if your column name is different.
        if 'text' not in df.columns:
            st.error("The CSV file must contain a column named 'text'.")
        else:
            df['cleaned_text'] = df['text'].apply(clean_text) #Clean the text data
            df['sentiment'] = df['cleaned_text'].apply(analyze_sentiment)

            st.write(df.head()) # Display the first few rows with sentiment

            # Display sentiment distribution
            sentiment_counts = df['sentiment'].value_counts()
            st.bar_chart(sentiment_counts)


            #Option to download the updated CSV
            st.download_button(
                label="Download updated CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='sentiment_analyzed_data.csv',
                mime='text/csv',
            )


    except Exception as e:
        st.error(f"An error occurred: {e}")

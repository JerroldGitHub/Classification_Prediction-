import streamlit as st
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import sys
from wc import WordCloud 
import matplotlib.pyplot as plt

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function for NLP preprocessing
def nlp_preprocess(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [word.lower() for word in tokens]
    # Remove punctuation
    words = [word for word in tokens if word.isalpha()]
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    # Stemming
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in words]
    return stemmed

# Function for extracting keywords
def extract_keywords(tokens):
    freq_dist = nltk.FreqDist(tokens)
    keywords = [word for word, freq in freq_dist.items() if freq > 1]
    return keywords

# Function for sentiment analysis
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment

# Function to plot sentiment analysis
def plot_sentiment(sentiment):
    fig, ax = plt.subplots()
    ax.bar(['Polarity', 'Subjectivity'], [sentiment.polarity, sentiment.subjectivity])
    return fig

# Function to create a word cloud
def plot_wordcloud(text):
    wordcloud = WordCloud(width = 800, height = 800, 
                          background_color ='white', 
                          stopwords = set(stopwords.words('english')), 
                          min_font_size = 10).generate(text)
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Streamlit app layout
st.title('NLP Text Processor and Analyzer')

# Text input
text = st.text_area("Input your text here")

if st.button('Process Text'):
    if text:
        preprocessed_text = nlp_preprocess(text)
        st.subheader('NLP Preprocess')
        # Using st.text_area for scrollable text display
        st.text_area('Tokenized, lowercased, and punctuation removed:', ', '.join(preprocessed_text), height=150)

        # Extract keywords
        keywords = extract_keywords(preprocessed_text)
        st.subheader('Keywords')
        # Using st.code for a code-like box display
        st.code(', '.join(keywords))

        # Sentiment analysis
        sentiment = analyze_sentiment(text)
        st.subheader('Sentiment Analysis')
        st.write('Polarity:', sentiment.polarity)
        st.write('Subjectivity:', sentiment.subjectivity)

        # Plot sentiment analysis
        st.subheader('Sentiment Analysis Plot')
        st.pyplot(plot_sentiment(sentiment))

        # Create and display word cloud
        st.subheader('Word Cloud')
        st.pyplot(plot_wordcloud(text))
    else:
        st.warning("Please input some text for processing.")
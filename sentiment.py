import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from newsapi import NewsApiClient
import requests
from datetime import datetime, timedelta
from transformers import BertForSequenceClassification, DistilBertForSequenceClassification, BertTokenizer, DistilBertTokenizer
import torch
import tensorflow as tf
from transformers import RobertaForSequenceClassification, RobertaTokenizer
tf.compat.v1.reset_default_graph()

# Load environment variables
load_dotenv()

# Check if environment variables are loaded
if not all([os.getenv("GEMINI_API_KEY"), os.getenv("newsapi_key"), os.getenv("GNEWS_API_KEY")]):
    st.error("Missing one or more API keys in .env file. Please ensure GEMINI_API_KEY, newsapi_key, and GNEWS_API_KEY are set.")

# Configure the Google Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Generation configuration for Gemini
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 300,
    "response_mime_type": "text/plain",
}

# Initialize the generative model for Google Gemini
try:
    model_gemini = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest",
        safety_settings=[],
        generation_config=generation_config,
    )
except Exception as e:
    st.error(f"Error initializing Google Gemini model: {e}")

# Initialize NewsApiClient
newsapi = NewsApiClient(api_key=os.getenv("newsapi_key"))

# Function to fetch articles from GNews
def fetch_gnews_articles(topic):
    gnews_url = f"https://gnews.io/api/v4/search?q={topic}&token={os.getenv('GNEWS_API_KEY')}&lang=en"
    response = requests.get(gnews_url)
    if response.status_code == 200:
        return response.json().get('articles', [])
    else:
        st.error(f"Error fetching GNews articles: {response.status_code}")
        return []

# Define model paths for BERT and DistilBERT
bert_model_path = './saved_model_BERT'
distilbert_model_path = './saved_model_DistilBert'
roberta_model_path = './saved_model_RoBerta'
# Load BERT and DistilBERT models and tokenizers with error handling
try:
    # Load BERT model and tokenizer
    tokenizer_bert = BertTokenizer.from_pretrained(bert_model_path)
    model_bert = BertForSequenceClassification.from_pretrained(bert_model_path)

    # Load DistilBERT model and tokenizer
    tokenizer_distilbert = DistilBertTokenizer.from_pretrained(distilbert_model_path)
    model_distilbert = DistilBertForSequenceClassification.from_pretrained(distilbert_model_path)
    
     # Load RoBERTa model and tokenizer
    tokenizer_roberta = RobertaTokenizer.from_pretrained(roberta_model_path)
    model_roberta = RobertaForSequenceClassification.from_pretrained(roberta_model_path)

    model_roberta.eval()
    model_bert.eval()
    model_distilbert.eval()
except Exception as e:
    st.error(f"Error loading models or tokenizers: {e}")

# Streamlit app
st.title('News Sentiment Analysis with BERT and Gemini')
st.write('Fetch articles, summarize them, and analyze their sentiment.')

# User input for topic
topic = st.text_input('Enter topic for news articles (e.g., bitcoin)')

# Dropdown for news source selection
source_option = st.selectbox("Select news source", options=["NewsAPI", "GNews"])


# Fetch articles based on user input
def fetch_articles(topic, source):
    today = datetime.now().strftime('%Y-%m-%d')
    three_days_ago = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    
    if source == "NewsAPI":
        try:
            articles = newsapi.get_everything(
                q=topic,
                language='en',
                from_param=three_days_ago,
                to=today,
                sort_by='relevancy'
            ).get('articles', [])
        except Exception as e:
            st.error(f"Error fetching articles from NewsAPI: {e}")
            articles = []
    else:
        articles = fetch_gnews_articles(topic)

    return articles[:5]  # Limit to top 5 articles

# Summarize text using Gemini
def summarize_text(text):
    try:
        chat_session = model_gemini.start_chat(history=[])
        message = f"Summarize briefly with less than 50 words: {text}"
        response = chat_session.send_message(message)  # No max_output_tokens here
        return response.text.strip()
    except Exception as e:
        st.error(f"Error during summarization with Gemini: {e}")
        return "Summary unavailable."

# Analyze sentiment using the selected model
def analyze_sentiment(text, model_option):
    if model_option == "BERT":
        tokenizer = tokenizer_bert
        model = model_bert
    elif model_option == "DistilBERT":
        tokenizer = tokenizer_distilbert
        model = model_distilbert
    else:  # For RoBERTa
        tokenizer = tokenizer_roberta
        model = model_roberta
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).item()
    sentiment_mapping = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiment_mapping[predictions]

# Dropdown to choose model (BERT, DistilBERT, or RoBERTa)
model_option = st.selectbox("Select Model", options=["BERT", "DistilBERT", "RoBERTa"])


# Button to fetch and display articles
if st.button('Fetch Articles'):
    with st.spinner('Fetching articles...'):
        articles = fetch_articles(topic, source_option)
    
    # Check if no articles were found
    if not articles:
        st.warning("No articles found for the specified topic.")
    else:
        for idx, article in enumerate(articles):
            title = article.get('title', 'No Title')
            description = article.get('description', '')
            content = article.get('content', '') or description
            url = article.get('url', '#')  # Use '#' if URL is not available

            # Summarize the article
            summary = summarize_text(content)
            
            # Analyze sentiment using the selected model
            sentiment = analyze_sentiment(summary, model_option)

            # Display the article details
            st.markdown(f"### Article {idx + 1}: [{title}]({url})")  # Title as a clickable link
            st.markdown(f"**Summary:** {summary}")
            st.markdown(f"**Sentiment ({model_option}):** {sentiment}")

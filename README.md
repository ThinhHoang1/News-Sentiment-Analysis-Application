# News Sentiment Analysis Application

This application is a sentiment analysis tool designed to fetch, summarize, and analyze news articles on specific topics. Users enter a topic and select a news source (NewsAPI or GNews) to retrieve relevant articles. Google Gemini generates summaries, and pre-trained BERT models analyze sentiment to provide insight into the sentiment for each article.

## Features

- **Fetch articles** from NewsAPI or GNews based on a topic.
- **Summarize articles** using Google Gemini.
- **Sentiment analysis** using BERT, DistilBERT, or RoBERTa models with sentiment classified as positive, neutral, or negative.
- **Simple UI** using Streamlit for easy interaction and display.

## Requirements

- Python 3.8 or above
- [Streamlit](https://streamlit.io/)
- [NewsAPI](https://newsapi.org/docs/get-started)
- [GNews API](https://gnews.io/docs/)
- [Google Gemini](https://developers.generative.google/) access
- [Transformers library](https://huggingface.co/transformers/) from Hugging Face for BERT models

## Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ThinhHoang1/news-sentiment-analysis.git
   cd news-sentiment-analysis

## Install Dependencies
```bash
   pip install -r requirements.txt
```

## Set Up Environment Variables
- Create a .env file in the project root and add your API keys:
  ``` python
  GEMINI_API_KEY=your_gemini_api_key
  newsapi_key=your_newsapi_key
  GNEWS_API_KEY=your_gnews_api_key
  ```

## Download Pre-trained Models
- Save pre-trained models (BERT, DistilBERT, RoBERTa) to ./saved_model_BERT, ./saved_model_DistilBert, and ./saved_model_RoBerta directories, respectively.
- You can download these models from: https://drive.google.com/file/d/1Pws-KjvX_0OYgF5Qj_qzy9D2jEAtasxq/view?usp=sharing

## Running the Application
``` bash
streamlit run app.py
```

## Demo application

![Demo App](https://github.com/ThinhHoang1/stock-recommendation/blob/main/Demo/News_sentiment.png)



  




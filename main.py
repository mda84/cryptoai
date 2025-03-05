# main.py

import streamlit as st
import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import clip
from PIL import Image
import os
import pandas as pd
from pycoingecko import CoinGeckoAPI
import mplfinance as mpf
from datetime import datetime, timedelta
import tempfile

# Ensure the VADER lexicon is available
nltk.download('vader_lexicon')

@st.cache(allow_output_mutation=True)
def load_llm_model():
    model = AutoModelForSequenceClassification.from_pretrained("./models/llm_finetuned")
    tokenizer = AutoTokenizer.from_pretrained("./models/llm_finetuned")
    return model, tokenizer

@st.cache(allow_output_mutation=True)
def load_vlm_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.load_state_dict(torch.load("./models/vlm_finetuned.pth", map_location=device))
    model.to(device)
    model.eval()
    return model, preprocess, device

def sentiment_analysis(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    logits = outputs.logits.detach().cpu().numpy()
    sentiment = "Positive" if logits.argmax() == 1 else "Negative"
    return sentiment

def fetch_latest_news():
    rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries:
        text = entry.title + ". " + entry.summary
        articles.append(text)
    return articles

def fetch_bitcoin_data(days=30):
    cg = CoinGeckoAPI()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = cg.get_coin_market_chart_range_by_id(id='bitcoin', vs_currency='usd',
                                                from_timestamp=start_date.timestamp(),
                                                to_timestamp=end_date.timestamp())
    prices = data['prices']
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    daily = df.resample('1D').agg({'price': ['first', 'max', 'min', 'last']})
    daily.columns = ['Open', 'High', 'Low', 'Close']
    daily = daily.dropna()
    return daily

def generate_chart(df):
    # Generate a candlestick chart and save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    mpf.plot(df, type='candle', style='charles', title="Bitcoin Candlestick Chart", savefig=temp_file.name)
    return temp_file.name

def classify_chart_image(image, vlm_model, preprocess, device):
    image_input = preprocess(image).unsqueeze(0).to(device)
    # Classify using text tokens "bullish" and "bearish"
    text_inputs = clip.tokenize(["bullish", "bearish"]).to(device)
    with torch.no_grad():
        image_features = vlm_model.encode_image(image_input)
        text_features = vlm_model.encode_text(text_inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (image_features @ text_features.t()) * 100.0
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        label = "bullish" if probs.argmax() == 0 else "bearish"
    return label, probs

def main():
    st.title("CryptoAI Market Analysis Dashboard")
    
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["News Sentiment", "Chart Analysis", "Market Data"])
    
    if page == "News Sentiment":
        st.header("Latest Crypto News Sentiment Analysis")
        news = fetch_latest_news()
        st.write("Fetched {} articles from Coindesk.".format(len(news)))
        llm_model, tokenizer = load_llm_model()
        for i, article in enumerate(news[:5]):
            sentiment = sentiment_analysis(article, llm_model, tokenizer)
            st.subheader(f"Article {i+1}")
            st.write(article)
            st.write(f"Sentiment: **{sentiment}**")
    
    elif page == "Chart Analysis":
        st.header("Bitcoin Chart Pattern Analysis")
        days = st.slider("Select number of days for chart", min_value=7, max_value=60, value=30)
        df = fetch_bitcoin_data(days=days)
        chart_path = generate_chart(df)
        st.image(chart_path, caption="Bitcoin Candlestick Chart", use_column_width=True)
        vlm_model, preprocess, device = load_vlm_model()
        image = Image.open(chart_path).convert("RGB")
        label, probs = classify_chart_image(image, vlm_model, preprocess, device)
        st.write(f"Predicted Pattern: **{label}**")
        st.write("Confidence Scores (bullish, bearish): ", probs)
    
    elif page == "Market Data":
        st.header("Bitcoin Market Data")
        days = st.slider("Select number of days", min_value=30, max_value=180, value=90)
        df = fetch_bitcoin_data(days=days)
        st.write("Bitcoin Price Data")
        st.dataframe(df)
        st.line_chart(df[['Open', 'Close']])
    
if __name__ == "__main__":
    main()

# CryptoAI: Multi-Modal Crypto Market Analysis Platform

CryptoAI is a project that integrates fine-tuning of Large Language Models (LLM) and Vision-Language Models (VLM) to analyze the cryptocurrency market. It fetches real-time crypto news and market data, fine-tunes models for sentiment analysis and chart pattern recognition, and presents insights via an interactive dashboard.

## Features

- **LLM Fine-Tuning:**  
  - Fetches recent crypto news from Coindesk's RSS feed.
  - Uses NLTK's VADER to label news articles by sentiment.
  - Fine-tunes a DistilBERT model on this data for sentiment classification.

- **VLM Fine-Tuning:**  
  - Downloads Bitcoin historical market data from CoinGecko.
  - Generates candlestick charts with mplfinance.
  - Fine-tunes OpenAI's CLIP model to classify charts as "bullish" or "bearish".

- **Dashboard:**  
  - Displays the latest news with sentiment analysis.
  - Shows a Bitcoin candlestick chart generated from real market data.
  - Uses the fine-tuned CLIP model to classify chart patterns.
  - Provides interactive data visualizations of market trends.

## Project Structure

- `finetune_llm.py`: Script to fetch crypto news, label articles using VADER, and fine-tune a DistilBERT model.
- `finetune_vlm.py`: Script to fetch Bitcoin data from CoinGecko, generate candlestick charts, and fine-tune the CLIP model.
- `main.py`: Streamlit dashboard that integrates both models and visualizes real-time crypto data.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Containerization for deployment.

## Installation

Clone the repository:
```   
git clone https://github.com/mda84/cryptoai.git
cd cryptoai
```

Install dependencies:

Make sure you have Python 3.7+ installed, then run:
```   
pip install --upgrade pip
pip install -r requirements.txt
```   

Download NLTK VADER lexicon (if not already downloaded):
```   
python -c "import nltk; nltk.download('vader_lexicon')"
```   

## Usage
1. Fine-Tune Models
Before launching the dashboard, fine-tune both the LLM and VLM models:
```   
python finetune_llm.py
python finetune_vlm.py
```   
These scripts will fetch real data, train the models, and save them in the ./models directory.

2. Launch the Dashboard
Once the models are fine-tuned, start the Streamlit dashboard:
```   
streamlit run main.py
```   

The dashboard offers multiple pages for:

News Sentiment: Displaying the latest crypto news and sentiment analysis.
Chart Analysis: Showing a generated Bitcoin candlestick chart with pattern classification.
Market Data: Visualizing Bitcoin market data with interactive charts.

## Docker Deployment
You can also build and run the project in a Docker container. See the Dockerfile for details.

Build the Docker Image
```   
docker build -t cryptoai .
```   

Run the Docker Container
```   
docker run -p 8501:8501 cryptoai
```   
Then, open your browser and navigate to http://localhost:8501.

## Notes
Adjust the number of training epochs and batch sizes in the scripts as needed.
The project uses real data sources (Coindesk and CoinGecko). Ensure you have an active internet connection.
For production use, consider robust error handling and logging.
Enjoy exploring CryptoAI!
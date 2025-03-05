# finetune_llm.py

import feedparser
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datasets
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

def fetch_crypto_news():
    rss_url = "https://www.coindesk.com/arc/outboundfeeds/rss/"
    feed = feedparser.parse(rss_url)
    articles = []
    for entry in feed.entries:
        # Combine title and summary for analysis
        text = entry.title + ". " + entry.summary
        articles.append(text)
    return articles

def label_articles(articles):
    sia = SentimentIntensityAnalyzer()
    data = []
    for text in articles:
        sentiment = sia.polarity_scores(text)
        compound = sentiment['compound']
        # Binary label: 1 if non-negative, 0 if negative
        label = 1 if compound >= 0 else 0
        data.append({'text': text, 'label': label})
    return data

def fine_tune_llm():
    print("Fetching crypto news...")
    articles = fetch_crypto_news()
    data = label_articles(articles)
    print(f"Fetched and labeled {len(data)} articles.")
    
    # Create a Hugging Face dataset
    ds = Dataset.from_list(data)
    ds = ds.train_test_split(test_size=0.2, seed=42)
    
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)
    
    tokenized_ds = ds.map(tokenize_function, batched=True)
    
    training_args = TrainingArguments(
        output_dir="./models/llm_finetuned",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["test"],
    )
    
    print("Starting LLM fine-tuning...")
    trainer.train()
    model.save_pretrained("./models/llm_finetuned")
    tokenizer.save_pretrained("./models/llm_finetuned")
    print("LLM fine-tuning complete and model saved to ./models/llm_finetuned")

if __name__ == "__main__":
    fine_tune_llm()

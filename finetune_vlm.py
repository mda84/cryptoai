# finetune_vlm.py

import os
import pandas as pd
import mplfinance as mpf
from pycoingecko import CoinGeckoAPI
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset, DataLoader
import clip  # Install via: pip install git+https://github.com/openai/CLIP.git
from PIL import Image
import torchvision.transforms as transforms

# Directory to save generated chart images
CHART_DIR = "data/charts"
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

def fetch_bitcoin_data(days=60):
    cg = CoinGeckoAPI()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data = cg.get_coin_market_chart_range_by_id(id='bitcoin', vs_currency='usd',
                                                from_timestamp=start_date.timestamp(),
                                                to_timestamp=end_date.timestamp())
    prices = data['prices']  # each element: [timestamp, price]
    df = pd.DataFrame(prices, columns=['timestamp', 'price'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    # Group into daily OHLC (approximated)
    daily = df.resample('1D').agg({'price': ['first', 'max', 'min', 'last']})
    daily.columns = ['Open', 'High', 'Low', 'Close']
    daily = daily.dropna()
    return daily

def generate_candlestick_charts(df, chart_period=7):
    # Create non-overlapping charts for each chart_period days
    chart_files = []
    for i in range(0, len(df) - chart_period + 1, chart_period):
        df_chunk = df.iloc[i:i+chart_period]
        start_date = df_chunk.index[0].strftime("%Y-%m-%d")
        end_date = df_chunk.index[-1].strftime("%Y-%m-%d")
        file_name = f"bitcoin_{start_date}_to_{end_date}.png"
        file_path = os.path.join(CHART_DIR, file_name)
        mpf.plot(df_chunk, type='candle', style='charles', title=f"BTC {start_date} to {end_date}",
                 savefig=file_path)
        chart_files.append((file_path, df_chunk))
    return chart_files

class ChartDataset(Dataset):
    def __init__(self, chart_files, transform=None):
        self.chart_files = chart_files
        self.transform = transform
        
    def __len__(self):
        return len(self.chart_files)
    
    def __getitem__(self, idx):
        file_path, df_chunk = self.chart_files[idx]
        image = Image.open(file_path).convert("RGB")
        # Label: bullish if final close > initial open; otherwise bearish
        label = 1 if df_chunk['Close'].iloc[-1] > df_chunk['Open'].iloc[0] else 0
        if self.transform:
            image = self.transform(image)
        return image, label

def fine_tune_vlm():
    print("Fetching Bitcoin market data...")
    df = fetch_bitcoin_data(days=60)
    chart_files = generate_candlestick_charts(df, chart_period=7)
    print(f"Generated {len(chart_files)} candlestick charts.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Create dataset with real chart images
    transform = transforms.Compose([preprocess])
    dataset = ChartDataset(chart_files, transform=transform)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    model.train()
    print("Starting VLM fine-tuning...")
    for epoch in range(3):  # Fine-tune for 3 epochs
        for images, labels in dataloader:
            images = images.to(device)
            labels = torch.tensor(labels).to(device)
            # Map labels to text tokens: 1 -> "bullish", 0 -> "bearish"
            texts = ["bullish" if label == 1 else "bearish" for label in labels.cpu().numpy()]
            text_inputs = clip.tokenize(texts).to(device)
            
            image_features = model.encode_image(images)
            text_features = model.encode_text(text_inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute similarity and loss
            logits_per_image = (image_features @ text_features.t()) * 100.0
            loss = torch.nn.functional.cross_entropy(logits_per_image, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch} Loss: {loss.item()}")
    
    torch.save(model.state_dict(), "./models/vlm_finetuned.pth")
    print("VLM fine-tuning complete and model saved to ./models/vlm_finetuned.pth")

if __name__ == "__main__":
    fine_tune_vlm()

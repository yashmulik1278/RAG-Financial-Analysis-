import os
import requests
from datetime import datetime, timedelta
import json

API_KEY = os.getenv("NEWS_API_KEY")
NEWS_DIR = "data/forex_news"

def fetch_forex_news():
    """Fetch latest forex news from free API and save locally."""
    url = f"https://newsapi.org/v2/everything?q=forex&apiKey={API_KEY}"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    # Clean old news (>2 days)
    for file in os.listdir(NEWS_DIR):
        file_path = os.path.join(NEWS_DIR, file)
        if os.path.getmtime(file_path) < (datetime.now() - timedelta(days=2)).timestamp():
            os.remove(file_path)
    
    # Save new articles
    for article in articles:
        date_str = datetime.fromisoformat(article['publishedAt']).strftime("%Y-%m-%d")
        filename = f"{date_str}_{article['source']['name']}.txt"
        filepath = os.path.join(NEWS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:  # Add encoding='utf-8'
            f.write(f"Title: {article['title']}\n")
            f.write(f"Source: {article['source']['name']}\n")
            f.write(f"Date: {article['publishedAt']}\n")
            f.write(f"Content: {article['content']}\n")

if __name__ == "__main__":
    os.makedirs(NEWS_DIR, exist_ok=True)
    fetch_forex_news()
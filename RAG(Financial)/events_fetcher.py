import os
import requests
from datetime import datetime, timedelta, timezone
import json

EVENTS_DIR = "data/economic_events"

def fetch_economic_events():
    """Fetch economic events using News API (free tier)."""
    url = "https://newsapi.org/v2/everything?q=forex&apiKey=362ae2e9fe1d496a8c7f75da4499433f"
    response = requests.get(url)
    articles = response.json().get('articles', [])
    
    # Clean old events (>1 week)
    for file in os.listdir(EVENTS_DIR):
        file_path = os.path.join(EVENTS_DIR, file)
        if "date" in file and datetime.fromisoformat(file.split("_")[0]) < datetime.now(timezone.utc) - timedelta(days=7):
            os.remove(file_path)
    
    # Save new events
    for article in articles:
        event_date = datetime.fromisoformat(article['publishedAt']).replace(tzinfo=timezone.utc)
        if event_date > datetime.now(timezone.utc):  # Compare using UTC time
            filename = f"{event_date.strftime('%Y-%m-%d')}_{article['source']['name']}.json"
            filepath = os.path.join(EVENTS_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(article, f)

if __name__ == "__main__":
    os.makedirs(EVENTS_DIR, exist_ok=True)
    fetch_economic_events()
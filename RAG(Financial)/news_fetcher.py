import os
import requests
import json
from datetime import datetime, timedelta, timezone
import re

API_KEY = os.getenv("NEWS_API_KEY")

def extract_currency_pairs(text):
    patterns = [
        r'\b[A-Z]{3}/[A-Z]{3}\b',  
        r'\b([A-Z]{3})([A-Z]{3})\b'  
    ]
    pairs = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if isinstance(match, tuple):
                pairs.append(f"{match[0]}/{match[1]}")
            else:
                pairs.append(match)
    return list(set(pairs))  # Remove duplicates

def fetch_forex_news(news_dir):
    API_KEY = os.getenv("NEWS_API_KEY")
    if not API_KEY:
        print("NEWS_API_KEY not found. Please set it in your environment.")
        return
    
    current_time_utc = datetime.now(timezone.utc)
    cutoff_date_utc = current_time_utc - timedelta(days=2)
    start_date = cutoff_date_utc.isoformat(timespec='seconds')  # Use exact cutoff time
    end_date = current_time_utc.isoformat(timespec='seconds')

    # More targeted query for forex-related news
    query_terms = 'forex OR "foreign exchange" OR "currency trading"'
    url = f"https://newsapi.org/v2/everything?q={query_terms}&from={start_date}&to={end_date}&sortBy=publishedAt&apiKey={API_KEY}"
    
    print(f"Fetching news from URL: {url}")
    response = requests.get(url)
    
    # Check for HTTP errors
    if response.status_code != 200:
        print(f"NewsAPI request failed with status {response.status_code}: {response.text}")
        return
    
    data = response.json()
    
    # Check API error status
    if data.get('status') != 'ok':
        print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
        return
    
    articles = data.get('articles', [])
    if not articles:
        print("No articles found in the specified date range.")
        return
    
    saved_count = 0

    for article in articles:
        try:
            published_at = article.get('publishedAt')
            if not published_at:
                continue  # Skip articles without publication date
            
            # Parse publication date correctly
            article_date = datetime.fromisoformat(published_at.replace('Z', '+00:00')).astimezone(timezone.utc)
            
            # Skip articles older than the cutoff
            if article_date < cutoff_date_utc:
                continue
            
            # Generate unique filename using publication timestamp
            timestamp = article_date.strftime('%Y%m%d_%H%M%S')
            filename = f"forex_news_{timestamp}.json"
            filepath = os.path.join(news_dir, filename)
            
            if os.path.exists(filepath):
                continue  # Skip already saved articles
            
            # Extract content from article (fallback to description if needed)
            content = article.get('content', '') or article.get('description', '')
            pairs = extract_currency_pairs(content)
            
            # Save article data
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    'title': article.get('title', ''),
                    'content': content,
                    'pairs': pairs,
                    'source': article.get('source', {}).get('name', 'Unknown'),
                    'publishedAt': published_at,
                    'url': article.get('url', '')
                }, f, ensure_ascii=False)
            
            saved_count += 1
            print(f"Saved: {filename}")
            
        except Exception as e:
            print(f"Error processing article: {str(e)}")
            continue

    # print(f"Saved {saved_count} new news articles.")

if __name__ == "__main__":
    os.makedirs("data/forex_news", exist_ok=True)
    fetch_forex_news("data/forex_news")
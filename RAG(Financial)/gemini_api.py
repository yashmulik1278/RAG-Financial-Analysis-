import os
import requests
import tiktoken
from dotenv import load_dotenv

load_dotenv()

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_AI_STUDIO_API_KEY")
        self.base_url = 'https://generativelanguage.googleapis.com/v1beta'
        self.model = 'gemini-1.5-flash'

    def trim_context(self, docs):
        tokenizer = tiktoken.get_encoding('cl100k_base')
        combined = '\n'.join(docs)
        encoded = tokenizer.encode(combined)
        max_tokens = 32768
        return tokenizer.decode(encoded[:max_tokens]) if encoded else ''

    def generate_answer(self, query, context_docs):
        if not self.api_key:
            return "API key not found."
        
        context = self.trim_context(context_docs)
        if not context:
            return "No valid context found."

        prompt = f"Answer the query \"{query}\" using the context:\n\n{context}\n\nYour response (start of reply):"

        headers = {"Content-Type": "application/json"}
        endpoint = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generation_config": {
                "temperature":0.7,
                "max_output_tokens": 8192
            }
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            candidates = response_json.get('candidates', [])
            if candidates:
                content = candidates[0].get('content', {})
                return content.get('parts', [{}])[0].get('text', 'No answer.')
            else:
                return 'No answer.'
        except Exception as e:
            return f"Error generating answer: {str(e)}"
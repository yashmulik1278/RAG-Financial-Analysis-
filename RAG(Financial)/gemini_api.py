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
        self.max_tokens = 2048

    def trim_context(self, docs):
        """Trim context to fit within Gemini’s token limit."""
        tokenizer = tiktoken.get_encoding('gpt2')
        combined = '\n'.join(docs)
        encoded = tokenizer.encode(combined)
        # Return the first 2000 tokens or as much as possible
        return tokenizer.decode(encoded[:self.max_tokens]) if encoded else ''

    def generate_answer(self, query, context_docs):
        """Generate an answer using Google Gemini API with strict context."""
        if not self.api_key:
            return "API key not found. Set the GOOGLE_AI_STUDIO_API_KEY."

        trimmed_context = self.trim_context(context_docs)
        if not trimmed_context:
            return "No valid context found for the query."

        prompt = f"""
            Answer the following financial query using ONLY the provided context.

            Context:
            {trimmed_context}

            Question:
            {query}

            Answer:
        """

        headers = {"Content-Type": "application/json"}
        endpoint = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        try:
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            return response_json.get(
                'candidates', [{}]
            )[0].get('content', {}).get('parts', [{}])[0].get('text', 'No answer available.')
        except Exception as e:
            return f"Error generating answer: {e}"
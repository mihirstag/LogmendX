import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class OllamaRouter:
    def __init__(self):
        self.base_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
        self.model = os.getenv("LOGRESP_MODEL", "qwen3.8:27b")
        self.chat_endpoint = f"{self.base_url}/api/chat"
        
    def generate_response(self, system_prompt, user_prompt, require_json=False):
        print(f"      [LLM] Querying {self.model}... (Please wait)") # <-- Added this so you know it's working
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_ctx": 8192,     # <-- REDUCED FROM 32768 TO SAVE VRAM
                "num_gpu": 100
            }
        }   
        
        if require_json:
            payload["format"] = "json"

        try:
            response = requests.post(self.chat_endpoint, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["message"]["content"]
        
        except requests.exceptions.RequestException as e:
            print(f"[Error] Failed to connect to Ollama: {e}")
            return None
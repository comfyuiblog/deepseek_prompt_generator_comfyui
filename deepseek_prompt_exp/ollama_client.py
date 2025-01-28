import requests
import json

class OllamaClient:
    def __init__(self, host):
        self.host = host.rstrip('/')

    def list_models(self):
        try:
            response = requests.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []

    def generate(self, model, system_prompt, user_prompt, temperature=0.7, seed=None):
        url = f"{self.host}/api/generate"
        
        data = {
            "model": model,
            "prompt": user_prompt,
            "system": system_prompt,
            "temperature": temperature,
        }
        if seed is not None:
            data["seed"] = seed

        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                # Ollama streams responses, we need to combine them
                full_response = ""
                for line in response.text.split('\n'):
                    if line:
                        try:
                            data = json.loads(line)
                            if 'response' in data:
                                full_response += data['response']
                        except json.JSONDecodeError:
                            continue
                return full_response
            return "Error: Failed to generate response"
        except Exception as e:
            return f"Error: {str(e)}"
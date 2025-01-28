from .exp_config import APIConfig
from .ollama_client import OllamaClient
from openai import OpenAI
import random
import re

class PromptGeneratorNode:
    def __init__(self):
        self.previous_input = None
        self.previous_output = None
        self.config = APIConfig()
        self.ollama_client = OllamaClient(self.config.ollama_host)
        self.cached_models = self.ollama_client.list_models()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["deepseek", "ollama"], {"default": "ollama"}),
                "ollama_model": (cls().cached_models, {"default": cls().cached_models[0] if cls().cached_models else None}) if cls().cached_models else {"default": None},
                "system_prompt": (
                    "STRING", {
                        "default": "You are a creative writer. Your task is to expand the given prompt into a detailed stable diffusion prompt. Output only the expanded prompt without any additional text, tags, or explanations.",
                        "multiline": True
                    }
                ),
                "prompt": (
                    "STRING", {
                        "default": "",
                        "multiline": True
                    }
                ),
                "seed": (
                    "INT", {
                        "default": -1,
                        "min": -1,
                        "max": 100_000_000_000
                    }
                ),
                "min_char": (
                    "INT", {
                        "default": 1000,
                        "min": 10,
                        "max": 5000,
                    }
                ),
                "max_char": (
                    "INT", {
                        "default": 3000,
                        "min": 10,
                        "max": 10000,
                    }
                ),
                "output_widget": (
                    "STRING", {
                        "default": 'Final prompt will display here. \n\nSet minimum and maximum characters to control output length. 1000 min and 3000 max is a good range to start.',
                        "multiline": True,
                        "forceInput": False,
                        "readonly": True,
                    }
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_prompt",)
    FUNCTION = "process"
    CATEGORY = "DeepSeek Prompt Expansion"
    OUTPUT_NODE = True

    def process(
            self,
            model_type: str,
            ollama_model: str,
            system_prompt: str,
            prompt: str,
            seed: int,
            min_char: int,
            max_char: int,
            output_widget: str,
    ):
        if model_type == "deepseek" and not self.config.api_token:
            return {"ui": {"output_widget": "Please configure your DeepSeek API token in config.ini first"}, "result": ("",)}

        if not seed or seed == -1:
            seed = random.randint(0, 100_000_000_000)

        # Add variation to prevent repetition
        prompt = self._add_variation(prompt)
       
        # Generate character count requirement
        char_count = random.randint(min_char, max_char)
        user_content = (
            f"Generate a detailed stable diffusion prompt of exactly {char_count} characters based on this input: \"{prompt}\". "
            "Output only the expanded prompt without any additional text, tags, or explanations."
        )

        # Process with selected model
        if model_type == "deepseek":
            output = self._process_deepseek(system_prompt, user_content, seed)
        else:
            output = self._process_ollama(ollama_model, system_prompt, user_content, seed)

        # Clean the output
        output = self._clean_output(output)

        # Store for future variation
        self.previous_input = prompt
        self.previous_output = output

        return {"ui": {"output_widget": output}, "result": (output,)}

    def _process_deepseek(self, system_prompt, user_content, seed):
        try:
            client = OpenAI(api_key=self.config.api_token, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                stream=False,
                temperature=1.5,
                seed=seed,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error with DeepSeek API: {str(e)}"

    def _process_ollama(self, model, system_prompt, user_content, seed):
        if not model:
            return "Error: No Ollama models available. Please install models using 'ollama pull <model_name>'"
       
        try:
            return self.ollama_client.generate(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_content,
                temperature=1.5,
                seed=seed
            ).strip()
        except Exception as e:
            return f"Error with Ollama: {str(e)}"

    def _add_variation(self, prompt):
        """Add variation to prevent repetitive outputs for the same input"""
        if prompt == self.previous_input and self.previous_output:
            sentences = [s.strip() for s in self.previous_output.split('.') if s.strip()]
            if sentences:
                # Add a random sentence from previous output
                random_sentence = random.choice(sentences)
                return f"{prompt}. {random_sentence}."
        return prompt

    def _clean_output(self, text):
        """Clean the output by removing thinking tags and other unwanted content"""
        # Remove <think> tags and their content
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
       
        # Remove any other XML-like tags
        text = re.sub(r'<[^>]+>', '', text)
       
        # Remove lines that start with common prefixes
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if not any(
            line.strip().lower().startswith(prefix) for prefix in [
                'here', 'expanded', 'let me', 'i will', 'thinking', 'thought'
            ]
        )]
       
        # Join the remaining lines
        text = ' '.join(cleaned_lines)
       
        # Clean up extra spaces and punctuation
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.]+', '.', text)
        text = re.sub(r'[\s,]+,', ',', text)
       
        return text.strip()

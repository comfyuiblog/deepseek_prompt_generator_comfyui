from .exp_config import APIConfig
from .ollama_client import OllamaClient
from openai import OpenAI
import random

class PromptGeneratorNode:
    api_token: str = None
    previous_input: str = None
    previous_output: str = None
    use_char_limit: bool = True
    ollama_client = None
    cached_models = []

    @classmethod
    def INPUT_TYPES(cls):
        if cls.ollama_client is None:
            config = APIConfig()
            cls.ollama_client = OllamaClient(config.ollama_host)
            cls.cached_models = cls.ollama_client.list_models()

        return {
            "required": {
                "model_type": (["deepseek", "ollama"], {"default": "deepseek"}),
                "ollama_model": (cls.cached_models, {"default": cls.cached_models[0] if cls.cached_models else None}) if cls.cached_models else {"default": None},
                "system_prompt": (
                    "STRING", {
                        "default": "You are a creative writer tasked to turn basic stable diffusion prompts into something expressive, succinct, and beautiful.",
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
                        "default": 'Final prompt will display here. \n\nSet minimum and maximum characters to control output length. 1000 min and 3000 max is a good range to start. ',
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
        if not seed or seed == -1:
            seed = random.randint(0, 100_000_000_000)

        prompt = self.construct_prompt(prompt)
        user_content = f"Please expand this prompt: \"{prompt}\"."

        if self.use_char_limit:
            char_count = random.randint(min_char, max_char)
            user_content = f"{user_content} Please expand this prompt to {char_count} characters."

        if model_type == "deepseek":
            output = self._process_deepseek(system_prompt, user_content, seed)
        else:  # ollama
            output = self._process_ollama(ollama_model, system_prompt, user_content, seed)

        self.previous_output = output
        return {"ui": {"output_widget": output}, "result": (output,)}

    def _process_deepseek(self, system_prompt, user_content, seed):
        config = APIConfig()
        if not config.api_token:
            return "Error: DeepSeek API token not configured in config.ini"

        client = OpenAI(api_key=config.api_token, base_url="https://api.deepseek.com")
        
        try:
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
            return response.choices[0].message.content
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
            )
        except Exception as e:
            return f"Error with Ollama: {str(e)}"

    def get_random_sentences(self, paragraph):
        sentences = paragraph.split(".")
        sentences = [s for s in sentences if s.strip()]

        num = len(sentences)
        count = int(num / 4)

        count = min(count, 3)  # not more than 3 random sentences

        if count <= 1:
            count = 1

        return random.sample(sentences, count)

    def construct_prompt(self, input: str):
        p = input

        if self.previous_input == input:
            # if the same input is used, add a random sentence from previous input to ensure result is not repeated
            if self.previous_output is not None:
                rs = self.get_random_sentences(self.previous_output)
                r = '. '.join(rs)
                p = f"{p}. {r}."

        self.previous_input = input

        return p
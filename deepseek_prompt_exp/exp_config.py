import os
import configparser

class APIConfig:
    def __init__(self):
        self._api_token = None
        self._ollama_host = None
        self._load_config()

    def _load_config(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        config_path = os.path.join(parent_dir, "config.ini")

        config = configparser.ConfigParser()
        config.read(config_path)

        try:
            self._api_token = config['API']['DEEPSEEK_API_TOKEN']
        except KeyError:
            print("Warning: DEEPSEEK_API_TOKEN not found in config.ini")

        try:
            self._ollama_host = config['OLLAMA']['HOST']
        except KeyError:
            self._ollama_host = "http://localhost:11434"

    @property
    def api_token(self):
        return self._api_token

    @property
    def ollama_host(self):
        return self._ollama_host
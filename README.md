# Deepseek-comfyui-prompt-expansion
You can run this locally or using API

For Locally
Download the Ollama 

https://ollama.com/

Download Deep Seek Local Model
https://ollama.com/library/deepseek-r1

FOR API

Stable Diffusion Prompt Expansion using Deepseek API

- Create API key at https://platform.deepseek.com
- Copy `config.ini.example` to `config.ini` and put the replicate key there. 

## Installation

https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.ckpt 

download v1-5-pruned-emaonly.ckpt and Save it into the comfyui/Models/checkpoint Folder


Navigate to where you have installed ComfyUI. For example:

```shell
cd ~/dev/ComfyUI/
```

Go to the custom nodes folder:

```shell
cd custom_nodes
```

Clone this repo

```shell
git clone https://github.com/comfyuiblog/deepseek_prompt_generator_comfyui

```

Go inside the repo folder

```shell
cd sml-comfyui-prompt-expansion
```

Install the requirements

```shell
pip install -r requirements.txt
```

Copy the example config `config.ini.example` to `config.ini`, then edit the `config.ini` with the actual Repliate API token.

```shell
cp config.ini.example config.ini
```

Start ComfyUI.


## Suggestions

Please open an issue if you have any suggestions or questions. https://comfyuiblog.com/

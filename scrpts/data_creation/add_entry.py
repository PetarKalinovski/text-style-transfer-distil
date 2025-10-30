import dotenv
from litellm import completion
from pathlib import Path
import sys
import os
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.prompt_creation import create_cot_prompt_few_shot



def change_style(input:str, style:str) -> dict:
    """
    Changes the style of the input text using a few-shot reasoning prompt.
    Args:
        input (str): The input text to be transformed.
        style (str): The target style for transformation.
    Returns:
        dict: A dictionary containing the timestamp, input, style, prompt, response, and model
    information.
    """
    openrouter_key=os.getenv('OPEN_ROUTER_KEY_PAID')
    prompt=create_cot_prompt_few_shot(input, style)
    dotenv.load_dotenv()
    messages=[
        {"role": "system", "content": prompt}
    ]
    response=completion(
        model="openrouter/openai/gpt-oss-120b",
        messages=messages,
        api_key=openrouter_key
    )

    entry = {
        "timestamp": datetime.now().isoformat(),
        "input": input,
        "style": style,
        "prompt": prompt,
        "response": response.choices[0].message['content'],
        "model": "openrouter/openai/gpt-oss-120b"
    }

    return entry

def change_style_multi_step_reasoning(input:str, style:str) -> dict:
    """
    Changes the style of the input text using a multi-step reasoning prompt.
    Args:
        input (str): The input text to be transformed.
        style (str): The target style for transformation.
    Returns:
        dict: A dictionary containing the timestamp, input, style, prompt, response, and model
    information.
    """
    return ""
import dotenv
from litellm import completion, openrouter_key
from pathlib import Path
import sys
import os
import datetime


sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.prompt_creation import create_cot_prompt_few_shot



def change_style(input_file, style):
    openrouter_key=os.getenv('OPEN_ROUTER_KEY')
    prompt=create_cot_prompt_few_shot(input_file, style)
    dotenv.load_dotenv()
    messages=[
        {"role": "system", "content": prompt}
    ]
    response=completion(
        model="openrouter/deepseek/deepseek-r1:free",
        messages=messages,
        api_key=openrouter_key
    )

    entry = {
        "timestamp": datetime.now().isoformat(),
        "input_file": input_file,
        "style": style,
        "prompt": prompt,
        "response": response.choices[0].message.content,
        "model": "openrouter/deepseek/deepseek-r1:free"
    }

    return entry

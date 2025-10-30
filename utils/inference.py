from unsloth import FastLanguageModel
from peft import PeftModel
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import json
from litellm import completion
import os
from dotenv import load_dotenv
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.qwen3_class import QwenModel

load_dotenv()

def load_merged_model():
    """Load the merged model from HuggingFace"""
    print("Loading merged model...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B",  # Your saved model path
        max_seq_length=2048,
        dtype=None,
    )
    checkpoint_path = "qwen3_style_transfer"
    checkpoint_path = str(Path(__file__).resolve().parent.parent / checkpoint_path)
    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
    )

    # Enable inference mode
    FastLanguageModel.for_inference(model)
    print("Model loaded successfully!")

    return model, tokenizer



def transform_text(model, tokenizer, text, target_style="non-toxic", ):
    """
    Transform text to target style

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        text (str): Input text to transform
        target_style (str): Target style

    Returns:
        str: Transformed text
    """
    prompt = f"""You are an expert at text style transfer. Your task is to transform text from one style to another. First, explain your reasoning about what needs to be changed, then provide the transformed text.

Source Text: "{text}" 
Target Style: {target_style}
"""

    # Format as chat
    messages = [{"role": "user", "content": prompt}]

    # Tokenize
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    # Generate
    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    # Decode
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

def run_inference_on_test_set(test_set_path: str, output_path: str):
    # Load model once
    model, tokenizer = load_merged_model()

    # Load test set
    df = pd.read_csv(test_set_path, sep=',', encoding='utf-8')
    inputs = df.iloc[:, 0].tolist()  # Assuming the first column contains the input texts

    results = []
    i = 0
    for text in tqdm(inputs, desc="Running inference on test set"):
        all_text = transform_text(model, tokenizer, text, target_style="Neutral, unbiased")
        transformed_text = all_text.split("assistant")[1].strip()
        results.append({
            "input": text,
            "transformed": transformed_text
        })
        i += 1

        if i % 50 == 0 or i== 1 or i==2:
            if not Path(output_path).parent.exists():
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Intermediate results saved to {output_path} at iteration {i}")

    results_df = pd.DataFrame(results)
    # Ensure output directory exists
    if not Path(output_path).parent.exists():
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def inferene_on_base_model(test_set_path: str, output_path: str, target_style="Neutral, unbiased"):
    # Load test set
    df = pd.read_csv(test_set_path, sep=',', encoding='utf-8')
    inputs = df.iloc[:, 0].tolist()  # Assuming the first column contains the input texts

    results = []
    i = 0
    model= QwenModel()
    for text in tqdm(inputs, desc="Running inference on test set with base model"):
        prompt = f"""You are an expert at text style transfer. Your task is to transform text from one style to another. First, explain your reasoning about what needs to be changed, then provide the transformed text.
Source Text: "{text}"
Target Style: {target_style}
"""

        all_text = model.outputing(prompt)
        print(all_text)
        transformed_text = all_text.split("assistant")[1].strip()

        entry = {
            "input": text,
            "transformed": transformed_text
        }

        results.append(entry)
        i += 1

        if i % 50 == 0 or i== 1 or i==2:
            if not Path(output_path).parent.exists():
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            print(f"Intermediate base model results saved to {output_path} at iteration {i}")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
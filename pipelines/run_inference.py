from accelerate.commands.config.config_args import cache_dir
from pathlib import Path
from unsloth import FastLanguageModel
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
sys.path.append(str(Path(__file__).parent.parent))
from utils.inference import transform_text

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


if __name__ == "__main__":
    # Load model once
    model, tokenizer = load_merged_model()

    # Test examples
    test_texts = [
        "you're such an idiot for thinking that",
        "this is fucking terrible",
        "get lost, nobody wants you here"
    ]

    for text in test_texts:
        print(f"\nOriginal: {text}")
        result = transform_text(model, tokenizer, text)
        print(f"Transformed:\n{result}")
        print("-" * 80)
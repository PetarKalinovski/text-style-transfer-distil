from unsloth import FastLanguageModel
from peft import PeftModel
from huggingface_hub import login
from pathlib import Path
import sys
import os
from dotenv import load_dotenv

load_dotenv()

def load_checkpoint(checkpoint_path):
    """
    Load a model checkpoint from the specified path.

    Args:
        checkpoint_path (str): The path to the checkpoint file.

    Returns:
        model: The loaded model.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-4B",  # Your saved model path
        max_seq_length=2048,
        dtype=None,
    )

    model = PeftModel.from_pretrained(
        model,
        checkpoint_path,
    )


    model = model.merge_and_unload()


    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    hf_model_name="PetarKal/qwen3-4b-lora-style-transferSFT-Merged"
    model.push_to_hub(hf_model_name, use_auth_token=True)
    tokenizer.push_to_hub(hf_model_name, use_auth_token=True)\


if __name__ == "__main__":
    checkpoint_path = "qwen3_style_transfer"
    checkpoint_path = str(Path(__file__).resolve().parent.parent / checkpoint_path)
    load_checkpoint(checkpoint_path)
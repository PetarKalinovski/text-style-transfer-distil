from unsloth import FastLanguageModel
from peft import PeftModel
from huggingface_hub import login
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
        model_name="unsloth/Qwen2.5-Coder-3B-Instruct",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    # Load the LoRA adapter from checkpoint
    model = PeftModel.from_pretrained(model, checkpoint_path)
    # Enable inference mode
    FastLanguageModel.for_inference(model)

    login(token=os.getenv("HUGGINGFACE_TOKEN"))
    hf_model_name="PetarK/qwen3-4b-style-transferSFT"
    model.push_to_hub(hf_model_name, use_auth_token=True)
    tokenizer.push_to_hub(hf_model_name, use_auth_token=True)\


if __name__ == "__main__":
    checkpoint_path = "./lora_model"
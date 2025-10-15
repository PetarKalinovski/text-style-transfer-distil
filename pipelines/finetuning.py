from pathlib import Path
import sys

import pandas as pd
import json
from tqdm import tqdm
sys.path.append(str(Path(__file__).parent.parent))
from utils.qwen3_class import QwenModel
from utils.qwen3_class_unsloth import QwenModelUnsloth
from loguru import logger
from utils.prompt_creation import create_cot_prompt_few_shot

def main():
    """
    Fine-tune the Qwen3-4B model for text style transfer using a dataset created with few-shot reasoning prompts.
    """
    # Normal Qwen finetuning
    # model=QwenModel()
    #
    # train_location=Path(__file__).resolve().parent.parent / "data" / "train.jsonl"
    # eval_location=Path(__file__).resolve().parent.parent / "data" / "eval.jsonl"
    #
    # model.finetuning_from_jsonl(train_location, eval_location, experiment_name="cot_finetuning")

    # Unsloth Qwen finetuning

    model = QwenModelUnsloth()

    train_location=Path(__file__).resolve().parent.parent / "data" / "train.jsonl"
    eval_location=Path(__file__).resolve().parent.parent / "data" / "eval.jsonl"

    model.finetuning_from_jsonl(train_location, eval_location)


if __name__ == "__main__":
    main()
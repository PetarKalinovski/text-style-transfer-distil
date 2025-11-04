from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.inference import transform_text
from utils.inference import load_merged_model
from utils.qwen3_class import QwenModel
import json

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

if __name__ == "__main__":
    # Load model once
    base_model= QwenModel()

    with open("data/new_test.txt", "r", encoding="utf-8") as f:
        base_text = f.readlines()
    # Test examples
    test_texts = [
        "The activist decried the government's use of enhanced interrogation techniques on enemy combatants.",
        "I hate you with all my heart.",
    ]
    base_results = []
    for text in base_text[3:]:
        base_result = base_model.outputing(f"""You are an expert at text style transfer. Your task is to transform text from one style to another. First, explain your reasoning about what needs to be changed, then provide the transformed text.
        Source Text: "{text}"
        Target Style: Neutral, unbiased
        """)
        transformed_text = base_result.split("assistant")[1].strip()

        print(f"Base Model Result: {base_result}")
        base_results.append({
            "input": text,
            "transformed": transformed_text
        })


    with open("results/new_test_results_base.json", 'w', encoding='utf-8') as f:
        json.dump(base_results, f, ensure_ascii=False, indent=4)
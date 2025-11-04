from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.inference import transform_text
from utils.inference import load_merged_model
from utils.qwen3_class import QwenModel
import io
import json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

if __name__ == "__main__":
    # Load model once
    model, tokenizer = load_merged_model()
    base_model= QwenModel()

    with open("data/new_test.txt", "r", encoding="utf-8") as f:
        base_text = f.readlines()
    # Test examples
    test_texts = [
        "The activist decried the government's use of enhanced interrogation techniques on enemy combatants.",
        "I hate you with all my heart.",
    ]
    results = []
    for text in base_text:
        result = transform_text(model, tokenizer, text)
        transformed_text = result.split("assistant")[1].strip()
        print(f"Fine-tuned Model Result: {result}")
        results.append({
            "input": text,
            "transformed": transformed_text
        })

    with open("results/new_test_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

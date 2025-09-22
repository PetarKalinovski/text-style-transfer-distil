from pathlib import Path
import sys
import pandas as pd
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from scrpts.data_creation.add_entry import change_style

def main():

    wnc_data=pd.read_csv('data/training_subset.csv')

    already_processed = set()
    if Path("data/cot_dataset.jsonl").exists():
        with open("data/cot_dataset.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    already_processed.add(entry['input'])
                except json.JSONDecodeError:
                    continue
        wnc_data = wnc_data[~wnc_data["Style 1"].isin(already_processed)].reset_index(drop=True)

    results=[]
    try:
        for i, input_entry in enumerate(tqdm(wnc_data["Style 1"], desc="Processing pre-neutrality datasets")):
            result = change_style(input_entry, "Neutral, unbiased")
            results.append(result)

            if (i + 1) % 10 == 0:
                with open("data/cot_dataset.jsonl", 'a', encoding='utf-8') as f:
                    for entry in results:
                        if entry is not None:
                            f.write(json.dumps(entry) + '\n')
                results = []
                tqdm.write(f"Saved batch ending at item {i + 1}")



    finally:
        if results:
            with open("data/cot_dataset.jsonl", 'a', encoding='utf-8') as f:
                for entry in results:
                    if entry is not None:
                        f.write(json.dumps(entry) + '\n')
            print(f"Saved final batch of {len(results)} items")




if __name__ == "__main__":
    main()

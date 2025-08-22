from pathlib import Path
import sys
import pandas as pd
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))
from scrpts.data_creation.add_entry import change_style

def main():

    paradetox_data=pd.read_csv('data/paradetox.tsv', sep='\t')

    results=[]
    try:
        for i, input_entry in enumerate(tqdm(paradetox_data["toxic"], desc="Processing toxic entries")):
            result = change_style(input_entry, "Non-toxic")
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

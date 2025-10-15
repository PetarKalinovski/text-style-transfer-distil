import json
import random
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

def main():
    all_data = []
    path=Path(__file__).resolve().parent.parent / "data" / "cot_dataset.jsonl"
    with open(path, "r") as f:
        for line in f:
            all_data.append(json.loads(line.strip()))


    random.shuffle(all_data)
    split_idx = int(0.8 * len(all_data))

    train_data = all_data[:split_idx]
    eval_data = all_data[split_idx:]

    train_location=path=Path(__file__).resolve().parent.parent / "data" / "train.jsonl"
    eval_location=path=Path(__file__).resolve().parent.parent / "data" / "eval.jsonl"
    # Save splits
    with open(train_location, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(eval_location, "w") as f:
        for item in eval_data:
            f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
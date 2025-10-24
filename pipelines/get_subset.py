from pathlib import Path
import sys
import pandas as pd
import json
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))


def get_subset(n=3000):
    path= 'data/train_en.txt'
    df = pd.read_csv(path, sep='\t', encoding='utf-8')

    style1_data = df.iloc[:, 0]

    random_sample = style1_data.sample(n, random_state=42)

    return random_sample


def save_to_csv(data, filename='training_subset.csv'):
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    filepath = data_dir / filename
    data.to_csv(filepath, index=False, encoding='utf-8')

    return filepath


if __name__ == "__main__":
    data = get_subset(n=3000)
    save_to_csv(data)
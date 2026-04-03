import json
import pandas as pd
import os

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_csv(data, path):
    ensure_dir(path)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

def save_json(data, path):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
import pandas as pd

def final_polish(df):
    """
    Final data scientist's cleanup:
    1. Removes persistent subword artifacts.
    2. Resolves overlapping spans by keeping the longest word.
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    df_clean = df_clean[~df_clean['word'].astype(str).str.contains('##', na=False)]
    df_clean = df_clean[df_clean['word'].astype(str).str.len() > 2]
    df_clean = df_clean[~df_clean['entity_group'].astype(str).str.contains('LABEL', na=False)]
    df_clean['word'] = df_clean['word'].astype(str).str.lower().str.strip()
    df_clean = df_clean.sort_values(['note_id', 'sentence_index', 'word'])
    
    return df_clean


if __name__ == "__main__":
    input_path = '../data/processed/entities.csv'
    output_path = '../data/processed/entities_refined.csv'
    
    print(f"Loading raw entities from {input_path}...")
    try:
        df_entities = pd.read_csv(input_path)
        print(f"Original entities count: {len(df_entities)}")
        
        clean_entities = final_polish(df_entities)
        
        clean_entities.to_csv(output_path, index=False)
        print(f"Polished down to {len(clean_entities)} high-quality entities.")
        print(f"Saved to {output_path}")
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}. Run stage 2 first.")
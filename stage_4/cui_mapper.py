import pandas as pd
from tqdm import tqdm

class CUIMapper:
    def __init__(self, linker):
        self.linker = linker

    def map_dataframe(self, df):
        results = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Mapping to UMLS"):
            row_dict = row.to_dict()
            
            # Map the first entity
            cui_1, canonical_1 = self.linker.get_umls_concept(row['word_1'])
            row_dict['cui_1'] = cui_1
            row_dict['canonical_name_1'] = canonical_1
            
            # Map the second entity
            cui_2, canonical_2 = self.linker.get_umls_concept(row['word_2'])
            row_dict['cui_2'] = cui_2
            row_dict['canonical_name_2'] = canonical_2
            
            results.append(row_dict)
            
        return pd.DataFrame(results)
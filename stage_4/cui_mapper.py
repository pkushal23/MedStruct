import pandas as pd
from tqdm import tqdm

# Enable tqdm with pandas apply
tqdm.pandas(desc="Mapping to UMLS")

class CUIMapper:
    def __init__(self, linker):
        self.linker = linker

    def _map_row(self, row):
        # Map the first entity
        cui_1, canonical_1 = self.linker.get_umls_concept(row['word_1'])
        
        # Map the second entity
        cui_2, canonical_2 = self.linker.get_umls_concept(row['word_2'])
        
        return pd.Series([cui_1, canonical_1, cui_2, canonical_2])

    def map_dataframe(self, df):
        # Create a copy to avoid SettingWithCopyWarning
        df_mapped = df.copy()
        
        # Apply the mapping function to generate new columns
        df_mapped[['cui_1', 'canonical_name_1', 'cui_2', 'canonical_name_2']] = df_mapped.progress_apply(self._map_row, axis=1)
            
        return df_mapped
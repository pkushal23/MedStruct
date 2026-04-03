import re
import pandas as pd
import logging

# Set up logging
logger = logging.getLogger(__name__)

def clean_mimic_text(text: str) -> str:
    """
    Cleans raw MIMIC-IV note text using Regex to prepare it for NER.
    """
    if not isinstance(text, str):
        return ""
    
    
    text = re.sub(r'\[\*\*.*?\*\*\]', '[PHI]', text)
    
    # 2. Clean up formatting artifacts
    # Doctors often use long lines of underscores or dashes as visual separators
    text = re.sub(r'_{3,}', ' ', text)  # Replace 3+ underscores with a space
    text = re.sub(r'-{4,}', ' ', text)  # Replace 4+ dashes with a space
    text = re.sub(r'\*{3,}', ' ', text) # Replace 3+ asterisks with a space
    
    # 3. Normalize Whitespace
    # Reduce excessive blank lines (3 or more) down to standard double spacing
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Reduce multiple inline spaces to a single space
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()

def preprocess_notes_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the cleaning pipeline to the text column of the dataframe.
    """
    logger.info("Starting text cleaning phase...")
    
    # Work on a copy to prevent SettingWithCopy warnings
    df_clean = df.copy()
    
    # Apply the regex cleaning to create a new column
    df_clean['cleaned_text'] = df_clean['text'].apply(clean_mimic_text)
    
    logger.info("Text cleaning complete.")
    return df_clean

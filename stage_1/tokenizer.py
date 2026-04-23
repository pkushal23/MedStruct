import pandas as pd
import spacy
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class ClinicalTokenizer:
    def __init__(self, model_name: str = "en_core_sci_lg"):
        """
        Initializes the SciSpaCy model.
        Note: You must install the model via pip before running:
        pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_lg-0.5.4.tar.gz
        """
        logger.info(f"Loading SciSpaCy model: {model_name}...")
        # We only need the parser for sentence boundaries, disabling NER saves massive RAM/Time
        self.nlp = spacy.load(model_name, disable=["ner", "tagger", "lemmatizer", "textcat"])
        # Increase max length for massive notes
        self.nlp.max_length = 5000000

    def tokenize_dataframe(self, df_segmented: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Takes a segmented dataframe and tokenizes section text into sentences.
        Returns:
            df_sections: (note_id, hadm_id, section_name, section_text)
            df_sentences: (note_id, hadm_id, section_name, sentence_index, sentence_text)
        """
        logger.info("Preparing section rows for sentence tokenization...")

        # The current segmenter already emits long-format rows with section_name and section_text.
        # Keep compatibility with older wide-format inputs by melting only when needed.
        if {'section_name', 'section_text'}.issubset(df_segmented.columns):
            df_sections = df_segmented.copy()
        else:
            id_cols = [c for c in ['subject_id', 'hadm_id', 'note_id'] if c in df_segmented.columns]
            section_cols = [c for c in df_segmented.columns if c not in id_cols]

            if 'section_text' in section_cols:
                section_cols.remove('section_text')
            if 'section_name' in section_cols:
                section_cols.remove('section_name')

            df_sections = df_segmented.melt(
                id_vars=id_cols,
                value_vars=section_cols,
                var_name='section_name',
                value_name='section_text'
            )

        # 1. Drop true missing values (NaNs)
        df_sections = df_sections.dropna(subset=['section_text'])

        # 2. Force the column to be strings (prevents float errors)
        df_sections['section_text'] = df_sections['section_text'].astype(str)

        # 3. Drop empty string sections
        df_sections = df_sections[df_sections['section_text'].str.strip() != ""].copy()
       
        logger.info("Running SciSpaCy sentence tokenization (this may take a while)...")
        sentences_data = []
       
        # We use nlp.pipe for rapid batch processing
        texts = df_sections['section_text'].tolist()
        docs = self.nlp.pipe(texts, batch_size=50)
       
        for idx, doc in enumerate(docs):
            base_row = df_sections.iloc[idx]
           
            for sent_idx, sent in enumerate(doc.sents):
                clean_sent = sent.text.strip()
                if len(clean_sent) > 2: # Ignore purely punctuation/empty sentences
                    sentences_data.append({
                        'note_id': base_row['note_id'],
                        'hadm_id': base_row['hadm_id'],
                        'section_name': base_row['section_name'],
                        'sentence_index': sent_idx,
                        'sentence_text': clean_sent
                    })

        df_sentences = pd.DataFrame(sentences_data)
        logger.info(f"Generated {len(df_sentences)} discrete sentences.")
        return df_sections, df_sentences
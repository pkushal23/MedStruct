"""
Radiology Report Specific Enhancement Module
Extracts structured information from radiology reports:
- Measurements/Dimensions
- Severity/Abnormality classification
- Section-based findings
- Clinical correlations (Finding-Finding relationships)
"""

import re
import pandas as pd
from typing import List, Dict, Tuple


class RadiologyEnhancer:
    """Extract structured data from radiology reports"""
    
    def __init__(self):
        # Patterns for measurement extraction
        self.measurement_patterns = [
            r'(\d+\.?\d*)\s*x\s*(\d+\.?\d*)\s*(?:x\s*(\d+\.?\d*))?\s*(cm|mm|inches?)',
            r'(\d+\.?\d*)\s*(cm|mm|inches?)\s*(?:in\s+)?(?:diameter|size)',
        ]
        
        # Abnormality indicators
        self.abnormal_keywords = [
            'abnormal', 'abnormality', 'finding', 'lesion', 'mass', 'collection',
            'thickening', 'enhancement', 'narrowing', 'stenosis', 'occlusion',
            'fracture', 'dislocation', 'rupture', 'infarction', 'hemorrhage'
        ]
        
        # Negation patterns (normal findings)
        self.normal_patterns = [
            r'no\s+evidence\s+of',
            r'negative\s+for',
            r'rule\s+out',
            r'excluded',
            r'normal',
            r'unremarkable'
        ]
    
    def extract_measurements(self, text: str) -> List[Dict]:
        """Extract measurement dimensions from text"""
        measurements = []
        for pattern in self.measurement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                groups = match.groups()
                measurements.append({
                    'measurement': match.group(0),
                    'dimensions': groups[:-1],
                    'unit': groups[-1] if groups[-1] else 'cm',
                    'start': match.start(),
                    'end': match.end()
                })
        return measurements
    
    def classify_finding_severity(self, entity_word: str, context: str) -> str:
        """Classify if a finding is normal or abnormal"""
        text = (context + ' ' + entity_word).lower()
        
        # Check for negation patterns
        for pattern in self.normal_patterns:
            if re.search(pattern, text):
                return 'normal'
        
        # Check for abnormality keywords
        for keyword in self.abnormal_keywords:
            if keyword in text:
                return 'abnormal'
        
        return 'uncertain'
    
    def extract_negations(self, text: str) -> List[Dict]:
        """Extract negated findings (e.g., 'No evidence of DVT')"""
        negations = []
        for pattern in self.normal_patterns:
            matches = re.finditer(f'{pattern}\\s+([a-zA-Z\\s]+?)(?:[,.]|$)', text, re.IGNORECASE)
            for match in matches:
                negations.append({
                    'negation_type': 'normal_finding',
                    'finding': match.group(1).strip(),
                    'full_text': match.group(0),
                    'start': match.start(),
                    'end': match.end()
                })
        return negations
    
    def extract_section_summary(self, df_entities: pd.DataFrame) -> Dict[str, List[str]]:
        """Group entities by section (Indication, Findings, Impression)"""
        if 'section_name' not in df_entities.columns:
            return {'all_sections': df_entities['word'].unique().tolist()}
        
        summary = {}
        for section in df_entities['section_name'].unique():
            entities = df_entities[df_entities['section_name'] == section]['word'].unique().tolist()
            summary[section] = entities
        return summary
    
    def extract_finding_relationships(self, df_entities: pd.DataFrame, 
                                     df_sentences: pd.DataFrame, 
                                     window: int = 2) -> List[Dict]:
        """
        Extract finding-to-finding relationships for radiology
        (e.g., "Normal DVT screening" relates to symptom "DVT suspicion")
        """
        relationships = []
        
        if 'sentence_index' not in df_entities.columns:
            return relationships
        
        # Group entities by sentence
        for sent_idx in df_entities['sentence_index'].unique():
            sent_entities = df_entities[df_entities['sentence_index'] == sent_idx]
            
            if len(sent_entities) < 2:
                continue
            
            # Get sentence text for context
            sent_text = None
            if 'note_id' in df_entities.columns and 'note_id' in sent_entities.columns:
                note_id = sent_entities['note_id'].iloc[0]
                matching_sent = df_sentences[
                    (df_sentences['note_id'] == note_id) & 
                    (df_sentences['sentence_index'] == sent_idx)
                ]
                if not matching_sent.empty:
                    sent_text = matching_sent['sentence_text'].iloc[0]
            
            # Create pairwise relationships between entities in same sentence
            entities_list = sent_entities.to_dict('records')
            for i, ent1 in enumerate(entities_list):
                for ent2 in entities_list[i+1:]:
                    rel_type = self._infer_relation_type(
                        ent1['word'], ent2['word'], 
                        ent1.get('entity_group'), ent2.get('entity_group'),
                        sent_text or ''
                    )
                    
                    if rel_type:
                        relationships.append({
                            'entity_1': ent1['word'],
                            'entity_type_1': ent1.get('entity_group', 'unknown'),
                            'entity_2': ent2['word'],
                            'entity_type_2': ent2.get('entity_group', 'unknown'),
                            'relation_type': rel_type,
                            'sentence_index': sent_idx,
                            'confidence': 0.75
                        })
        
        return relationships
    
    def _infer_relation_type(self, word1: str, word2: str, 
                            type1: str, type2: str, context: str) -> str:
        """Infer relationship between two findings using conservative lexical cues."""
        context_lower = context.lower()

        # Check for causality
        if any(pattern in context_lower for pattern in 
               ['caused by', 'due to', 'secondary to', 'resulting in', 'leading to']):
            return 'CAUSED_BY'

        # Check for explicit clinical correlation phrases only (avoid broad tokens like 'with'/'and').
        if any(pattern in context_lower for pattern in 
               ['associated with', 'compatible with', 'in keeping with', 'consistent with']):
            return 'CORRELATED_WITH'

        # Check for contrast (normal vs abnormal)
        if any(pattern in context_lower for pattern in 
               ['no evidence of', 'negative for', 'without evidence of']):
            return 'CONTRASTED_WITH'

        return None
    
    def enhance_entities_dataframe(self, df_entities: pd.DataFrame, 
                                   df_sentences: pd.DataFrame,
                                   df_raw_text: pd.DataFrame = None) -> Dict:
        """
        Create comprehensive radiology analysis output
        
        Returns:
            Dictionary with enhanced radiology data:
            - measurements: extracted dimensions
            - section_summary: entities grouped by section
            - finding_relationships: finding-to-finding relations
            - severity_classification: normal/abnormal classification
        """
        
        output = {
            'measurements': [],
            'section_summary': {},
            'finding_relationships': [],
            'severity_classification': {},
            'negations': []
        }
        
        # Extract measurements from raw text if available
        if df_raw_text is not None and not df_raw_text.empty:
            text = df_raw_text['text'].iloc[0] if isinstance(df_raw_text, pd.DataFrame) else str(df_raw_text)
            output['measurements'] = self.extract_measurements(text)
            output['negations'] = self.extract_negations(text)
        
        # Section summary
        output['section_summary'] = self.extract_section_summary(df_entities)
        
        # Finding relationships
        output['finding_relationships'] = self.extract_finding_relationships(df_entities, df_sentences)
        
        # Severity classification per entity
        if df_sentences is not None and not df_sentences.empty:
            for _, entity in df_entities.iterrows():
                sent_idx = entity.get('sentence_index')
                matching_sent = df_sentences[
                    (df_sentences['sentence_index'] == sent_idx) if sent_idx is not None else [False] * len(df_sentences)
                ]
                context = matching_sent['sentence_text'].iloc[0] if not matching_sent.empty else ''
                severity = self.classify_finding_severity(entity['word'], context)
                output['severity_classification'][entity['word']] = severity
        
        return output

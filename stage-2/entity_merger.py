import pandas as pd

def merge_and_deduplicate(df_list):
    if not df_list:
        return pd.DataFrame()
    
    combined_df = pd.concat(df_list, ignore_index=True)
    
    if combined_df.empty:
        return combined_df

    combined_df = combined_df.sort_values(
        by=['note_id', 'sentence_index', 'start', 'score'],
        ascending=[True, True, True, False]
    )

    deduplicated_df = combined_df.drop_duplicates(
        subset=['note_id', 'sentence_index', 'start', 'end'],
        keep='first'
    )

    return deduplicated_df.reset_index(drop=True)
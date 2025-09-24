# src/ingestion/excel_parser.py

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def parse_excel_qa(file_path: str) -> list[dict] or None:
    """
    Parses a two-column Excel file (user_desc, user_reply_desc) into a list of dictionaries.
    
    Args:
        file_path: The path to the .xlsx file.

    Returns:
        A list of dictionaries, where each dictionary is a Q&A pair, or None if an error occurs.
    """
    try:
        df = pd.read_excel(file_path)
        # Ensure the required columns exist
        if 'user_desc' not in df.columns or 'user_reply_desc' not in df.columns:
            log.error(f"Excel file at {file_path} must contain 'user_desc' and 'user_reply_desc' columns.")
            return None
        
        # Convert dataframe to a list of records (dictionaries)
        qa_list = df.to_dict(orient='records')
        log.info(f"Successfully parsed {len(qa_list)} Q&A pairs from {file_path}")
        return qa_list

    except FileNotFoundError:
        log.error(f"Excel file not found at path: {file_path}")
        return None
    except Exception as e:
        log.error(f"An error occurred while parsing the Excel file: {e}")
        return None
"""
Data Loading Module for Fashion Trend Analysis
Handles loading and preprocessing of extracted text content from PDFs
"""

import os
import glob
import pandas as pd
import re
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FashionDataLoader:
    def __init__(self, extracted_content_dir: str = "extracted_content"):
        self.extracted_content_dir = extracted_content_dir
        self.texts = []
        self.filenames = []
        self.metadata = {}
        
    def load_all_texts(self) -> Tuple[List[str], List[str]]:
        """
        Load all text files from extracted content directories
        
        Returns:
            Tuple of (texts, filenames)
        """
        texts = []
        filenames = []
        
        # Get all subdirectories in extracted_content
        subdirs = [d for d in os.listdir(self.extracted_content_dir) 
                  if os.path.isdir(os.path.join(self.extracted_content_dir, d))]
        
        logger.info(f"Found {len(subdirs)} subdirectories to process")
        
        for subdir in subdirs:
            subdir_path = os.path.join(self.extracted_content_dir, subdir)
            
            # Look for text files in each subdirectory
            text_files = glob.glob(os.path.join(subdir_path, "*.txt"))
            
            for text_file in text_files:
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if content.strip():  # Only add non-empty files
                            texts.append(content)
                            filenames.append(os.path.basename(text_file))
                            logger.info(f"Loaded: {text_file}")
                except Exception as e:
                    logger.error(f"Error loading {text_file}: {e}")
                    continue
        
        self.texts = texts
        self.filenames = filenames
        logger.info(f"Successfully loaded {len(texts)} text files")
        return texts, filenames
    
    def preprocess_text(self, text: str) -> str:
        """
        Basic text preprocessing
        
        Args:
            text: Raw text content
            
        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page markers
        text = re.sub(r'PAGE \d+', '', text)
        text = re.sub(r'={5,}', '', text)
        text = re.sub(r'-{5,}', '', text)
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Clean up extra spaces
        text = text.strip()
        
        return text
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get processed data as DataFrame
        
        Returns:
            DataFrame with processed text data
        """
        if not self.texts:
            self.load_all_texts()
        
        processed_texts = [self.preprocess_text(text) for text in self.texts]
        
        df = pd.DataFrame({
            'filename': self.filenames,
            'raw_text': self.texts,
            'processed_text': processed_texts,
            'text_length': [len(text) for text in processed_texts],
            'word_count': [len(text.split()) for text in processed_texts]
        })
        
        return df
    
    def categorize_documents(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize documents based on filename patterns
        
        Args:
            df: DataFrame with document data
            
        Returns:
            DataFrame with category information
        """
        def get_category(filename):
            filename_lower = filename.lower()
            
            if 'men' in filename_lower:
                return 'Mens'
            elif 'women' in filename_lower:
                return 'Womens'
            elif 'catwalk' in filename_lower:
                return 'Catwalk'
            elif 'collection' in filename_lower:
                return 'Collection'
            elif 'denim' in filename_lower:
                return 'Denim'
            elif 'accessories' in filename_lower:
                return 'Accessories'
            elif 'footwear' in filename_lower:
                return 'Footwear'
            else:
                return 'General'
        
        def get_season(filename):
            if 'a_w_24_25' in filename.lower():
                return 'AW_24_25'
            elif 'a_w_25_26' in filename.lower():
                return 'AW_25_26'
            else:
                return 'Unknown'
        
        def get_location(filename):
            filename_lower = filename.lower()
            cities = ['london', 'paris', 'milan', 'new_york', 'copenhagen']
            for city in cities:
                if city in filename_lower:
                    return city.replace('_', ' ').title()
            return 'General'
        
        df['category'] = df['filename'].apply(get_category)
        df['season'] = df['filename'].apply(get_season)
        df['location'] = df['filename'].apply(get_location)
        
        return df
    
    def get_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of the loaded data
        
        Args:
            df: DataFrame with document data
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {
            'total_documents': len(df),
            'total_words': df['word_count'].sum(),
            'average_document_length': df['text_length'].mean(),
            'category_distribution': df['category'].value_counts().to_dict(),
            'season_distribution': df['season'].value_counts().to_dict(),
            'location_distribution': df['location'].value_counts().to_dict()
        }
        
        return stats

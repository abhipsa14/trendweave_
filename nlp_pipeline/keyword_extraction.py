"""
Keyword Extraction Module for Fashion Trend Analysis
Provides keyword extraction using RAKE and KeyBERT
"""

import pandas as pd
import numpy as np
from rake_nltk import Rake
from keybert import KeyBERT
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from typing import List, Dict, Tuple
import logging
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

logger = logging.getLogger(__name__)

class FashionKeywordExtractor:
    def __init__(self):
        self.rake = Rake()
        self.keybert_model = KeyBERT()
        self.lemmatizer = WordNetLemmatizer()
        
        # Fashion-specific stop words
        self.fashion_stop_words = {
            'fashion', 'style', 'trend', 'collection', 'season', 'wear', 'clothing',
            'garment', 'item', 'piece', 'look', 'show', 'brand', 'designer',
            'page', 'image', 'figure', 'table', 'source', 'report', 'analysis',
            'catwalk', 'runway', 'model', 'brand', 'line', 'product'
        }
        
        # Configure RAKE with custom stop words
        stop_words = set(stopwords.words('english'))
        stop_words.update(self.fashion_stop_words)
        self.rake = Rake(stopwords=stop_words)
        
    def extract_rake_keywords(self, text: str, num_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using RAKE algorithm
        
        Args:
            text: Input text
            num_keywords: Number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            self.rake.extract_keywords_from_text(text)
            keywords_with_scores = self.rake.get_ranked_phrases_with_scores()
            
            # Filter out single character keywords and sort by score
            filtered_keywords = [(phrase, score) for score, phrase in keywords_with_scores 
                               if len(phrase) > 2 and not phrase.isdigit()]
            
            return filtered_keywords[:num_keywords]
            
        except Exception as e:
            logger.error(f"Error in RAKE keyword extraction: {e}")
            return []
    
    def extract_keybert_keywords(self, text: str, num_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using KeyBERT
        
        Args:
            text: Input text
            num_keywords: Number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples
        """
        try:
            keywords = self.keybert_model.extract_keywords(
                text, 
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=num_keywords,
                use_maxsum=True,
                nr_candidates=20,
                diversity=0.5
            )
            
            # Filter out fashion stop words
            filtered_keywords = [(phrase, score) for phrase, score in keywords 
                               if phrase.lower() not in self.fashion_stop_words]
            
            return filtered_keywords
            
        except Exception as e:
            logger.error(f"Error in KeyBERT keyword extraction: {e}")
            return []
    
    def extract_tfidf_keywords(self, texts: List[str], num_keywords: int = 10) -> Dict[int, List[Tuple[str, float]]]:
        """
        Extract keywords using TF-IDF
        
        Args:
            texts: List of text documents
            num_keywords: Number of keywords to extract per document
            
        Returns:
            Dictionary with document_id as key and list of (keyword, score) tuples
        """
        try:
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{3,}\b',
                ngram_range=(1, 3)
            )
            
            # Fit and transform texts
            tfidf_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Extract keywords for each document
            document_keywords = {}
            
            for doc_idx in range(len(texts)):
                doc_scores = tfidf_matrix[doc_idx].toarray().flatten()
                
                # Get top keywords for this document
                top_indices = doc_scores.argsort()[-num_keywords:][::-1]
                top_keywords = [(feature_names[i], doc_scores[i]) for i in top_indices 
                              if doc_scores[i] > 0]
                
                # Filter out fashion stop words
                filtered_keywords = [(phrase, score) for phrase, score in top_keywords 
                                   if phrase.lower() not in self.fashion_stop_words]
                
                document_keywords[doc_idx] = filtered_keywords
            
            return document_keywords
            
        except Exception as e:
            logger.error(f"Error in TF-IDF keyword extraction: {e}")
            return {}
    
    def extract_all_keywords(self, texts: List[str], filenames: List[str] = None) -> pd.DataFrame:
        """
        Extract keywords using all methods
        
        Args:
            texts: List of text documents
            filenames: List of filenames (optional)
            
        Returns:
            DataFrame with keyword extraction results
        """
        results = []
        
        # Extract TF-IDF keywords for all documents
        tfidf_keywords = self.extract_tfidf_keywords(texts)
        
        for i, text in enumerate(texts):
            logger.info(f"Extracting keywords for document {i+1}/{len(texts)}")
            
            # RAKE keywords
            rake_keywords = self.extract_rake_keywords(text)
            
            # KeyBERT keywords
            keybert_keywords = self.extract_keybert_keywords(text)
            
            # TF-IDF keywords
            tfidf_doc_keywords = tfidf_keywords.get(i, [])
            
            result = {
                'document_id': i,
                'filename': filenames[i] if filenames else f"doc_{i}",
                'rake_keywords': [kw[0] for kw in rake_keywords],
                'rake_scores': [kw[1] for kw in rake_keywords],
                'keybert_keywords': [kw[0] for kw in keybert_keywords],
                'keybert_scores': [kw[1] for kw in keybert_keywords],
                'tfidf_keywords': [kw[0] for kw in tfidf_doc_keywords],
                'tfidf_scores': [kw[1] for kw in tfidf_doc_keywords]
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_keyword_trends(self, keyword_df: pd.DataFrame, text_df: pd.DataFrame) -> Dict:
        """
        Analyze keyword trends across categories and seasons
        
        Args:
            keyword_df: DataFrame with keyword extraction results
            text_df: DataFrame with document metadata
            
        Returns:
            Dictionary with keyword trend analysis
        """
        # Combine keyword and text data
        combined_df = pd.concat([keyword_df, text_df], axis=1)
        
        # Analyze trends by category
        category_trends = {}
        for category in combined_df['category'].unique():
            category_data = combined_df[combined_df['category'] == category]
            
            # Collect all keywords for this category
            all_rake_keywords = []
            all_keybert_keywords = []
            
            for _, row in category_data.iterrows():
                all_rake_keywords.extend(row['rake_keywords'])
                all_keybert_keywords.extend(row['keybert_keywords'])
            
            # Count keyword frequencies
            rake_counter = Counter(all_rake_keywords)
            keybert_counter = Counter(all_keybert_keywords)
            
            category_trends[category] = {
                'document_count': len(category_data),
                'top_rake_keywords': rake_counter.most_common(10),
                'top_keybert_keywords': keybert_counter.most_common(10),
                'total_unique_rake_keywords': len(rake_counter),
                'total_unique_keybert_keywords': len(keybert_counter)
            }
        
        # Analyze trends by season
        season_trends = {}
        for season in combined_df['season'].unique():
            season_data = combined_df[combined_df['season'] == season]
            
            # Collect all keywords for this season
            all_rake_keywords = []
            all_keybert_keywords = []
            
            for _, row in season_data.iterrows():
                all_rake_keywords.extend(row['rake_keywords'])
                all_keybert_keywords.extend(row['keybert_keywords'])
            
            # Count keyword frequencies
            rake_counter = Counter(all_rake_keywords)
            keybert_counter = Counter(all_keybert_keywords)
            
            season_trends[season] = {
                'document_count': len(season_data),
                'top_rake_keywords': rake_counter.most_common(10),
                'top_keybert_keywords': keybert_counter.most_common(10),
                'total_unique_rake_keywords': len(rake_counter),
                'total_unique_keybert_keywords': len(keybert_counter)
            }
        
        return {
            'category_trends': category_trends,
            'season_trends': season_trends
        }
    
    def get_fashion_specific_keywords(self, keyword_df: pd.DataFrame, text_df: pd.DataFrame) -> Dict:
        """
        Analyze fashion-specific keyword patterns
        
        Args:
            keyword_df: DataFrame with keyword extraction results
            text_df: DataFrame with document metadata
            
        Returns:
            Dictionary with fashion-specific keyword insights
        """
        # Combine keyword and text data
        combined_df = pd.concat([keyword_df, text_df], axis=1)
        
        # Fashion categories for analysis
        fashion_categories = {
            'colors': ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple', 'brown', 'grey', 'beige', 'navy'],
            'materials': ['cotton', 'silk', 'wool', 'denim', 'leather', 'synthetic', 'linen', 'cashmere', 'polyester'],
            'styles': ['casual', 'formal', 'sporty', 'elegant', 'vintage', 'modern', 'minimalist', 'bohemian'],
            'garments': ['dress', 'shirt', 'pants', 'jacket', 'coat', 'skirt', 'sweater', 'blouse', 'suit'],
            'accessories': ['bag', 'shoes', 'jewelry', 'watch', 'belt', 'hat', 'scarf', 'sunglasses'],
            'trends': ['sustainable', 'eco-friendly', 'vintage', 'minimalist', 'oversized', 'cropped', 'high-waisted']
        }
        
        fashion_insights = {}
        
        for category, keywords in fashion_categories.items():
            category_results = []
            
            for keyword in keywords:
                # Count occurrences across all documents
                rake_count = 0
                keybert_count = 0
                
                for _, row in combined_df.iterrows():
                    # Check in RAKE keywords
                    if any(keyword.lower() in kw.lower() for kw in row['rake_keywords']):
                        rake_count += 1
                    
                    # Check in KeyBERT keywords
                    if any(keyword.lower() in kw.lower() for kw in row['keybert_keywords']):
                        keybert_count += 1
                
                if rake_count > 0 or keybert_count > 0:
                    category_results.append({
                        'keyword': keyword,
                        'rake_occurrences': rake_count,
                        'keybert_occurrences': keybert_count,
                        'total_occurrences': rake_count + keybert_count
                    })
            
            # Sort by total occurrences
            category_results.sort(key=lambda x: x['total_occurrences'], reverse=True)
            fashion_insights[category] = category_results
        
        return fashion_insights
    
    def get_keyword_evolution(self, keyword_df: pd.DataFrame, text_df: pd.DataFrame) -> Dict:
        """
        Analyze keyword evolution between seasons
        
        Args:
            keyword_df: DataFrame with keyword extraction results
            text_df: DataFrame with document metadata
            
        Returns:
            Dictionary with keyword evolution analysis
        """
        # Combine keyword and text data
        combined_df = pd.concat([keyword_df, text_df], axis=1)
        
        # Compare keywords between seasons
        seasons = combined_df['season'].unique()
        keyword_evolution = {}
        
        for season in seasons:
            season_data = combined_df[combined_df['season'] == season]
            
            # Collect all keywords for this season
            all_keywords = []
            for _, row in season_data.iterrows():
                all_keywords.extend(row['rake_keywords'])
                all_keywords.extend(row['keybert_keywords'])
            
            # Count keyword frequencies
            keyword_counter = Counter(all_keywords)
            
            keyword_evolution[season] = {
                'total_documents': len(season_data),
                'unique_keywords': len(keyword_counter),
                'top_keywords': keyword_counter.most_common(20),
                'keyword_distribution': dict(keyword_counter)
            }
        
        return keyword_evolution

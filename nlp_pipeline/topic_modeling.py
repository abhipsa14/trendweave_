"""
Topic Modeling Module for Fashion Trend Analysis
Provides topic modeling using BERTopic and LDA
"""

import pandas as pd
import numpy as np
from bertopic import BERTopic
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from typing import List, Dict, Tuple
import logging

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

class FashionTopicModeler:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add fashion-specific stop words
        fashion_stop_words = {
            'fashion', 'style', 'trend', 'collection', 'season', 'wear', 'clothing',
            'garment', 'item', 'piece', 'look', 'show', 'brand', 'designer',
            'page', 'image', 'figure', 'table', 'source', 'report', 'analysis'
        }
        self.stop_words.update(fashion_stop_words)
        
        self.bertopic_model = None
        self.lda_model = None
        self.vectorizer = None
        
    def preprocess_for_topics(self, text: str) -> str:
        """
        Preprocess text for topic modeling
        
        Args:
            text: Input text
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def train_bertopic_model(self, texts: List[str], num_topics: int = 10) -> Tuple[BERTopic, List[int]]:
        """
        Train BERTopic model
        
        Args:
            texts: List of text documents
            num_topics: Number of topics to extract
            
        Returns:
            Tuple of (trained_model, topic_assignments)
        """
        logger.info("Training BERTopic model...")
        
        # Preprocess texts
        processed_texts = [self.preprocess_for_topics(text) for text in texts]
        
        # Initialize BERTopic model
        self.bertopic_model = BERTopic(
            language="english",
            nr_topics=num_topics,
            verbose=True,
            calculate_probabilities=True
        )
        
        # Fit the model
        topics, probabilities = self.bertopic_model.fit_transform(processed_texts)
        
        logger.info(f"BERTopic model trained with {len(self.bertopic_model.get_topic_info())} topics")
        
        return self.bertopic_model, topics
    
    def train_lda_model(self, texts: List[str], num_topics: int = 10) -> Tuple[LatentDirichletAllocation, np.ndarray]:
        """
        Train LDA model
        
        Args:
            texts: List of text documents
            num_topics: Number of topics to extract
            
        Returns:
            Tuple of (trained_model, topic_distributions)
        """
        logger.info("Training LDA model...")
        
        # Preprocess texts
        processed_texts = [self.preprocess_for_topics(text) for text in texts]
        
        # Create vectorizer
        self.vectorizer = CountVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        
        # Fit vectorizer and transform texts
        doc_term_matrix = self.vectorizer.fit_transform(processed_texts)
        
        # Initialize and train LDA model
        self.lda_model = LatentDirichletAllocation(
            n_components=num_topics,
            random_state=42,
            max_iter=100,
            learning_method='online'
        )
        
        # Fit the model
        doc_topic_probs = self.lda_model.fit_transform(doc_term_matrix)
        
        logger.info(f"LDA model trained with {num_topics} topics")
        
        return self.lda_model, doc_topic_probs
    
    def get_bertopic_results(self, texts: List[str], filenames: List[str] = None) -> Dict:
        """
        Get BERTopic analysis results
        
        Args:
            texts: List of text documents
            filenames: List of filenames (optional)
            
        Returns:
            Dictionary with BERTopic results
        """
        if self.bertopic_model is None:
            model, topics = self.train_bertopic_model(texts)
        else:
            processed_texts = [self.preprocess_for_topics(text) for text in texts]
            topics, _ = self.bertopic_model.transform(processed_texts)
        
        # Get topic information
        topic_info = self.bertopic_model.get_topic_info()
        
        # Get topic keywords
        topic_keywords = {}
        for topic_id in topic_info['Topic'].values:
            if topic_id != -1:  # Skip outlier topic
                keywords = self.bertopic_model.get_topic(topic_id)
                topic_keywords[topic_id] = [word for word, _ in keywords[:10]]
        
        # Document-topic assignments
        document_topics = []
        for i, topic in enumerate(topics):
            doc_info = {
                'document_id': i,
                'topic_id': topic,
                'topic_keywords': topic_keywords.get(topic, []),
                'filename': filenames[i] if filenames else f"doc_{i}"
            }
            document_topics.append(doc_info)
        
        return {
            'model': self.bertopic_model,
            'topic_info': topic_info,
            'topic_keywords': topic_keywords,
            'document_topics': document_topics,
            'topics': topics
        }
    
    def get_lda_results(self, texts: List[str], filenames: List[str] = None) -> Dict:
        """
        Get LDA analysis results
        
        Args:
            texts: List of text documents
            filenames: List of filenames (optional)
            
        Returns:
            Dictionary with LDA results
        """
        if self.lda_model is None:
            model, doc_topic_probs = self.train_lda_model(texts)
        else:
            processed_texts = [self.preprocess_for_topics(text) for text in texts]
            doc_term_matrix = self.vectorizer.transform(processed_texts)
            doc_topic_probs = self.lda_model.transform(doc_term_matrix)
        
        # Get topic keywords
        feature_names = self.vectorizer.get_feature_names_out()
        topic_keywords = {}
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_keywords[topic_idx] = top_words
        
        # Document-topic assignments
        document_topics = []
        for i, doc_probs in enumerate(doc_topic_probs):
            dominant_topic = np.argmax(doc_probs)
            doc_info = {
                'document_id': i,
                'dominant_topic': dominant_topic,
                'topic_probabilities': doc_probs.tolist(),
                'topic_keywords': topic_keywords[dominant_topic],
                'filename': filenames[i] if filenames else f"doc_{i}"
            }
            document_topics.append(doc_info)
        
        return {
            'model': self.lda_model,
            'topic_keywords': topic_keywords,
            'document_topics': document_topics,
            'doc_topic_probabilities': doc_topic_probs
        }
    
    def analyze_topic_trends(self, bertopic_results: Dict, text_df: pd.DataFrame) -> Dict:
        """
        Analyze topic trends across categories and seasons
        
        Args:
            bertopic_results: Results from BERTopic analysis
            text_df: DataFrame with document metadata
            
        Returns:
            Dictionary with trend analysis
        """
        # Create DataFrame with topic assignments
        topic_df = pd.DataFrame(bertopic_results['document_topics'])
        
        # Merge with text metadata
        combined_df = pd.concat([topic_df, text_df], axis=1)
        
        # Analyze trends by category
        category_trends = {}
        for category in combined_df['category'].unique():
            category_data = combined_df[combined_df['category'] == category]
            topic_dist = category_data['topic_id'].value_counts()
            
            category_trends[category] = {
                'document_count': len(category_data),
                'topic_distribution': topic_dist.to_dict(),
                'dominant_topics': topic_dist.head(3).index.tolist()
            }
        
        # Analyze trends by season
        season_trends = {}
        for season in combined_df['season'].unique():
            season_data = combined_df[combined_df['season'] == season]
            topic_dist = season_data['topic_id'].value_counts()
            
            season_trends[season] = {
                'document_count': len(season_data),
                'topic_distribution': topic_dist.to_dict(),
                'dominant_topics': topic_dist.head(3).index.tolist()
            }
        
        return {
            'category_trends': category_trends,
            'season_trends': season_trends
        }
    
    def get_topic_evolution(self, bertopic_results: Dict, text_df: pd.DataFrame) -> Dict:
        """
        Analyze topic evolution between seasons
        
        Args:
            bertopic_results: Results from BERTopic analysis
            text_df: DataFrame with document metadata
            
        Returns:
            Dictionary with topic evolution analysis
        """
        # Create DataFrame with topic assignments
        topic_df = pd.DataFrame(bertopic_results['document_topics'])
        combined_df = pd.concat([topic_df, text_df], axis=1)
        
        # Compare topics between seasons
        seasons = combined_df['season'].unique()
        topic_evolution = {}
        
        for season in seasons:
            season_data = combined_df[combined_df['season'] == season]
            topic_counts = season_data['topic_id'].value_counts()
            
            topic_evolution[season] = {
                'total_documents': len(season_data),
                'topic_distribution': topic_counts.to_dict(),
                'top_topics': topic_counts.head(5).index.tolist()
            }
        
        return topic_evolution

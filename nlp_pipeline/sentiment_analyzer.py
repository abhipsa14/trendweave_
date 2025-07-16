"""
Sentiment Analysis Module for Fashion Trend Analysis
Provides sentiment analysis using TextBlob, VADER, and DistilBERT
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
from typing import List, Dict, Tuple
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# Download required NLTK data
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

logger = logging.getLogger(__name__)

class FashionSentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
        # Initialize BERT model for sentiment analysis
        try:
            self.bert_sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info("DistilBERT sentiment model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load DistilBERT model: {e}. Using fallback model.")
            try:
                self.bert_sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Fallback RoBERTa model loaded successfully")
            except Exception as e2:
                logger.error(f"Failed to load fallback model: {e2}")
                self.bert_sentiment_pipeline = None
        
    def analyze_textblob_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using TextBlob
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment scores
        """
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment
            if polarity > 0.1:
                sentiment_label = 'Positive'
            elif polarity < -0.1:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
                
            return {
                'textblob_polarity': polarity,
                'textblob_subjectivity': subjectivity,
                'textblob_sentiment': sentiment_label
            }
        except Exception as e:
            logger.error(f"Error in TextBlob sentiment analysis: {e}")
            return {
                'textblob_polarity': 0.0,
                'textblob_subjectivity': 0.0,
                'textblob_sentiment': 'Neutral'
            }
    
    def analyze_vader_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using VADER
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with VADER sentiment scores
        """
        try:
            scores = self.sia.polarity_scores(text)
            
            # Determine overall sentiment
            if scores['compound'] >= 0.05:
                sentiment_label = 'Positive'
            elif scores['compound'] <= -0.05:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
                
            return {
                'vader_positive': scores['pos'],
                'vader_negative': scores['neg'],
                'vader_neutral': scores['neu'],
                'vader_compound': scores['compound'],
                'vader_sentiment': sentiment_label
            }
        except Exception as e:
            logger.error(f"Error in VADER sentiment analysis: {e}")
            return {
                'vader_positive': 0.0,
                'vader_negative': 0.0,
                'vader_neutral': 1.0,
                'vader_compound': 0.0,
                'vader_sentiment': 'Neutral'
            }
    
    def analyze_bert_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using BERT model
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with BERT sentiment scores
        """
        try:
            if self.bert_sentiment_pipeline is None:
                logger.warning("BERT model not available, returning neutral sentiment")
                return {
                    'bert_label': 'NEUTRAL',
                    'bert_score': 0.0,
                    'bert_sentiment': 'Neutral'
                }
            
            # Truncate text to avoid token limit issues
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Get prediction
            result = self.bert_sentiment_pipeline(text)[0]
            
            # Standardize label format
            label = result['label'].upper()
            score = result['score']
            
            # Map labels to standard format (DistilBERT uses POSITIVE/NEGATIVE)
            if label in ['POSITIVE', 'POS']:
                sentiment_label = 'Positive'
            elif label in ['NEGATIVE', 'NEG']:
                sentiment_label = 'Negative'
            else:
                sentiment_label = 'Neutral'
            
            return {
                'bert_label': label,
                'bert_score': score,
                'bert_sentiment': sentiment_label
            }
        except Exception as e:
            logger.error(f"Error in BERT sentiment analysis: {e}")
            return {
                'bert_label': 'NEUTRAL',
                'bert_score': 0.0,
                'bert_sentiment': 'Neutral'
            }
    
    def analyze_ensemble_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment using ensemble of TextBlob, VADER, and BERT
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with ensemble sentiment scores
        """
        try:
            # Get results from all three methods
            textblob_results = self.analyze_textblob_sentiment(text)
            vader_results = self.analyze_vader_sentiment(text)
            bert_results = self.analyze_bert_sentiment(text)
            
            # Normalize scores to -1 to 1 range
            textblob_norm = textblob_results['textblob_polarity']
            vader_norm = vader_results['vader_compound']
            
            # Convert BERT score to -1 to 1 range based on label
            bert_norm = bert_results['bert_score']
            if bert_results['bert_label'] == 'NEGATIVE':
                bert_norm = -bert_norm
            elif bert_results['bert_label'] == 'NEUTRAL':
                bert_norm = 0.0
            # For POSITIVE, keep the score as is
            
            # Calculate ensemble score (weighted average)
            ensemble_score = (textblob_norm * 0.3 + vader_norm * 0.3 + bert_norm * 0.4)
            
            # Determine ensemble sentiment
            if ensemble_score > 0.1:
                ensemble_sentiment = 'Positive'
            elif ensemble_score < -0.1:
                ensemble_sentiment = 'Negative'
            else:
                ensemble_sentiment = 'Neutral'
            
            # Calculate confidence (agreement between methods)
            sentiments = [
                textblob_results['textblob_sentiment'],
                vader_results['vader_sentiment'],
                bert_results['bert_sentiment']
            ]
            
            # Count agreements
            sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
            max_count = max(sentiment_counts.values())
            confidence = max_count / len(sentiments)
            
            return {
                'ensemble_score': ensemble_score,
                'ensemble_sentiment': ensemble_sentiment,
                'ensemble_confidence': confidence,
                'method_agreement': sentiment_counts,
                **textblob_results,
                **vader_results,
                **bert_results
            }
        except Exception as e:
            logger.error(f"Error in ensemble sentiment analysis: {e}")
            return {
                'ensemble_score': 0.0,
                'ensemble_sentiment': 'Neutral',
                'ensemble_confidence': 0.0,
                'method_agreement': {'Neutral': 3}
            }
    
    def analyze_batch_sentiment(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Processing sentiment for document {i+1}/{len(texts)}")
            
            # TextBlob analysis
            textblob_results = self.analyze_textblob_sentiment(text)
            
            # VADER analysis
            vader_results = self.analyze_vader_sentiment(text)
            
            # BERT analysis
            bert_results = self.analyze_bert_sentiment(text)
            
            # Ensemble analysis
            ensemble_results = self.analyze_ensemble_sentiment(text)
            
            # Combine results
            result = {
                'document_id': i,
                **textblob_results,
                **vader_results,
                **bert_results,
                **ensemble_results
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for sentiment analysis
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with sentiment summary
        """
        summary = {
            'textblob_sentiment_distribution': df['textblob_sentiment'].value_counts().to_dict(),
            'vader_sentiment_distribution': df['vader_sentiment'].value_counts().to_dict(),
            'bert_sentiment_distribution': df['bert_sentiment'].value_counts().to_dict(),
            'ensemble_sentiment_distribution': df['ensemble_sentiment'].value_counts().to_dict(),
            'average_textblob_polarity': df['textblob_polarity'].mean(),
            'average_vader_compound': df['vader_compound'].mean(),
            'average_bert_score': df['bert_score'].mean(),
            'average_ensemble_score': df['ensemble_score'].mean(),
            'most_positive_textblob': {
                'document_id': df.loc[df['textblob_polarity'].idxmax()]['document_id'],
                'score': df['textblob_polarity'].max()
            },
            'most_negative_textblob': {
                'document_id': df.loc[df['textblob_polarity'].idxmin()]['document_id'],
                'score': df['textblob_polarity'].min()
            },
            'most_positive_vader': {
                'document_id': df.loc[df['vader_compound'].idxmax()]['document_id'],
                'score': df['vader_compound'].max()
            },
            'most_negative_vader': {
                'document_id': df.loc[df['vader_compound'].idxmin()]['document_id'],
                'score': df['vader_compound'].min()
            },
            'most_positive_bert': {
                'document_id': df.loc[df['bert_score'].idxmax()]['document_id'],
                'score': df['bert_score'].max()
            },
            'most_negative_bert': {
                'document_id': df.loc[df['bert_score'].idxmin()]['document_id'],
                'score': df['bert_score'].min()
            },
            'most_positive_ensemble': {
                'document_id': df.loc[df['ensemble_score'].idxmax()]['document_id'],
                'score': df['ensemble_score'].max()
            },
            'most_negative_ensemble': {
                'document_id': df.loc[df['ensemble_score'].idxmin()]['document_id'],
                'score': df['ensemble_score'].min()
            }
        }
        
        return summary
    
    def analyze_sentiment_by_category(self, df: pd.DataFrame, category_col: str) -> Dict:
        """
        Analyze sentiment by category
        
        Args:
            df: DataFrame with sentiment results and categories
            category_col: Column name for categories
            
        Returns:
            Dictionary with category-wise sentiment analysis
        """
        category_sentiment = {}
        
        for category in df[category_col].unique():
            category_data = df[df[category_col] == category]
            
            category_sentiment[category] = {
                'count': len(category_data),
                'avg_textblob_polarity': category_data['textblob_polarity'].mean(),
                'avg_vader_compound': category_data['vader_compound'].mean(),
                'avg_bert_score': category_data['bert_score'].mean(),
                'avg_ensemble_score': category_data['ensemble_score'].mean(),
                'textblob_sentiment_dist': category_data['textblob_sentiment'].value_counts().to_dict(),
                'vader_sentiment_dist': category_data['vader_sentiment'].value_counts().to_dict(),
                'bert_sentiment_dist': category_data['bert_sentiment'].value_counts().to_dict(),
                'ensemble_sentiment_dist': category_data['ensemble_sentiment'].value_counts().to_dict()
            }
        
        return category_sentiment
    
    def get_fashion_specific_sentiment_insights(self, df: pd.DataFrame, text_df: pd.DataFrame) -> Dict:
        """
        Get fashion-specific sentiment insights
        
        Args:
            df: DataFrame with sentiment results
            text_df: DataFrame with original text data
            
        Returns:
            Dictionary with fashion-specific insights
        """
        # Combine sentiment and text data
        combined_df = pd.concat([df, text_df], axis=1)
        
        # Fashion-specific keywords for analysis
        fashion_keywords = {
            'positive_trends': ['innovative', 'trendy', 'stylish', 'fashionable', 'popular', 'emerging'],
            'negative_trends': ['declining', 'outdated', 'unfashionable', 'boring', 'weak'],
            'colors': ['black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple'],
            'materials': ['cotton', 'silk', 'wool', 'denim', 'leather', 'synthetic'],
            'styles': ['casual', 'formal', 'sporty', 'elegant', 'vintage', 'modern']
        }
        
        insights = {}
        
        for category, keywords in fashion_keywords.items():
            category_results = []
            
            for keyword in keywords:
                # Find documents containing the keyword
                keyword_docs = combined_df[combined_df['processed_text'].str.contains(keyword, case=False, na=False)]
                
                if len(keyword_docs) > 0:
                    avg_sentiment = keyword_docs['textblob_polarity'].mean()
                    doc_count = len(keyword_docs)
                    
                    category_results.append({
                        'keyword': keyword,
                        'document_count': doc_count,
                        'average_sentiment': avg_sentiment,
                        'sentiment_classification': 'Positive' if avg_sentiment > 0.1 else 'Negative' if avg_sentiment < -0.1 else 'Neutral'
                    })
            
            insights[category] = category_results
        
        return insights
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded models
        
        Returns:
            Dictionary with model information
        """
        info = {
            'textblob': 'TextBlob sentiment analyzer',
            'vader': 'VADER sentiment analyzer',
            'bert_model': 'distilbert-base-uncased-finetuned-sst-2-english',
            'bert_available': self.bert_sentiment_pipeline is not None,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        return info

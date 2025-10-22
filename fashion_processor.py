# fashion_processor.py
import pandas as pd
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer
# from bertopic import BERTopic  # Commented out due to compilation issues with hdbscan
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import flair
from flair.models import SequenceTagger
from flair.data import Sentence
import re
from collections import Counter
import warnings
import os
import json
warnings.filterwarnings('ignore')

class FashionProcessor:
    def __init__(self, use_fine_tuned_models=True, model_dir="./fine_tuned_models"):
        self.model_dir = model_dir
        self.use_fine_tuned_models = use_fine_tuned_models
        
        print("🚀 Loading fashion analysis models...")
        
        # Load fine-tuned models if available and requested
        if use_fine_tuned_models and self._check_fine_tuned_models():
            self._load_fine_tuned_models()
        else:
            self._load_pretrained_models()
        
        # Fashion parameters
        self.fashion_parameters = {
            'season': {
                'summer': ['summer', 'hot', 'beach', 'vacation', 'sun', 'lightweight', 'breathable'],
                'winter': ['winter', 'cold', 'snow', 'warm', 'layered', 'cozy', 'insulated'],
                'spring': ['spring', 'bloom', 'fresh', 'renewal', 'light', 'transitional'],
                'fall': ['fall', 'autumn', 'crisp', 'harvest', 'layering', 'transition']
            },
            'palette': {
                'pastel': ['pastel', 'soft', 'light', 'muted', 'delicate', 'pale'],
                'neon': ['neon', 'bright', 'fluorescent', 'electric', 'vibrant'],
                'earth_tones': ['earth', 'neutral', 'natural', 'organic', 'earthy', 'taupe', 'beige'],
                'monochrome': ['monochrome', 'black', 'white', 'gray', 'grayscale', 'single color'],
                'jewel': ['jewel', 'rich', 'sapphire', 'emerald', 'ruby', 'amethyst', 'deep'],
                'warm': ['warm', 'golden', 'amber', 'terracotta', 'rust', 'spice'],
                'metallic': ['metallic', 'gold', 'silver', 'bronze', 'shiny', 'glitter']
            },
            'style_theme': {
                'boho': ['boho', 'bohemian', 'hippie', 'flowy', 'ethnic', 'artistic'],
                'y2k': ['y2k', 'retro', '2000s', 'nostalgic', 'vintage', 'throwback'],
                'minimalist': ['minimalist', 'simple', 'clean', 'essential', 'basic'],
                'vintage': ['vintage', 'retro', 'classic', 'old school', 'heritage'],
                'elegant_chic': ['elegant', 'chic', 'sophisticated', 'refined', 'polished']
            }
        }
        
        print("✅ All models loaded successfully!")
    
    def _check_fine_tuned_models(self):
        """Check if fine-tuned models exist"""
        sentiment_path = os.path.join(self.model_dir, 'fashion_sentiment_model')
        embedding_path = os.path.join(self.model_dir, 'fashion_embedding_model')
        entities_path = os.path.join(self.model_dir, 'fashion_entity_patterns.json')
        
        return all([
            os.path.exists(sentiment_path),
            os.path.exists(embedding_path),
            os.path.exists(entities_path)
        ])
    
    def _load_fine_tuned_models(self):
        """Load fine-tuned models"""
        print("🔄 Loading fine-tuned models...")
        
        try:
            # Load fine-tuned sentiment model
            sentiment_path = os.path.join(self.model_dir, 'fashion_sentiment_model')
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=sentiment_path,
                tokenizer=sentiment_path
            )
            print("✅ Fine-tuned sentiment model loaded")
            
            # Load fine-tuned embedding model
            embedding_path = os.path.join(self.model_dir, 'fashion_embedding_model')
            self.sentence_model = SentenceTransformer(embedding_path)
            print("✅ Fine-tuned embedding model loaded")
            
            # Load fashion entity patterns
            entities_path = os.path.join(self.model_dir, 'fashion_entity_patterns.json')
            with open(entities_path, 'r', encoding='utf-8') as f:
                self.fashion_entity_patterns = json.load(f)
            print("✅ Fashion entity patterns loaded")
            
            self.fine_tuned_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading fine-tuned models: {e}")
            print("🔄 Falling back to pretrained models...")
            self._load_pretrained_models()
    
    def _load_pretrained_models(self):
        """Load standard pretrained models"""
        print("🔄 Loading pretrained models...")
        
        # Sentiment Analysis
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        # Sentence Transformer
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Alternative Topic Modeling using scikit-learn (replaces BERTopic)
        print("📝 Loading topic modeling components...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.kmeans_model = KMeans(n_clusters=5, random_state=42)
        
        # KeyBERT
        self.keyword_extractor = KeyBERT(model=self.sentence_model)
        
        # Flair NER
        self.ner_tagger = SequenceTagger.load("flair/ner-english-large")
        
        # Default fashion entity patterns
        self.fashion_entity_patterns = {
            'brands': [], 'materials': [], 'styles': [], 'colors': [], 'garments': [], 'accessories': []
        }
        
        self.fine_tuned_loaded = False
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using either fine-tuned or pretrained model"""
        try:
            if len(text) > 500:
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                sentiments = []
                
                for chunk in chunks[:3]:
                    result = self.sentiment_analyzer(chunk[:512])[0]
                    sentiments.append(result)
            else:
                sentiments = [self.sentiment_analyzer(text[:512])[0]]
            
            sentiment_scores = {
                'POSITIVE': 0,
                'NEGATIVE': 0, 
                'NEUTRAL': 0
            }
            
            for sentiment in sentiments:
                label = sentiment['label'].upper()  # Convert to uppercase
                score = sentiment['score']
                sentiment_scores[label] += score
            
            overall_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            confidence = sentiment_scores[overall_sentiment] / len(sentiments)
            
            model_type = "Fine-tuned Fashion Model" if self.fine_tuned_loaded else "Pretrained Model"
            
            return {
                'overall_sentiment': overall_sentiment,
                'confidence': confidence,
                'detailed_scores': sentiment_scores,
                'model': model_type,
                'model_type': 'fine_tuned' if self.fine_tuned_loaded else 'pretrained'
            }
            
        except Exception as e:
            return {'error': f"Sentiment analysis failed: {str(e)}"}
    
    def extract_keywords(self, text, top_n=20):
        """Extract fashion keywords"""
        try:
            keywords = self.keyword_extractor.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n,
                diversity=0.7
            )
            
            return {
                'keywords': keywords,
                'total_keywords': len(keywords),
                'model': 'KeyBERT'
            }
            
        except Exception as e:
            return {'error': f"Keyword extraction failed: {str(e)}"}
    
    def extract_entities(self, text):
        """Extract fashion entities"""
        try:
            sentence = Sentence(text)
            self.ner_tagger.predict(sentence)
            
            entities = []
            for entity in sentence.get_spans('ner'):
                entity_text = entity.text.lower()
                
                # Enhanced fashion entity detection
                fashion_category = self._categorize_fashion_entity_enhanced(entity_text, entity.tag)
                
                entities.append({
                    'text': entity.text,
                    'label': entity.tag,
                    'score': entity.score,
                    'category': fashion_category,
                    'is_fashion_related': fashion_category != 'other'
                })
            
            # Add fashion-specific entity recognition using our patterns
            fashion_entities = self._extract_fashion_entities_from_patterns(text)
            entities.extend(fashion_entities)
            
            entity_categories = {}
            for entity in entities:
                category = entity['category']
                if category not in entity_categories:
                    entity_categories[category] = []
                entity_categories[category].append(entity)
            
            return {
                'entities': entities,
                'categories': entity_categories,
                'total_entities': len(entities),
                'fashion_entities_count': len([e for e in entities if e['is_fashion_related']]),
                'model': 'Flair NER + Fashion Patterns'
            }
            
        except Exception as e:
            return {'error': f"NER failed: {str(e)}"}
    
    def _categorize_fashion_entity_enhanced(self, entity_text, entity_label):
        """Enhanced fashion entity categorization"""
        for category, patterns in self.fashion_entity_patterns.items():
            if any(pattern in entity_text for pattern in patterns):
                return f"fashion_{category}"
        
        return self._categorize_fashion_entity(entity_text, entity_label)
    
    def _extract_fashion_entities_from_patterns(self, text):
        """Extract fashion entities using patterns"""
        entities = []
        text_lower = text.lower()
        
        for category, patterns in self.fashion_entity_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    entities.append({
                        'text': pattern.title(),
                        'label': 'FASHION',
                        'score': 0.9,
                        'category': f'fashion_{category}',
                        'is_fashion_related': True,
                        'source': 'pattern_matching'
                    })
        
        return entities
    
    def _categorize_fashion_entity(self, entity_text, entity_label):
        """Categorize entity into fashion categories"""
        entity_lower = entity_text.lower()
        
        for category, options in self.fashion_parameters.items():
            for option_name, keywords in options.items():
                if any(keyword in entity_lower for keyword in keywords):
                    return f"{category}_{option_name}"
        
        return entity_label.lower()
    
    def extract_topics(self, texts):
        """Extract topics using BERTopic"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            topics, probabilities = self.topic_model.fit_transform(texts)
            topic_info = self.topic_model.get_topic_info()
            
            topic_results = []
            for topic_id in topic_info['Topic'].unique():
                if topic_id != -1:
                    topic_words = self.topic_model.get_topic(topic_id)
                    
                    topic_results.append({
                        'topic_id': topic_id,
                        'topic_size': len([t for t in topics if t == topic_id]),
                        'keywords': [(word, score) for word, score in topic_words[:8]],
                        'topic_name': f"Topic_{topic_id}"
                    })
            
            return {
                'topics': topic_results,
                'total_topics': len(topic_results),
                'model': 'BERTopic'
            }
            
        except Exception as e:
            return {'error': f"Topic modeling failed: {str(e)}"}
    
    def analyze_fashion_parameters(self, text):
        """Analyze fashion parameters"""
        text_lower = text.lower()
        parameter_analysis = {}
        
        for param_category, param_options in self.fashion_parameters.items():
            parameter_analysis[param_category] = {}
            
            for param_name, keywords in param_options.items():
                score = sum(1 for keyword in keywords if keyword in text_lower)
                if score > 0:
                    parameter_analysis[param_category][param_name] = {
                        'score': score,
                        'confidence': min(score / len(keywords), 1.0),
                        'mentions': [kw for kw in keywords if kw in text_lower]
                    }
        
        return parameter_analysis
    
    def analyze_fashion_trends(self, text):
        """Comprehensive fashion trend analysis"""
        print("🔍 Analyzing fashion trends...")
        
        results = {
            'sentiment_analysis': self.analyze_sentiment(text),
            'topic_modeling': self.extract_topics([text]),
            'keyword_extraction': self.extract_keywords(text),
            'entity_recognition': self.extract_entities(text),
            'fashion_parameters': self.analyze_fashion_parameters(text)
        }
        
        return results
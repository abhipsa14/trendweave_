"""
Prediction and Forecasting Module for Fashion Trend Analysis
Combines all NLP results to make predictions and forecasts
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)

class FashionTrendPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_importance = {}
        
    def create_feature_matrix(self, 
                            sentiment_df: pd.DataFrame,
                            topic_results: Dict,
                            keyword_df: pd.DataFrame,
                            entity_df: pd.DataFrame,
                            text_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create feature matrix by combining all NLP analysis results
        
        Args:
            sentiment_df: Sentiment analysis results
            topic_results: Topic modeling results
            keyword_df: Keyword extraction results
            entity_df: NER analysis results
            text_df: Original text data with metadata
            
        Returns:
            Combined feature matrix
        """
        logger.info("Creating feature matrix...")
        
        # Start with basic text features
        features_df = text_df[['filename', 'category', 'season', 'location', 'text_length', 'word_count']].copy()
        
        # Add sentiment features
        sentiment_features = sentiment_df[['textblob_polarity', 'textblob_subjectivity', 
                                         'vader_compound', 'vader_positive', 'vader_negative']]
        features_df = pd.concat([features_df, sentiment_features], axis=1)
        
        # Add topic features
        topic_assignments = pd.DataFrame(topic_results['document_topics'])
        features_df['dominant_topic'] = topic_assignments['topic_id']
        
        # Add keyword-based features
        features_df['num_rake_keywords'] = keyword_df['rake_keywords'].apply(len)
        features_df['num_keybert_keywords'] = keyword_df['keybert_keywords'].apply(len)
        features_df['avg_rake_score'] = keyword_df['rake_scores'].apply(lambda x: np.mean(x) if x else 0)
        features_df['avg_keybert_score'] = keyword_df['keybert_scores'].apply(lambda x: np.mean(x) if x else 0)
        
        # Add entity-based features
        features_df['total_entities'] = entity_df['total_entities']
        features_df['unique_entity_types'] = entity_df['unique_entity_types']
        
        # Add fashion-specific entity counts
        for _, row in entity_df.iterrows():
            fashion_entities = row['fashion_entities']
            features_df.loc[row.name, 'brand_count'] = len(fashion_entities.get('BRAND', []))
            features_df.loc[row.name, 'color_count'] = len(fashion_entities.get('COLOR', []))
            features_df.loc[row.name, 'material_count'] = len(fashion_entities.get('MATERIAL', []))
            features_df.loc[row.name, 'style_count'] = len(fashion_entities.get('STYLE', []))
            features_df.loc[row.name, 'garment_count'] = len(fashion_entities.get('GARMENT', []))
            features_df.loc[row.name, 'location_count'] = len(row['fashion_locations'])
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        return features_df
    
    def create_trend_labels(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend labels based on various criteria
        
        Args:
            features_df: Feature matrix
            
        Returns:
            DataFrame with trend labels
        """
        logger.info("Creating trend labels...")
        
        # Create multiple types of labels for different prediction tasks
        
        # 1. Sentiment-based trend (Positive/Negative/Neutral)
        features_df['sentiment_trend'] = features_df['textblob_polarity'].apply(
            lambda x: 'Positive' if x > 0.1 else 'Negative' if x < -0.1 else 'Neutral'
        )
        
        # 2. Season-based trend (Current/Emerging)
        features_df['season_trend'] = features_df['season'].apply(
            lambda x: 'Current' if x == 'AW_24_25' else 'Emerging'
        )
        
        # 3. Category-based popularity (High/Medium/Low)
        # Based on entity count and sentiment
        features_df['popularity_score'] = (
            features_df['total_entities'] * 0.3 +
            features_df['brand_count'] * 0.2 +
            features_df['color_count'] * 0.1 +
            features_df['material_count'] * 0.1 +
            features_df['textblob_polarity'] * 0.3
        )
        
        # Normalize popularity score
        features_df['popularity_score'] = (features_df['popularity_score'] - features_df['popularity_score'].min()) / \
                                         (features_df['popularity_score'].max() - features_df['popularity_score'].min())
        
        features_df['popularity_trend'] = pd.cut(
            features_df['popularity_score'],
            bins=3,
            labels=['Low', 'Medium', 'High']
        )
        
        # 4. Innovation trend (Traditional/Moderate/Innovative)
        # Based on keyword diversity and entity types
        features_df['innovation_score'] = (
            features_df['num_keybert_keywords'] * 0.3 +
            features_df['unique_entity_types'] * 0.3 +
            features_df['style_count'] * 0.4
        )
        
        # Normalize innovation score
        features_df['innovation_score'] = (features_df['innovation_score'] - features_df['innovation_score'].min()) / \
                                         (features_df['innovation_score'].max() - features_df['innovation_score'].min())
        
        features_df['innovation_trend'] = pd.cut(
            features_df['innovation_score'],
            bins=3,
            labels=['Traditional', 'Moderate', 'Innovative']
        )
        
        return features_df
    
    def train_prediction_models(self, features_df: pd.DataFrame) -> Dict:
        """
        Train prediction models for different trend types
        
        Args:
            features_df: Feature matrix with labels
            
        Returns:
            Dictionary with model results
        """
        logger.info("Training prediction models...")
        
        # Prepare features
        feature_columns = [
            'text_length', 'word_count', 'textblob_polarity', 'textblob_subjectivity',
            'vader_compound', 'vader_positive', 'vader_negative', 'dominant_topic',
            'num_rake_keywords', 'num_keybert_keywords', 'avg_rake_score', 'avg_keybert_score',
            'total_entities', 'unique_entity_types', 'brand_count', 'color_count',
            'material_count', 'style_count', 'garment_count', 'location_count'
        ]
        
        # Encode categorical features
        le_category = LabelEncoder()
        le_location = LabelEncoder()
        
        features_df['category_encoded'] = le_category.fit_transform(features_df['category'])
        features_df['location_encoded'] = le_location.fit_transform(features_df['location'])
        
        feature_columns.extend(['category_encoded', 'location_encoded'])
        
        X = features_df[feature_columns]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        self.scalers['main'] = scaler
        self.label_encoders['category'] = le_category
        self.label_encoders['location'] = le_location
        
        # Train models for different prediction tasks
        prediction_tasks = {
            'sentiment_trend': features_df['sentiment_trend'],
            'season_trend': features_df['season_trend'],
            'popularity_trend': features_df['popularity_trend'],
            'innovation_trend': features_df['innovation_trend']
        }
        
        results = {}
        
        for task_name, y in prediction_tasks.items():
            logger.info(f"Training models for {task_name}...")
            
            # Skip if all labels are the same
            if len(y.unique()) <= 1:
                logger.warning(f"Skipping {task_name} - insufficient label diversity")
                continue
            
            # Encode labels
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y)
            self.label_encoders[task_name] = le_target
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            # Train multiple models
            models = {
                'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
            }
            
            task_results = {}
            
            for model_name, model in models.items():
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate accuracy
                    accuracy = accuracy_score(y_test, y_pred)
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
                    
                    # Get feature importance (if available)
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(feature_columns, model.feature_importances_))
                    else:
                        feature_importance = {}
                    
                    task_results[model_name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'feature_importance': feature_importance,
                        'predictions': y_pred,
                        'true_labels': y_test
                    }
                    
                    logger.info(f"{model_name} - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name} for {task_name}: {e}")
                    continue
            
            results[task_name] = task_results
            
            # Store best model
            if task_results:
                best_model_name = max(task_results.keys(), key=lambda x: task_results[x]['accuracy'])
                self.models[task_name] = task_results[best_model_name]['model']
                self.feature_importance[task_name] = task_results[best_model_name]['feature_importance']
        
        return results
    
    def make_predictions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions using trained models
        
        Args:
            features_df: Feature matrix
            
        Returns:
            DataFrame with predictions
        """
        logger.info("Making predictions...")
        
        # Prepare features
        feature_columns = [
            'text_length', 'word_count', 'textblob_polarity', 'textblob_subjectivity',
            'vader_compound', 'vader_positive', 'vader_negative', 'dominant_topic',
            'num_rake_keywords', 'num_keybert_keywords', 'avg_rake_score', 'avg_keybert_score',
            'total_entities', 'unique_entity_types', 'brand_count', 'color_count',
            'material_count', 'style_count', 'garment_count', 'location_count',
            'category_encoded', 'location_encoded'
        ]
        
        X = features_df[feature_columns]
        X_scaled = self.scalers['main'].transform(X)
        
        predictions_df = features_df[['filename', 'category', 'season', 'location']].copy()
        
        # Make predictions for each task
        for task_name, model in self.models.items():
            try:
                y_pred = model.predict(X_scaled)
                # Decode predictions
                y_pred_decoded = self.label_encoders[task_name].inverse_transform(y_pred)
                predictions_df[f'{task_name}_prediction'] = y_pred_decoded
                
                # Add prediction probabilities if available
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)
                    max_proba = np.max(proba, axis=1)
                    predictions_df[f'{task_name}_confidence'] = max_proba
            
            except Exception as e:
                logger.error(f"Error making predictions for {task_name}: {e}")
                continue
        
        return predictions_df
    
    def analyze_trend_patterns(self, predictions_df: pd.DataFrame) -> Dict:
        """
        Analyze patterns in trend predictions
        
        Args:
            predictions_df: DataFrame with predictions
            
        Returns:
            Dictionary with trend analysis
        """
        logger.info("Analyzing trend patterns...")
        
        analysis = {}
        
        # Analyze by category
        category_analysis = {}
        for category in predictions_df['category'].unique():
            category_data = predictions_df[predictions_df['category'] == category]
            
            category_analysis[category] = {
                'document_count': len(category_data),
                'sentiment_distribution': category_data['sentiment_trend_prediction'].value_counts().to_dict() if 'sentiment_trend_prediction' in category_data.columns else {},
                'popularity_distribution': category_data['popularity_trend_prediction'].value_counts().to_dict() if 'popularity_trend_prediction' in category_data.columns else {},
                'innovation_distribution': category_data['innovation_trend_prediction'].value_counts().to_dict() if 'innovation_trend_prediction' in category_data.columns else {}
            }
        
        analysis['category_analysis'] = category_analysis
        
        # Analyze by season
        season_analysis = {}
        for season in predictions_df['season'].unique():
            season_data = predictions_df[predictions_df['season'] == season]
            
            season_analysis[season] = {
                'document_count': len(season_data),
                'sentiment_distribution': season_data['sentiment_trend_prediction'].value_counts().to_dict() if 'sentiment_trend_prediction' in season_data.columns else {},
                'popularity_distribution': season_data['popularity_trend_prediction'].value_counts().to_dict() if 'popularity_trend_prediction' in season_data.columns else {},
                'innovation_distribution': season_data['innovation_trend_prediction'].value_counts().to_dict() if 'innovation_trend_prediction' in season_data.columns else {}
            }
        
        analysis['season_analysis'] = season_analysis
        
        # Overall trend summary
        overall_trends = {}
        for col in predictions_df.columns:
            if col.endswith('_prediction'):
                trend_type = col.replace('_prediction', '')
                overall_trends[trend_type] = predictions_df[col].value_counts().to_dict()
        
        analysis['overall_trends'] = overall_trends
        
        return analysis
    
    def generate_forecasts(self, predictions_df: pd.DataFrame, trend_analysis: Dict) -> Dict:
        """
        Generate fashion trend forecasts
        
        Args:
            predictions_df: DataFrame with predictions
            trend_analysis: Trend analysis results
            
        Returns:
            Dictionary with forecasts
        """
        logger.info("Generating forecasts...")
        
        forecasts = {}
        
        # Color trend forecasts
        color_trends = {}
        for season in ['AW_24_25', 'AW_25_26']:
            season_data = predictions_df[predictions_df['season'] == season]
            if len(season_data) > 0:
                # Get dominant trends for this season
                sentiment_dist = season_data['sentiment_trend_prediction'].value_counts() if 'sentiment_trend_prediction' in season_data.columns else {}
                popularity_dist = season_data['popularity_trend_prediction'].value_counts() if 'popularity_trend_prediction' in season_data.columns else {}
                
                color_trends[season] = {
                    'dominant_sentiment': sentiment_dist.index[0] if len(sentiment_dist) > 0 else 'Neutral',
                    'dominant_popularity': popularity_dist.index[0] if len(popularity_dist) > 0 else 'Medium',
                    'trend_strength': len(season_data)
                }
        
        forecasts['color_trends'] = color_trends
        
        # Category forecasts
        category_forecasts = {}
        for category in predictions_df['category'].unique():
            category_data = predictions_df[predictions_df['category'] == category]
            
            # Calculate trend scores
            positive_sentiment = len(category_data[category_data['sentiment_trend_prediction'] == 'Positive']) if 'sentiment_trend_prediction' in category_data.columns else 0
            high_popularity = len(category_data[category_data['popularity_trend_prediction'] == 'High']) if 'popularity_trend_prediction' in category_data.columns else 0
            innovative = len(category_data[category_data['innovation_trend_prediction'] == 'Innovative']) if 'innovation_trend_prediction' in category_data.columns else 0
            
            total_docs = len(category_data)
            
            if total_docs > 0:
                trend_score = (positive_sentiment + high_popularity + innovative) / (total_docs * 3)
                
                if trend_score >= 0.6:
                    forecast = 'Strong Growth'
                elif trend_score >= 0.4:
                    forecast = 'Moderate Growth'
                elif trend_score >= 0.2:
                    forecast = 'Stable'
                else:
                    forecast = 'Declining'
                
                category_forecasts[category] = {
                    'forecast': forecast,
                    'trend_score': trend_score,
                    'confidence': min(total_docs / 10, 1.0)  # Confidence based on sample size
                }
        
        forecasts['category_forecasts'] = category_forecasts
        
        # Overall market forecast
        total_docs = len(predictions_df)
        if total_docs > 0:
            positive_sentiment = len(predictions_df[predictions_df['sentiment_trend_prediction'] == 'Positive']) if 'sentiment_trend_prediction' in predictions_df.columns else 0
            high_innovation = len(predictions_df[predictions_df['innovation_trend_prediction'] == 'Innovative']) if 'innovation_trend_prediction' in predictions_df.columns else 0
            
            market_sentiment = positive_sentiment / total_docs
            innovation_rate = high_innovation / total_docs
            
            forecasts['market_forecast'] = {
                'overall_sentiment': 'Positive' if market_sentiment > 0.5 else 'Negative' if market_sentiment < 0.3 else 'Neutral',
                'innovation_level': 'High' if innovation_rate > 0.4 else 'Low' if innovation_rate < 0.2 else 'Medium',
                'market_confidence': (market_sentiment + innovation_rate) / 2
            }
        
        return forecasts
    
    def get_model_performance(self, model_results: Dict) -> Dict:
        """
        Get performance metrics for all models
        
        Args:
            model_results: Results from model training
            
        Returns:
            Dictionary with performance metrics
        """
        performance = {}
        
        for task_name, task_results in model_results.items():
            task_performance = {}
            
            for model_name, model_result in task_results.items():
                task_performance[model_name] = {
                    'accuracy': model_result['accuracy'],
                    'cv_mean': model_result['cv_mean'],
                    'cv_std': model_result['cv_std'],
                    'top_features': sorted(model_result['feature_importance'].items(), 
                                         key=lambda x: x[1], reverse=True)[:10] if model_result['feature_importance'] else []
                }
            
            performance[task_name] = task_performance
        
        return performance

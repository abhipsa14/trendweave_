"""
Main Fashion Trend Analysis Script
Orchestrates all NLP modules to perform comprehensive fashion trend analysis
"""

import os
import sys
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add the nlp_pipeline directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'nlp_pipeline'))

# Import all modules
from nlp_pipeline.data_loader import FashionDataLoader
from nlp_pipeline.sentiment_analyzer import FashionSentimentAnalyzer
from nlp_pipeline.topic_modeling import FashionTopicModeler
from nlp_pipeline.keyword_extraction import FashionKeywordExtractor
from nlp_pipeline.ner_analyzer import FashionNERAnalyzer
from nlp_pipeline.trend_predictor import FashionTrendPredictor
from nlp_pipeline.report_generator import FashionTrendReportGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fashion_trend_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function to run the complete fashion trend analysis pipeline
    """
    try:
        logger.info("Starting Fashion Trend Analysis Pipeline")
        logger.info("=" * 60)
        
        # 1. Data Loading
        logger.info("STEP 1: Loading and preprocessing data...")
        data_loader = FashionDataLoader("extracted_content")
        
        # Load all text files
        texts, filenames = data_loader.load_all_texts()
        
        if not texts:
            logger.error("No text files found in extracted_content directory!")
            return
        
        # Get processed data
        text_df = data_loader.get_processed_data()
        text_df = data_loader.categorize_documents(text_df)
        
        # Get summary statistics
        data_stats = data_loader.get_summary_statistics(text_df)
        
        logger.info(f"Loaded {len(texts)} documents")
        logger.info(f"Total words: {data_stats['total_words']:,}")
        logger.info("Data loading completed!")
        
        # 2. Sentiment Analysis
        logger.info("\\nSTEP 2: Performing sentiment analysis...")
        sentiment_analyzer = FashionSentimentAnalyzer()
        
        # Analyze sentiment for all texts
        sentiment_df = sentiment_analyzer.analyze_batch_sentiment(text_df['processed_text'].tolist())
        
        # Get sentiment summary
        sentiment_summary = sentiment_analyzer.get_sentiment_summary(sentiment_df)
        
        # Get sentiment by category
        combined_sentiment_df = sentiment_df.copy()
        combined_sentiment_df['category'] = text_df['category'].values
        sentiment_by_category = sentiment_analyzer.analyze_sentiment_by_category(combined_sentiment_df, 'category')
        
        # Get fashion-specific sentiment insights
        fashion_sentiment_insights = sentiment_analyzer.get_fashion_specific_sentiment_insights(sentiment_df, text_df)
        
        logger.info("Sentiment analysis completed!")
        
        # 3. Topic Modeling
        logger.info("\\nSTEP 3: Performing topic modeling...")
        topic_modeler = FashionTopicModeler()
        
        # Train BERTopic model
        bertopic_results = topic_modeler.get_bertopic_results(text_df['processed_text'].tolist(), filenames)
        
        # Analyze topic trends
        topic_trends = topic_modeler.analyze_topic_trends(bertopic_results, text_df)
        
        logger.info("Topic modeling completed!")
        
        # 4. Keyword Extraction
        logger.info("\\nSTEP 4: Performing keyword extraction...")
        keyword_extractor = FashionKeywordExtractor()
        
        # Extract keywords using all methods
        keyword_df = keyword_extractor.extract_all_keywords(text_df['processed_text'].tolist(), filenames)
        
        # Analyze keyword trends
        keyword_trends = keyword_extractor.analyze_keyword_trends(keyword_df, text_df)
        
        # Get fashion-specific keywords
        fashion_keywords = keyword_extractor.get_fashion_specific_keywords(keyword_df, text_df)
        
        logger.info("Keyword extraction completed!")
        
        # 5. Named Entity Recognition
        logger.info("\\nSTEP 5: Performing named entity recognition...")
        ner_analyzer = FashionNERAnalyzer()
        
        # Analyze entities for all texts
        entity_df = ner_analyzer.analyze_batch_entities(text_df['processed_text'].tolist(), filenames)
        
        # Analyze entity trends
        entity_trends = ner_analyzer.analyze_entity_trends(entity_df, text_df)
        
        # Get specific analyses
        brand_analysis = ner_analyzer.get_brand_analysis(entity_df, text_df)
        color_analysis = ner_analyzer.get_color_trend_analysis(entity_df, text_df)
        material_analysis = ner_analyzer.get_material_analysis(entity_df, text_df)
        
        logger.info("Named entity recognition completed!")
        
        # 6. Prediction and Forecasting
        logger.info("\\nSTEP 6: Performing prediction and forecasting...")
        trend_predictor = FashionTrendPredictor()
        
        # Create feature matrix
        features_df = trend_predictor.create_feature_matrix(
            sentiment_df, bertopic_results, keyword_df, entity_df, text_df
        )
        
        # Create trend labels
        features_df = trend_predictor.create_trend_labels(features_df)
        
        # Train prediction models
        model_results = trend_predictor.train_prediction_models(features_df)
        
        # Make predictions
        predictions_df = trend_predictor.make_predictions(features_df)
        
        # Analyze trend patterns
        trend_analysis = trend_predictor.analyze_trend_patterns(predictions_df)
        
        # Generate forecasts
        forecasts = trend_predictor.generate_forecasts(predictions_df, trend_analysis)
        
        # Get model performance
        model_performance = trend_predictor.get_model_performance(model_results)
        
        logger.info("Prediction and forecasting completed!")
        
        # 7. Report Generation
        logger.info("\\nSTEP 7: Generating comprehensive report...")
        report_generator = FashionTrendReportGenerator()
        
        # Generate complete report
        report_text = report_generator.generate_full_report(
            data_stats=data_stats,
            sentiment_summary=sentiment_summary,
            sentiment_by_category=sentiment_by_category,
            fashion_sentiment_insights=fashion_sentiment_insights,
            bertopic_results=bertopic_results,
            topic_trends=topic_trends,
            keyword_trends=keyword_trends,
            fashion_keywords=fashion_keywords,
            entity_trends=entity_trends,
            brand_analysis=brand_analysis,
            color_analysis=color_analysis,
            material_analysis=material_analysis,
            predictions_df=predictions_df,
            trend_analysis=trend_analysis,
            forecasts=forecasts,
            model_performance=model_performance
        )
        
        # Save report
        report_filename = f"fashion_trend_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        report_generator.save_report(report_text, report_filename)
        
        # Save results as JSON
        results_dict = {
            'data_stats': data_stats,
            'sentiment_summary': sentiment_summary,
            'sentiment_by_category': sentiment_by_category,
            'fashion_sentiment_insights': fashion_sentiment_insights,
            'topic_trends': topic_trends,
            'keyword_trends': keyword_trends,
            'fashion_keywords': fashion_keywords,
            'entity_trends': entity_trends,
            'brand_analysis': brand_analysis,
            'color_analysis': color_analysis,
            'material_analysis': material_analysis,
            'trend_analysis': trend_analysis,
            'forecasts': forecasts,
            'model_performance': model_performance
        }
        
        results_filename = f"fashion_trend_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_generator.save_results_json(results_dict, results_filename)
        
        logger.info("Report generation completed!")
        
        # 8. Print Summary to Console
        logger.info("\\n" + "=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        print("\\n" + report_text)
        
        logger.info("\\n" + "=" * 60)
        logger.info("Fashion Trend Analysis Pipeline Completed Successfully!")
        logger.info(f"Report saved to: {report_filename}")
        logger.info(f"Results saved to: {results_filename}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Error in main analysis pipeline: {e}")
        raise

if __name__ == "__main__":
    main()

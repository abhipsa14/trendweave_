"""
Report Generator Module for Fashion Trend Analysis
Generates comprehensive reports combining all analysis results
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class FashionTrendReportGenerator:
    def __init__(self):
        self.report_sections = []
        self.executive_summary = {}
        
    def generate_executive_summary(self, 
                                 data_stats: Dict,
                                 sentiment_summary: Dict,
                                 trend_analysis: Dict,
                                 forecasts: Dict,
                                 model_performance: Dict) -> str:
        """
        Generate executive summary section
        
        Args:
            data_stats: Data loading statistics
            sentiment_summary: Sentiment analysis summary
            trend_analysis: Trend analysis results
            forecasts: Forecast results
            model_performance: Model performance metrics
            
        Returns:
            Executive summary text
        """
        summary = []
        summary.append("EXECUTIVE SUMMARY")
        summary.append("=" * 80)
        summary.append("")
        
        # Data overview
        summary.append(f"📊 DATA OVERVIEW")
        summary.append(f"   • Total documents analyzed: {data_stats['total_documents']}")
        summary.append(f"   • Total words processed: {data_stats['total_words']:,}")
        summary.append(f"   • Average document length: {data_stats['average_document_length']:.0f} characters")
        summary.append("")
        
        # Category distribution
        summary.append("📁 CATEGORY DISTRIBUTION")
        for category, count in data_stats['category_distribution'].items():
            percentage = (count / data_stats['total_documents']) * 100
            summary.append(f"   • {category}: {count} documents ({percentage:.1f}%)")
        summary.append("")
        
        # Season distribution
        summary.append("🗓️ SEASON DISTRIBUTION")
        for season, count in data_stats['season_distribution'].items():
            percentage = (count / data_stats['total_documents']) * 100
            summary.append(f"   • {season}: {count} documents ({percentage:.1f}%)")
        summary.append("")
        
        # Overall sentiment
        summary.append("💭 OVERALL SENTIMENT")
        summary.append(f"   • Average sentiment score: {sentiment_summary['average_textblob_polarity']:.3f}")
        summary.append("   • Sentiment distribution:")
        for sentiment, count in sentiment_summary['textblob_sentiment_distribution'].items():
            summary.append(f"     - {sentiment}: {count} documents")
        summary.append("")
        
        # Key forecasts
        summary.append("🔮 KEY FORECASTS")
        market_forecast = forecasts.get('market_forecast', {})
        summary.append(f"   • Overall market sentiment: {market_forecast.get('overall_sentiment', 'Unknown')}")
        summary.append(f"   • Innovation level: {market_forecast.get('innovation_level', 'Unknown')}")
        summary.append(f"   • Market confidence: {market_forecast.get('market_confidence', 0):.1%}")
        summary.append("")

        
        # Model performance
        summary.append("🤖 MODEL PERFORMANCE")
        if model_performance:
            for task_name, task_perf in model_performance.items():
                best_model = max(task_perf.items(), key=lambda x: x[1]['accuracy'])
                summary.append(f"   • {task_name.replace('_', ' ').title()}: {best_model[1]['accuracy']:.1%} accuracy")
        summary.append("")
        
        return "\n".join(summary)
    
    def generate_sentiment_analysis_section(self, 
                                          sentiment_summary: Dict,
                                          sentiment_by_category: Dict,
                                          fashion_insights: Dict) -> str:
        """
        Generate sentiment analysis section
        
        Args:
            sentiment_summary: Sentiment analysis summary
            sentiment_by_category: Category-wise sentiment analysis
            fashion_insights: Fashion-specific sentiment insights
            
        Returns:
            Sentiment analysis section text
        """
        section = []
        section.append("SENTIMENT ANALYSIS")
        section.append("=" * 80)
        section.append("")
        
        # Overall sentiment metrics
        section.append("📈 OVERALL SENTIMENT METRICS")
        section.append(f"   • Average Polarity (TextBlob): {sentiment_summary['average_textblob_polarity']:.3f}")
        section.append(f"   • Average Compound Score (VADER): {sentiment_summary['average_vader_compound']:.3f}")
        section.append("")
        
        # Sentiment distribution
        section.append("📊 SENTIMENT DISTRIBUTION")
        section.append("   TextBlob Analysis:")
        for sentiment, count in sentiment_summary['textblob_sentiment_distribution'].items():
            section.append(f"     • {sentiment}: {count} documents")
        section.append("")
        section.append("   VADER Analysis:")
        for sentiment, count in sentiment_summary['vader_sentiment_distribution'].items():
            section.append(f"     • {sentiment}: {count} documents")
        section.append("")
        
        # Most positive and negative documents
        section.append("🏆 EXTREME SENTIMENT DOCUMENTS")
        most_positive = sentiment_summary['most_positive_textblob']
        most_negative = sentiment_summary['most_negative_textblob']
        section.append(f"   • Most positive document: ID {most_positive['document_id']} (score: {most_positive['score']:.3f})")
        section.append(f"   • Most negative document: ID {most_negative['document_id']} (score: {most_negative['score']:.3f})")
        section.append("")
        
        # Category-wise sentiment
        section.append("🏷️ SENTIMENT BY CATEGORY")
        for category, data in sentiment_by_category.items():
            section.append(f"   {category}:")
            section.append(f"     • Documents: {data['count']}")
            section.append(f"     • Avg. Polarity: {data['avg_textblob_polarity']:.3f}")
            section.append(f"     • Avg. VADER: {data['avg_vader_compound']:.3f}")
            section.append("")
        
        # Fashion-specific insights
        section.append("👗 FASHION-SPECIFIC SENTIMENT INSIGHTS")
        for category, insights in fashion_insights.items():
            if insights:
                section.append(f"   {category.replace('_', ' ').title()}:")
                for insight in insights[:3]:  # Show top 3
                    section.append(f"     • {insight['keyword']}: {insight['sentiment_classification']} "
                                 f"({insight['average_sentiment']:.3f})")
                section.append("")
        
        return "\n".join(section)
    
    def generate_topic_modeling_section(self, 
                                      bertopic_results: Dict,
                                      topic_trends: Dict) -> str:
        """
        Generate topic modeling section
        
        Args:
            bertopic_results: BERTopic analysis results
            topic_trends: Topic trend analysis
            
        Returns:
            Topic modeling section text
        """
        section = []
        section.append("TOPIC MODELING")
        section.append("=" * 80)
        section.append("")
        
        # Topic overview
        topic_info = bertopic_results['topic_info']
        section.append("📋 TOPIC OVERVIEW")
        section.append(f"   • Total topics identified: {len(topic_info)}")
        section.append(f"   • Documents processed: {len(bertopic_results['document_topics'])}")
        section.append("")
        
        # Top topics
        section.append("🔝 TOP TOPICS")
        for i, (topic_id, row) in enumerate(topic_info.head(10).iterrows()):
            if topic_id != -1:  # Skip outlier topic
                keywords = bertopic_results['topic_keywords'].get(topic_id, [])
                section.append(f"   Topic {topic_id}: {', '.join(keywords[:5])}")
                section.append(f"     • Documents: {row['Count']}")
                section.append("")
        
        # Category trends
        section.append("🏷️ TOPIC TRENDS BY CATEGORY")
        category_trends = topic_trends.get('category_trends', {})
        for category, data in category_trends.items():
            section.append(f"   {category}:")
            section.append(f"     • Documents: {data['document_count']}")
            section.append(f"     • Dominant topics: {data['dominant_topics'][:3]}")
            section.append("")
        
        # Season trends
        section.append("🗓️ TOPIC TRENDS BY SEASON")
        season_trends = topic_trends.get('season_trends', {})
        for season, data in season_trends.items():
            section.append(f"   {season}:")
            section.append(f"     • Documents: {data['document_count']}")
            section.append(f"     • Dominant topics: {data['dominant_topics'][:3]}")
            section.append("")
        
        return "\n".join(section)

    def generate_keyword_analysis_section(self, 
                                        keyword_trends: Dict,
                                        fashion_keywords: Dict) -> str:
        """
        Generate keyword analysis section
        
        Args:
            keyword_trends: Keyword trend analysis
            fashion_keywords: Fashion-specific keyword analysis
            
        Returns:
            Keyword analysis section text
        """
        section = []
        section.append("KEYWORD ANALYSIS")
        section.append("=" * 80)
        section.append("")
        
        # Category trends
        section.append("🏷️ KEYWORD TRENDS BY CATEGORY")
        category_trends = keyword_trends.get('category_trends', {})
        for category, data in category_trends.items():
            section.append(f"   {category}:")
            section.append(f"     • Documents: {data['document_count']}")
            section.append(f"     • Unique RAKE keywords: {data['total_unique_rake_keywords']}")
            section.append(f"     • Unique KeyBERT keywords: {data['total_unique_keybert_keywords']}")
            section.append("     • Top RAKE keywords:")
            for kw, count in data['top_rake_keywords'][:5]:
                section.append(f"       - {kw}: {count}")
            section.append("")
        
        # Season trends
        section.append("🗓️ KEYWORD TRENDS BY SEASON")
        season_trends = keyword_trends.get('season_trends', {})
        for season, data in season_trends.items():
            section.append(f"   {season}:")
            section.append(f"     • Documents: {data['document_count']}")
            section.append("     • Top keywords:")
            for kw, count in data['top_rake_keywords'][:5]:
                section.append(f"       - {kw}: {count}")
            section.append("")
        
        # Fashion-specific keywords
        section.append("👗 FASHION-SPECIFIC KEYWORD INSIGHTS")
        for category, keywords in fashion_keywords.items():
            if keywords:
                section.append(f"   {category.replace('_', ' ').title()}:")
                for kw_data in keywords[:5]:  # Show top 5
                    section.append(f"     • {kw_data['keyword']}: {kw_data['total_occurrences']} occurrences")
                section.append("")
        
        return "\n".join(section)

    def generate_ner_analysis_section(self, 
                                    entity_trends: Dict,
                                    brand_analysis: Dict,
                                    color_analysis: Dict,
                                    material_analysis: Dict) -> str:
        """
        Generate NER analysis section
        
        Args:
            entity_trends: Entity trend analysis
            brand_analysis: Brand analysis results
            color_analysis: Color trend analysis
            material_analysis: Material analysis results
            
        Returns:
            NER analysis section text
        """
        section = []
        section.append("NAMED ENTITY RECOGNITION")
        section.append("=" * 80)
        section.append("")
        
        # Brand analysis
        section.append("🏷️ BRAND ANALYSIS")
        section.append(f"   • Total brand mentions: {brand_analysis['total_brand_mentions']}")
        section.append(f"   • Unique brands: {brand_analysis['unique_brands']}")
        section.append("   • Top brands:")
        for brand, count in brand_analysis['overall_brand_ranking'][:10]:
            section.append(f"     - {brand}: {count} mentions")
        section.append("")
        
        # Color analysis
        section.append("🎨 COLOR TRENDS")
        section.append(f"   • Total color mentions: {color_analysis['total_color_mentions']}")
        section.append(f"   • Unique colors: {color_analysis['unique_colors']}")
        section.append("   • Top colors:")
        for color, count in color_analysis['overall_color_ranking'][:10]:
            section.append(f"     - {color}: {count} mentions")
        section.append("")
        
        # Material analysis
        section.append("🧵 MATERIAL TRENDS")
        section.append(f"   • Total material mentions: {material_analysis['total_material_mentions']}")
        section.append(f"   • Unique materials: {material_analysis['unique_materials']}")
        section.append("   • Top materials:")
        for material, count in material_analysis['overall_material_ranking'][:10]:
            section.append(f"     - {material}: {count} mentions")
        section.append("")
        
        # Category-wise entity analysis
        section.append("🏷️ ENTITY TRENDS BY CATEGORY")
        category_trends = entity_trends.get('category_trends', {})
        for category, data in category_trends.items():
            section.append(f"   {category}:")
            section.append(f"     • Documents: {data['document_count']}")
            section.append(f"     • Total entities: {data['total_entities']}")
            section.append("     • Top entities:")
            for entity, count in data['top_standard_entities'][:3]:
                section.append(f"       - {entity}: {count}")
            section.append("")

        return "\n".join(section)

    def generate_predictions_section(self, 
                                   predictions_df: pd.DataFrame,
                                   trend_analysis: Dict,
                                   forecasts: Dict,
                                   model_performance: Dict) -> str:
        """
        Generate predictions and forecasting section
        
        Args:
            predictions_df: Predictions DataFrame
            trend_analysis: Trend analysis results
            forecasts: Forecast results
            model_performance: Model performance metrics
            
        Returns:
            Predictions section text
        """
        section = []
        section.append("PREDICTIONS & FORECASTING")
        section.append("=" * 80)
        section.append("")
        
        # Model performance
        section.append("🤖 MODEL PERFORMANCE")
        for task_name, task_perf in model_performance.items():
            section.append(f"   {task_name.replace('_', ' ').title()}:")
            for model_name, perf in task_perf.items():
                section.append(f"     • {model_name}: {perf['accuracy']:.1%} accuracy "
                             f"(CV: {perf['cv_mean']:.1%} ± {perf['cv_std']:.1%})")
            section.append("")
        
        # Overall trend predictions
        section.append("📊 OVERALL TREND PREDICTIONS")
        overall_trends = trend_analysis.get('overall_trends', {})
        for trend_type, distribution in overall_trends.items():
            section.append(f"   {trend_type.replace('_', ' ').title()}:")
            for prediction, count in distribution.items():
                percentage = (count / len(predictions_df)) * 100
                section.append(f"     • {prediction}: {count} documents ({percentage:.1f}%)")
            section.append("")
        
        # Category forecasts
        section.append("🏷️ CATEGORY FORECASTS")
        category_forecasts = forecasts.get('category_forecasts', {})
        for category, forecast_data in category_forecasts.items():
            section.append(f"   {category}:")
            section.append(f"     • Forecast: {forecast_data['forecast']}")
            section.append(f"     • Trend score: {forecast_data['trend_score']:.1%}")
            section.append(f"     • Confidence: {forecast_data['confidence']:.1%}")
            section.append("")
        
        # Market forecast
        section.append("🌍 MARKET FORECAST")
        market_forecast = forecasts.get('market_forecast', {})
        section.append(f"   • Overall sentiment: {market_forecast.get('overall_sentiment', 'Unknown')}")
        section.append(f"   • Innovation level: {market_forecast.get('innovation_level', 'Unknown')}")
        section.append(f"   • Market confidence: {market_forecast.get('market_confidence', 0):.1%}")
        section.append("")
        
        return "\n".join(section)
    
    def generate_recommendations_section(self, 
                                       forecasts: Dict,
                                       trend_analysis: Dict,
                                       brand_analysis: Dict,
                                       color_analysis: Dict) -> str:
        """
        Generate recommendations section
        
        Args:
            forecasts: Forecast results
            trend_analysis: Trend analysis results
            brand_analysis: Brand analysis results
            color_analysis: Color analysis results
            
        Returns:
            Recommendations section text
        """
        section = []
        section.append("RECOMMENDATIONS")
        section.append("=" * 80)
        section.append("")
        
        # Strategic recommendations
        section.append("🎯 STRATEGIC RECOMMENDATIONS")
        
        # Based on market forecast
        market_forecast = forecasts.get('market_forecast', {})
        if market_forecast.get('overall_sentiment') == 'Positive':
            section.append("   • Market shows positive sentiment - consider expanding product lines")
        elif market_forecast.get('overall_sentiment') == 'Negative':
            section.append("   • Market shows negative sentiment - focus on core products and cost optimization")
        
        if market_forecast.get('innovation_level') == 'High':
            section.append("   • High innovation detected - invest in R&D and trend forecasting")
        elif market_forecast.get('innovation_level') == 'Low':
            section.append("   • Low innovation detected - opportunity for market disruption")
        
        section.append("")
        
        # Category-specific recommendations
        section.append("🏷️ CATEGORY-SPECIFIC RECOMMENDATIONS")
        category_forecasts = forecasts.get('category_forecasts', {})
        for category, forecast_data in category_forecasts.items():
            section.append(f"   {category}:")
            if forecast_data['forecast'] == 'Strong Growth':
                section.append("     • High growth potential - increase inventory and marketing")
            elif forecast_data['forecast'] == 'Moderate Growth':
                section.append("     • Steady growth expected - maintain current strategy")
            elif forecast_data['forecast'] == 'Stable':
                section.append("     • Stable market - focus on efficiency and quality")
            else:
                section.append("     • Declining trend - consider repositioning or exit strategy")
            section.append("")
        
        # Color recommendations
        section.append("🎨 COLOR TREND RECOMMENDATIONS")
        top_colors = color_analysis['overall_color_ranking'][:5]
        section.append("   • Top trending colors to focus on:")
        for color, count in top_colors:
            section.append(f"     - {color.title()}: {count} mentions")
        section.append("")
        
        # Brand positioning
        section.append("🏷️ BRAND POSITIONING INSIGHTS")
        top_brands = brand_analysis['overall_brand_ranking'][:5]
        section.append("   • Most mentioned brands (competitive landscape):")
        for brand, count in top_brands:
            section.append(f"     - {brand.title()}: {count} mentions")
        section.append("")
        
        return "\n".join(section)
    
    def generate_full_report(self, 
                           data_stats: Dict,
                           sentiment_summary: Dict,
                           sentiment_by_category: Dict,
                           fashion_sentiment_insights: Dict,
                           bertopic_results: Dict,
                           topic_trends: Dict,
                           keyword_trends: Dict,
                           fashion_keywords: Dict,
                           entity_trends: Dict,
                           brand_analysis: Dict,
                           color_analysis: Dict,
                           material_analysis: Dict,
                           predictions_df: pd.DataFrame,
                           trend_analysis: Dict,
                           forecasts: Dict,
                           model_performance: Dict) -> str:
        """
        Generate complete fashion trend analysis report
        
        Args:
            Various analysis results
            
        Returns:
            Complete report text
        """
        logger.info("Generating complete fashion trend analysis report...")
        
        report = []
        
        # Header
        report.append("FASHION TREND ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        report.append(self.generate_executive_summary(
            data_stats, sentiment_summary, trend_analysis, forecasts, model_performance
        ))
        report.append("")
        
        # Sentiment Analysis
        report.append(self.generate_sentiment_analysis_section(
            sentiment_summary, sentiment_by_category, fashion_sentiment_insights
        ))
        report.append("")
        
        # Topic Modeling
        report.append(self.generate_topic_modeling_section(
            bertopic_results, topic_trends
        ))
        report.append("")
        
        # Keyword Analysis
        report.append(self.generate_keyword_analysis_section(
            keyword_trends, fashion_keywords
        ))
        report.append("")
        
        # NER Analysis
        report.append(self.generate_ner_analysis_section(
            entity_trends, brand_analysis, color_analysis, material_analysis
        ))
        report.append("")
        
        # Predictions & Forecasting
        report.append(self.generate_predictions_section(
            predictions_df, trend_analysis, forecasts, model_performance
        ))
        report.append("")
        
        # Recommendations
        report.append(self.generate_recommendations_section(
            forecasts, trend_analysis, brand_analysis, color_analysis
        ))
        report.append("")
        
        # Footer
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)

        return "\n".join(report)

    def save_report(self, report_text: str, filename: str = "fashion_trend_analysis_report.txt"):
        """
        Save report to file
        
        Args:
            report_text: Complete report text
            filename: Output filename
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Report saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")
    
    def save_results_json(self, results: Dict, filename: str = "fashion_trend_analysis_results.json"):
        """
        Save analysis results as JSON
        
        Args:
            results: Analysis results dictionary
            filename: Output filename
        """
        try:
            # Convert numpy arrays and non-serializable objects to lists/dicts
            serializable_results = self._make_serializable(results)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def _make_serializable(self, obj):
        """
        Convert non-serializable objects to serializable format
        
        Args:
            obj: Object to convert
            
        Returns:
            Serializable object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj

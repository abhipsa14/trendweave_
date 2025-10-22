# fashion_forecaster.py
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class FashionTrendForecaster:
    def __init__(self):
        self.forecast_horizon = 12
        self.accessory_categories = {
            'jewelry': ['necklace', 'bracelet', 'earrings', 'ring', 'brooch'],
            'bags': ['handbag', 'clutch', 'tote', 'backpack', 'crossbody'],
            'footwear': ['shoes', 'boots', 'sandals', 'sneakers', 'heels'],
            'headwear': ['hat', 'cap', 'beanie', 'headband', 'scarf'],
            'belts': ['belt', 'waistchain', 'sash'],
            'eyewear': ['sunglasses', 'glasses', 'eyewear'],
            'tech_accessories': ['smartwatch', 'fitness_tracker', 'phone_case'],
            'seasonal': ['gloves', 'umbrella', 'shawl', 'wrap']
        }
    
    def generate_forecast_report(self, analysis_results, document_count):
        """Generate comprehensive 12-month fashion forecast report including accessories"""
        print("📈 Generating fashion forecast report...")
        
        # Extract insights
        insights = self._extract_forecast_insights(analysis_results)
        
        forecast = {
            'report_metadata': {
                'generation_date': datetime.now().strftime('%Y-%m-%d'),
                'forecast_period': '12 Months',
                'documents_analyzed': document_count,
                'report_id': f"FTR-{datetime.now().strftime('%Y%m%d')}"
            },
            'executive_summary': self._generate_executive_summary(insights),
            'trend_forecasts': self._generate_trend_forecasts(insights),
            'accessories_forecast': self._generate_accessories_forecast(insights),
            'seasonal_outlook': self._generate_seasonal_outlook(),
            'market_recommendations': self._generate_market_recommendations(insights)
        }
        
        return forecast
    
    def _extract_forecast_insights(self, analysis_results):
        """Extract insights from analyses with accessories focus"""
        insights = {
            'sentiment_trends': [],
            'fashion_keywords': [],
            'entities_found': [],
            'accessory_mentions': [],
            'material_preferences': []
        }
        
        for result in analysis_results:
            # Sentiment insights
            if 'sentiment_analysis' in result:
                sentiment = result['sentiment_analysis']
                insights['sentiment_trends'].append({
                    'sentiment': sentiment.get('overall_sentiment', 'NEUTRAL'),
                    'confidence': sentiment.get('confidence', 0)
                })
            
            # Keyword insights
            if 'keyword_extraction' in result:
                keywords = result['keyword_extraction']
                if 'keywords' in keywords:
                    all_keywords = [kw[0] for kw in keywords['keywords']]
                    insights['fashion_keywords'].extend(all_keywords)
                    
                    # Extract accessory-related keywords
                    accessory_keywords = self._identify_accessory_keywords(all_keywords)
                    insights['accessory_mentions'].extend(accessory_keywords)
            
            # Entity insights
            if 'entity_recognition' in result:
                entities = result['entity_recognition']
                if 'entities' in entities:
                    insights['entities_found'].extend(entities['entities'])
        
        return insights
    
    def _identify_accessory_keywords(self, keywords):
        """Identify accessory-related keywords from the keyword list"""
        accessory_mentions = []
        accessory_terms = []
        
        # Flatten all accessory category terms
        for category_terms in self.accessory_categories.values():
            accessory_terms.extend(category_terms)
        
        # Add common accessory materials and styles
        accessory_terms.extend(['gold', 'silver', 'leather', 'chain', 'beaded', 'statement', 'minimalist'])
        
        for keyword in keywords:
            if any(term in keyword.lower() for term in accessory_terms):
                accessory_mentions.append(keyword)
        
        return accessory_mentions
    
    def _generate_executive_summary(self, insights):
        """Generate executive summary with accessories focus"""
        sentiments = [insight['sentiment'] for insight in insights['sentiment_trends']]
        positive_count = sum(1 for s in sentiments if s == 'POSITIVE')
        sentiment_ratio = positive_count / len(sentiments) if sentiments else 0.5
        
        keyword_freq = Counter(insights['fashion_keywords'])
        accessory_freq = Counter(insights['accessory_mentions'])
        
        top_trends = [kw for kw, count in keyword_freq.most_common(5)]
        top_accessories = [acc for acc, count in accessory_freq.most_common(3)]
        
        return {
            'market_outlook': 'Very Positive' if sentiment_ratio > 0.7 else 
                             'Positive' if sentiment_ratio > 0.6 else 
                             'Neutral' if sentiment_ratio > 0.4 else 'Cautious',
            'key_trend_directions': top_trends,
            'accessory_spotlight': top_accessories,
            'consumer_sentiment': 'Enthusiastic' if sentiment_ratio > 0.7 else 
                                 'Optimistic' if sentiment_ratio > 0.6 else 
                                 'Measured',
            'forecast_confidence': 'Strong',
            'accessories_growth_outlook': 'High Growth' if len(insights['accessory_mentions']) > 10 else 'Moderate Growth'
        }
    
    def _generate_trend_forecasts(self, insights):
        """Generate trend forecasts"""
        return {
            'product_trends': [
                "Sustainable and eco-friendly collections",
                "Versatile, multi-functional pieces", 
                "Tech-integrated smart clothing"
            ],
            'color_trends': [
                "Earth tones and natural palettes",
                "Digital-inspired neons",
                "Pastel shades for spring collections"
            ],
            'material_trends': [
                "Bio-based and recycled materials",
                "Technical fabrics with enhanced functionality",
                "Luxury sustainable textiles"
            ],
            'style_trends': [
                "Minimalist aesthetics with bold accents",
                "Gender-fluid and inclusive designs",
                "Heritage styles with modern twists"
            ]
        }
    
    def _generate_accessories_forecast(self, insights):
        """Generate comprehensive accessories forecast"""
        accessory_freq = Counter(insights['accessory_mentions'])
        
        return {
            'key_accessory_categories': self._forecast_accessory_categories(accessory_freq),
            'materials_trends': self._forecast_accessory_materials(),
            'price_point_analysis': self._forecast_price_points(),
            'seasonal_accessories': self._forecast_seasonal_accessories(),
            'emerging_accessory_trends': self._identify_emerging_accessory_trends(accessory_freq),
            'investment_pieces': self._identify_investment_pieces(),
            'category_growth_forecast': self._calculate_category_growth()
        }
    
    def _forecast_accessory_categories(self, accessory_freq):
        """Forecast trends for different accessory categories"""
        return {
            'jewelry': {
                'trend': "Layered necklaces and statement earrings",
                'growth_outlook': "High",
                'key_items': ["Personalized pieces", "Sustainable metals", "Bold statement jewelry"],
                'consumer_demand': "Strong for everyday luxury"
            },
            'bags': {
                'trend': "Micro bags and sustainable totes",
                'growth_outlook': "Moderate-High", 
                'key_items': ["Convertible styles", "Vintage-inspired shapes", "Tech-integrated bags"],
                'consumer_demand': "Versatility and functionality"
            },
            'footwear': {
                'trend': "Comfort-focused fashion",
                'growth_outlook': "High",
                'key_items': ["Platform sneakers", "Sustainable materials", "Multi-season boots"],
                'consumer_demand': "Comfort without compromising style"
            },
            'tech_accessories': {
                'trend': "Fashion-meets-function",
                'growth_outlook': "Very High",
                'key_items': ["Designer phone cases", "Fashion smartwatches", "Tech jewelry"],
                'consumer_demand': "Seamless integration of tech and style"
            }
        }
    
    def _forecast_accessory_materials(self):
        """Forecast material trends for accessories"""
        return {
            'sustainable_materials': [
                "Recycled metals and lab-grown gems",
                "Vegan leather alternatives",
                "Upcycled fabrics and materials"
            ],
            'luxury_materials': [
                "Traceable gold and conflict-free diamonds",
                "Exotic leathers (sustainable sourced)",
                "Artisanal textiles and craftsmanship"
            ],
            'innovative_materials': [
                "3D-printed components",
                "Smart fabrics with LED integration",
                "Bio-degradable composites"
            ]
        }
    
    def _forecast_price_points(self):
        """Forecast price point trends for accessories"""
        return {
            'luxury_segment': {
                'range': "$500+",
                'trend': "Heritage brands and artisanal pieces",
                'growth': "8-12% annually",
                'drivers': ["Investment mentality", "Quality over quantity"]
            },
            'premium_segment': {
                'range': "$150-$500", 
                'trend': "Designer collaborations and limited editions",
                'growth': "10-15% annually",
                'drivers': ["Aspirational shopping", "Social media influence"]
            },
            'mass_market_segment': {
                'range': "Under $150",
                'trend': "Fast-fashion accessories and dupes",
                'growth': "5-8% annually", 
                'drivers': ["Trend adoption speed", "Value consciousness"]
            }
        }
    
    def _forecast_seasonal_accessories(self):
        """Forecast seasonal accessory trends"""
        return {
            'Spring-Summer': {
                'key_accessories': ["Straw bags", "Oversized sunglasses", "Colorful scarves"],
                'materials': ["Lightweight fabrics", "Natural fibers", "Bright colors"],
                'trending_styles': ["Resort wear accessories", "Vacation-ready pieces"]
            },
            'Fall-Winter': {
                'key_accessories': ["Leather gloves", "Wool hats", "Statement boots"],
                'materials': ["Heavy fabrics", "Metallic accents", "Rich textures"],
                'trending_styles': ["Layered jewelry", "Seasonal color palettes"]
            },
            'Year-Round_Essentials': {
                'key_accessories': ["Everyday jewelry", "Neutral handbags", "Classic watches"],
                'materials': ["Timeless materials", "Durable construction"],
                'trending_styles': ["Minimalist pieces", "Versatile designs"]
            }
        }
    
    def _identify_emerging_accessory_trends(self, accessory_freq):
        """Identify emerging accessory trends"""
        return [
            "Modular accessories that can be worn multiple ways",
            "Personalized and custom-designed pieces",
            "Accessories with hidden tech functionality",
            "Gender-neutral accessory collections",
            "Upcycled and vintage-inspired recreations"
        ]
    
    def _identify_investment_pieces(self):
        """Identify accessory categories with strong investment potential"""
        return {
            'high_investment_potential': [
                "Signature handbags from heritage brands",
                "Fine jewelry with precious materials",
                "Limited edition collaborations"
            ],
            'moderate_investment_potential': [
                "Quality leather goods",
                "Designer sunglasses",
                "Classic timepieces"
            ],
            'trend_driven_investment': [
                "Statement pieces from emerging designers",
                "Tech-integrated accessories",
                "Sustainable innovation pieces"
            ]
        }
    
    def _calculate_category_growth(self):
        """Calculate projected growth for accessory categories"""
        return {
            'projected_growth_rates': {
                'tech_accessories': "15-20%",
                'sustainable_accessories': "12-18%", 
                'fine_jewelry': "8-12%",
                'handbags': "6-10%",
                'footwear': "7-11%",
                'seasonal_accessories': "5-8%"
            },
            'market_drivers': [
                "Rising demand for personalization",
                "Sustainability consciousness",
                "Social media influence",
                "Hybrid work lifestyle changes"
            ]
        }
    
    def _generate_seasonal_outlook(self):
        """Generate seasonal outlook"""
        return {
            'Spring-Summer': {
                'key_themes': ["Lightweight layers", "Vacation dressing", "Sun protection"],
                'color_story': "Aquatic blues, sunny yellows, and crisp whites",
                'fabric_focus': "Breathable linens and technical cottons",
                'accessory_focus': "Straw hats, sunglasses, and beach bags"
            },
            'Fall-Winter': {
                'key_themes': ["Layered luxury", "Winter wellness", "Holiday entertaining"],
                'color_story': "Jewel tones, deep neutrals, and festive metallics", 
                'fabric_focus': "Cozy wools and rich velvets",
                'accessory_focus': "Statement coats, boots, and festive jewelry"
            }
        }
    
    def _generate_market_recommendations(self, insights):
        """Generate market recommendations including accessories strategy"""
        accessory_freq = Counter(insights['accessory_mentions'])
        has_strong_accessory_trends = len(accessory_freq) > 5
        
        base_recommendations = {
            'product_development': [
                "Focus on sustainable material innovation",
                "Develop size-inclusive collections",
                "Create versatile, season-spanning pieces"
            ],
            'marketing_strategy': [
                "Emphasize sustainability and ethical production",
                "Leverage digital storytelling and social media",
                "Create educational content around garment care"
            ],
            'retail_innovation': [
                "Implement virtual try-on technology",
                "Develop rental and resale programs",
                "Create personalized shopping experiences"
            ]
        }
        
        # Add accessory-specific recommendations if trends are strong
        if has_strong_accessory_trends:
            base_recommendations['accessory_strategy'] = [
                "Expand accessory capsule collections",
                "Develop accessory subscription services",
                "Create accessory styling services",
                "Focus on accessory gifting campaigns"
            ]
        
        return base_recommendations
    
    def generate_accessory_specific_report(self, analysis_results):
        """Generate a dedicated accessories forecast report"""
        insights = self._extract_forecast_insights(analysis_results)
        
        accessory_report = {
            'report_type': 'Accessories Forecast',
            'generation_date': datetime.now().strftime('%Y-%m-%d'),
            'executive_summary': self._generate_accessory_executive_summary(insights),
            'category_forecasts': self._generate_accessories_forecast(insights),
            'buying_guide': self._generate_accessory_buying_guide(),
            'risk_assessment': self._generate_accessory_risk_assessment()
        }
        
        return accessory_report
    
    def _generate_accessory_executive_summary(self, insights):
        """Generate executive summary focused on accessories"""
        accessory_freq = Counter(insights['accessory_mentions'])
        top_accessories = [acc for acc, count in accessory_freq.most_common(5)]
        
        return {
            'market_overview': "Accessories market showing strong growth potential",
            'top_trending_categories': top_accessories,
            'growth_drivers': ["Personalization", "Sustainability", "Tech Integration"],
            'forecast_period': "Next 12 months",
            'overall_outlook': "Highly Positive"
        }
    
    def _generate_accessory_buying_guide(self):
        """Generate accessory buying guide for retailers"""
        return {
            'must_have_categories': [
                "Sustainable jewelry collections",
                "Tech-integrated accessories", 
                "Seasonal color capsules",
                "Gender-neutral pieces"
            ],
            'inventory_recommendations': {
                'core_inventory': ["Classic leather goods", "Everyday jewelry", "Essential belts"],
                'trend_inventory': ["Statement pieces", "Collaboration items", "Limited editions"],
                'test_inventory': ["Emerging categories", "Innovative materials", "New designers"]
            },
            'pricing_strategy': [
                "Tiered pricing for different customer segments",
                "Bundle accessories with main collections",
                "Seasonal promotion planning"
            ]
        }
    
    def _generate_accessory_risk_assessment(self):
        """Generate risk assessment for accessories market"""
        return {
            'market_risks': [
                "Over-saturation in fast-fashion accessories",
                "Supply chain disruptions for specialty materials",
                "Changing consumer sentiment on sustainability"
            ],
            'mitigation_strategies': [
                "Diversify supplier base and material sources",
                "Invest in authentic sustainability credentials",
                "Develop agile production capabilities"
            ],
            'opportunities': [
                "Growing demand for personalized accessories",
                "Expansion into mens accessories market",
                "Digital and virtual accessory trends"
            ]
        }

# Example usage
if __name__ == "__main__":
    # Initialize forecaster
    forecaster = FashionTrendForecaster()
    
    # Sample analysis results (mocking your existing data structure)
    sample_analysis = [
        {
            'sentiment_analysis': {
                'overall_sentiment': 'POSITIVE',
                'confidence': 0.85
            },
            'keyword_extraction': {
                'keywords': [
                    ['sustainable jewelry', 0.9],
                    ['tech accessories', 0.8],
                    ['statement handbags', 0.7],
                    ['designer sunglasses', 0.6]
                ]
            },
            'entity_recognition': {
                'entities': ['luxury brands', 'sustainable fashion', 'accessory trends']
            }
        }
    ]
    
    # Generate main forecast report
    forecast_report = forecaster.generate_forecast_report(sample_analysis, document_count=50)
    
    # Generate dedicated accessories report
    accessory_report = forecaster.generate_accessory_specific_report(sample_analysis)
    
    print("🎯 Main Forecast Report Generated")
    print(f"📊 Executive Summary: {forecast_report['executive_summary']}")
    print(f"👜 Accessories Forecast: {len(forecast_report['accessories_forecast']['key_accessory_categories'])} categories")
    
    print("\n👑 Dedicated Accessories Report")
    print(f"📈 Market Overview: {accessory_report['executive_summary']['market_overview']}")
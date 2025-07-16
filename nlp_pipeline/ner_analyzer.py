"""
Named Entity Recognition Module for Fashion Trend Analysis
Provides NER using spaCy and custom fashion entity recognition
"""

import pandas as pd
import numpy as np
import spacy
from spacy import displacy
from collections import Counter, defaultdict
import re
from typing import List, Dict, Tuple, Set
import logging

logger = logging.getLogger(__name__)

class FashionNERAnalyzer:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found. Please install it using: python -m spacy download en_core_web_sm")
            raise
        
        # Fashion-specific entities
        self.fashion_brands = {
            'gucci', 'prada', 'chanel', 'dior', 'versace', 'armani', 'dolce', 'gabbana',
            'yves saint laurent', 'saint laurent', 'balenciaga', 'givenchy', 'fendi',
            'bottega veneta', 'valentino', 'moschino', 'hermès', 'louis vuitton',
            'burberry', 'celine', 'loewe', 'jil sander', 'marc jacobs', 'tom ford',
            'alexander mcqueen', 'stella mccartney', 'vivienne westwood', 'issey miyake',
            'comme des garçons', 'yohji yamamoto', 'kenzo', 'balmain', 'off-white',
            'vetements', 'jacquemus', 'acne studios', 'ganni', 'staud', 'khaite',
            'zara', 'h&m', 'uniqlo', 'cos', 'mango', 'massimo dutti', 'asos',
            'nike', 'adidas', 'converse', 'vans', 'supreme', 'stone island'
        }
        
        self.fashion_colors = {
            'black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple',
            'brown', 'grey', 'gray', 'beige', 'navy', 'burgundy', 'maroon',
            'turquoise', 'teal', 'olive', 'khaki', 'coral', 'salmon', 'ivory',
            'cream', 'gold', 'silver', 'bronze', 'rose gold', 'champagne',
            'emerald', 'sapphire', 'ruby', 'amber', 'camel', 'tan', 'taupe'
        }
        
        self.fashion_materials = {
            'cotton', 'silk', 'wool', 'cashmere', 'linen', 'denim', 'leather',
            'suede', 'velvet', 'satin', 'chiffon', 'lace', 'mesh', 'tulle',
            'organza', 'taffeta', 'polyester', 'nylon', 'spandex', 'elastane',
            'viscose', 'rayon', 'modal', 'bamboo', 'hemp', 'alpaca', 'mohair',
            'tweed', 'flannel', 'corduroy', 'jersey', 'knit', 'woven',
            'synthetic', 'organic', 'recycled', 'sustainable'
        }
        
        self.fashion_styles = {
            'casual', 'formal', 'business', 'smart casual', 'evening', 'cocktail',
            'sporty', 'athletic', 'streetwear', 'vintage', 'retro', 'bohemian',
            'minimalist', 'maximalist', 'grunge', 'preppy', 'punk', 'goth',
            'romantic', 'feminine', 'masculine', 'androgynous', 'avant-garde',
            'contemporary', 'modern', 'classic', 'timeless', 'trendy', 'edgy'
        }
        
        self.fashion_garments = {
            'dress', 'shirt', 'blouse', 'top', 'sweater', 'cardigan', 'jacket',
            'coat', 'blazer', 'suit', 'pants', 'trousers', 'jeans', 'shorts',
            'skirt', 'jumpsuit', 'romper', 'bodysuit', 'lingerie', 'underwear',
            'bra', 'panties', 'socks', 'tights', 'stockings', 'scarf', 'hat',
            'cap', 'beanie', 'gloves', 'mittens', 'belt', 'bag', 'purse',
            'handbag', 'backpack', 'clutch', 'wallet', 'shoes', 'boots',
            'sneakers', 'sandals', 'heels', 'flats', 'loafers', 'oxfords'
        }
        
        # Combine all fashion entities
        self.fashion_entities = {
            'BRAND': self.fashion_brands,
            'COLOR': self.fashion_colors,
            'MATERIAL': self.fashion_materials,
            'STYLE': self.fashion_styles,
            'GARMENT': self.fashion_garments
        }
    
    def extract_standard_entities(self, text: str) -> List[Dict]:
        """
        Extract standard named entities using spaCy
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries
        """
        try:
            doc = self.nlp(text)
            entities = []
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'description': spacy.explain(ent.label_)
                })
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in standard entity extraction: {e}")
            return []
    
    def extract_fashion_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract fashion-specific entities
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with fashion entity types and their occurrences
        """
        text_lower = text.lower()
        fashion_entities_found = defaultdict(list)
        
        for entity_type, entity_set in self.fashion_entities.items():
            for entity in entity_set:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(entity) + r'\b'
                matches = re.findall(pattern, text_lower)
                
                if matches:
                    fashion_entities_found[entity_type].extend(matches)
        
        # Remove duplicates while preserving order
        for entity_type in fashion_entities_found:
            fashion_entities_found[entity_type] = list(dict.fromkeys(fashion_entities_found[entity_type]))
        
        return dict(fashion_entities_found)
    
    def extract_fashion_locations(self, text: str) -> List[str]:
        """
        Extract fashion-related locations (fashion capitals, etc.)
        
        Args:
            text: Input text
            
        Returns:
            List of fashion locations found
        """
        fashion_locations = {
            'paris', 'milan', 'london', 'new york', 'tokyo', 'berlin',
            'copenhagen', 'stockholm', 'antwerp', 'florence', 'rome',
            'los angeles', 'miami', 'amsterdam', 'barcelona', 'madrid',
            'fashion week', 'runway', 'catwalk', 'showroom', 'boutique',
            'atelier', 'flagship store', 'concept store'
        }
        
        text_lower = text.lower()
        locations_found = []
        
        for location in fashion_locations:
            pattern = r'\b' + re.escape(location) + r'\b'
            if re.search(pattern, text_lower):
                locations_found.append(location)
        
        return list(set(locations_found))
    
    def analyze_batch_entities(self, texts: List[str], filenames: List[str] = None) -> pd.DataFrame:
        """
        Analyze entities for a batch of texts
        
        Args:
            texts: List of text documents
            filenames: List of filenames (optional)
            
        Returns:
            DataFrame with entity analysis results
        """
        results = []
        
        for i, text in enumerate(texts):
            logger.info(f"Analyzing entities for document {i+1}/{len(texts)}")
            
            # Standard entities
            standard_entities = self.extract_standard_entities(text)
            
            # Fashion entities
            fashion_entities = self.extract_fashion_entities(text)
            
            # Fashion locations
            fashion_locations = self.extract_fashion_locations(text)
            
            # Count entities by type
            entity_counts = Counter([ent['label'] for ent in standard_entities])
            
            result = {
                'document_id': i,
                'filename': filenames[i] if filenames else f"doc_{i}",
                'standard_entities': standard_entities,
                'fashion_entities': fashion_entities,
                'fashion_locations': fashion_locations,
                'entity_counts': dict(entity_counts),
                'total_entities': len(standard_entities),
                'unique_entity_types': len(entity_counts)
            }
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def analyze_entity_trends(self, entity_df: pd.DataFrame, text_df: pd.DataFrame) -> Dict:
        """
        Analyze entity trends across categories and seasons
        
        Args:
            entity_df: DataFrame with entity analysis results
            text_df: DataFrame with document metadata
            
        Returns:
            Dictionary with entity trend analysis
        """
        # Combine entity and text data
        combined_df = pd.concat([entity_df, text_df], axis=1)
        
        # Analyze trends by category
        category_trends = {}
        for category in combined_df['category'].unique():
            category_data = combined_df[combined_df['category'] == category]
            
            # Collect all entities for this category
            all_standard_entities = []
            all_fashion_entities = defaultdict(list)
            all_fashion_locations = []
            
            for _, row in category_data.iterrows():
                # Standard entities
                all_standard_entities.extend([ent['text'] for ent in row['standard_entities']])
                
                # Fashion entities
                for entity_type, entities in row['fashion_entities'].items():
                    all_fashion_entities[entity_type].extend(entities)
                
                # Fashion locations
                all_fashion_locations.extend(row['fashion_locations'])
            
            # Count frequencies
            standard_counter = Counter(all_standard_entities)
            fashion_location_counter = Counter(all_fashion_locations)
            
            category_trends[category] = {
                'document_count': len(category_data),
                'top_standard_entities': standard_counter.most_common(10),
                'top_fashion_locations': fashion_location_counter.most_common(10),
                'fashion_entity_counts': {
                    entity_type: len(entities) for entity_type, entities in all_fashion_entities.items()
                },
                'total_entities': len(all_standard_entities)
            }
        
        # Analyze trends by season
        season_trends = {}
        for season in combined_df['season'].unique():
            season_data = combined_df[combined_df['season'] == season]
            
            # Collect all entities for this season
            all_standard_entities = []
            all_fashion_entities = defaultdict(list)
            all_fashion_locations = []
            
            for _, row in season_data.iterrows():
                # Standard entities
                all_standard_entities.extend([ent['text'] for ent in row['standard_entities']])
                
                # Fashion entities
                for entity_type, entities in row['fashion_entities'].items():
                    all_fashion_entities[entity_type].extend(entities)
                
                # Fashion locations
                all_fashion_locations.extend(row['fashion_locations'])
            
            # Count frequencies
            standard_counter = Counter(all_standard_entities)
            fashion_location_counter = Counter(all_fashion_locations)
            
            season_trends[season] = {
                'document_count': len(season_data),
                'top_standard_entities': standard_counter.most_common(10),
                'top_fashion_locations': fashion_location_counter.most_common(10),
                'fashion_entity_counts': {
                    entity_type: len(entities) for entity_type, entities in all_fashion_entities.items()
                },
                'total_entities': len(all_standard_entities)
            }
        
        return {
            'category_trends': category_trends,
            'season_trends': season_trends
        }
    
    def get_brand_analysis(self, entity_df: pd.DataFrame, text_df: pd.DataFrame) -> Dict:
        """
        Analyze brand mentions across documents
        
        Args:
            entity_df: DataFrame with entity analysis results
            text_df: DataFrame with document metadata
            
        Returns:
            Dictionary with brand analysis
        """
        # Combine entity and text data
        combined_df = pd.concat([entity_df, text_df], axis=1)
        
        # Collect all brand mentions
        all_brands = []
        brand_by_category = defaultdict(list)
        brand_by_season = defaultdict(list)
        
        for _, row in combined_df.iterrows():
            brands = row['fashion_entities'].get('BRAND', [])
            all_brands.extend(brands)
            
            # Group by category
            brand_by_category[row['category']].extend(brands)
            
            # Group by season
            brand_by_season[row['season']].extend(brands)
        
        # Count brand frequencies
        brand_counter = Counter(all_brands)
        
        # Category analysis
        category_brand_analysis = {}
        for category, brands in brand_by_category.items():
            category_brand_analysis[category] = Counter(brands).most_common(10)
        
        # Season analysis
        season_brand_analysis = {}
        for season, brands in brand_by_season.items():
            season_brand_analysis[season] = Counter(brands).most_common(10)
        
        return {
            'overall_brand_ranking': brand_counter.most_common(20),
            'total_brand_mentions': len(all_brands),
            'unique_brands': len(brand_counter),
            'category_brand_analysis': category_brand_analysis,
            'season_brand_analysis': season_brand_analysis
        }
    
    def get_color_trend_analysis(self, entity_df: pd.DataFrame, text_df: pd.DataFrame) -> Dict:
        """
        Analyze color trends across documents
        
        Args:
            entity_df: DataFrame with entity analysis results
            text_df: DataFrame with document metadata
            
        Returns:
            Dictionary with color trend analysis
        """
        # Combine entity and text data
        combined_df = pd.concat([entity_df, text_df], axis=1)
        
        # Collect all color mentions
        all_colors = []
        color_by_category = defaultdict(list)
        color_by_season = defaultdict(list)
        
        for _, row in combined_df.iterrows():
            colors = row['fashion_entities'].get('COLOR', [])
            all_colors.extend(colors)
            
            # Group by category
            color_by_category[row['category']].extend(colors)
            
            # Group by season
            color_by_season[row['season']].extend(colors)
        
        # Count color frequencies
        color_counter = Counter(all_colors)
        
        # Category analysis
        category_color_analysis = {}
        for category, colors in color_by_category.items():
            category_color_analysis[category] = Counter(colors).most_common(10)
        
        # Season analysis
        season_color_analysis = {}
        for season, colors in color_by_season.items():
            season_color_analysis[season] = Counter(colors).most_common(10)
        
        return {
            'overall_color_ranking': color_counter.most_common(20),
            'total_color_mentions': len(all_colors),
            'unique_colors': len(color_counter),
            'category_color_analysis': category_color_analysis,
            'season_color_analysis': season_color_analysis
        }
    
    def get_material_analysis(self, entity_df: pd.DataFrame, text_df: pd.DataFrame) -> Dict:
        """
        Analyze material trends across documents
        
        Args:
            entity_df: DataFrame with entity analysis results
            text_df: DataFrame with document metadata
            
        Returns:
            Dictionary with material analysis
        """
        # Combine entity and text data
        combined_df = pd.concat([entity_df, text_df], axis=1)
        
        # Collect all material mentions
        all_materials = []
        material_by_category = defaultdict(list)
        material_by_season = defaultdict(list)
        
        for _, row in combined_df.iterrows():
            materials = row['fashion_entities'].get('MATERIAL', [])
            all_materials.extend(materials)
            
            # Group by category
            material_by_category[row['category']].extend(materials)
            
            # Group by season
            material_by_season[row['season']].extend(materials)
        
        # Count material frequencies
        material_counter = Counter(all_materials)
        
        # Category analysis
        category_material_analysis = {}
        for category, materials in material_by_category.items():
            category_material_analysis[category] = Counter(materials).most_common(10)
        
        # Season analysis
        season_material_analysis = {}
        for season, materials in material_by_season.items():
            season_material_analysis[season] = Counter(materials).most_common(10)
        
        return {
            'overall_material_ranking': material_counter.most_common(20),
            'total_material_mentions': len(all_materials),
            'unique_materials': len(material_counter),
            'category_material_analysis': category_material_analysis,
            'season_material_analysis': season_material_analysis
        }

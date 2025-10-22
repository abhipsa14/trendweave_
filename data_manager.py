# data_manager.py
import pandas as pd
import json
import os
from datetime import datetime

class FashionDataManager:
    def __init__(self, data_dir="./training_data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
    def collect_training_data_from_analysis(self, analysis_results):
        """Collect training data from analysis results"""
        training_data = []
        
        for result in analysis_results:
            # Extract sentiment training data
            if 'sentiment_analysis' in result and 'error' not in result['sentiment_analysis']:
                sentiment_data = {
                    'text': result.get('text_preview', '')[:500],
                    'sentiment': result['sentiment_analysis']['overall_sentiment'],
                    'confidence': result['sentiment_analysis']['confidence'],
                    'source_file': result.get('filename', 'unknown'),
                    'timestamp': datetime.now().isoformat(),
                    'type': 'sentiment'
                }
                training_data.append(sentiment_data)
            
            # Extract entity training data
            if 'entity_recognition' in result:
                for entity in result['entity_recognition'].get('entities', []):
                    if entity.get('is_fashion_related', False):
                        entity_data = {
                            'text': entity['text'],
                            'category': entity['category'],
                            'score': entity['score'],
                            'source_file': result.get('filename', 'unknown'),
                            'timestamp': datetime.now().isoformat(),
                            'type': 'entity'
                        }
                        training_data.append(entity_data)
        
        return training_data
    
    def save_training_data(self, training_data, dataset_name="fashion_training_data"):
        """Save collected training data"""
        if not training_data:
            return None
            
        filename = f"{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Training data saved to: {filepath}")
        return filepath
    
    def load_training_data(self):
        """Load all training data from files"""
        training_data = []
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json') and 'fashion_training_data' in filename:
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            training_data.extend(data)
                except Exception as e:
                    print(f"Warning: Could not load {filepath}: {e}")
        
        return training_data
    
    def get_training_stats(self):
        """Get statistics about collected training data"""
        all_data = self.load_training_data()
        
        if not all_data:
            return {
                'total_samples': 0,
                'sentiment_samples': 0,
                'entity_samples': 0,
                'files_count': 0
            }
        
        sentiment_data = [item for item in all_data if item.get('type') == 'sentiment']
        entity_data = [item for item in all_data if item.get('type') == 'entity']
        
        # Count unique source files
        source_files = set(item.get('source_file', '') for item in all_data)
        
        return {
            'total_samples': len(all_data),
            'sentiment_samples': len(sentiment_data),
            'entity_samples': len(entity_data),
            'files_count': len(source_files)
        }
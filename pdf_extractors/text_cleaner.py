# pdf_extractors/text_cleaner.py
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FashionTextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        # Keep fashion-related words that might normally be stopwords
        self.fashion_keep_words = {'wear', 'new', 'style', 'design', 'color', 'fashion', 'trend'}
        self.stop_words = self.stop_words - self.fashion_keep_words
        
    def clean_fashion_text(self, text):
        """Clean and preprocess fashion text while preserving fashion terminology"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep fashion-related symbols
        text = re.sub(r'[^\w\s#@&/-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_non_fashion_content(self, text):
        """Remove content that doesn't contain fashion-related terms"""
        sentences = text.split('.')
        fashion_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in [
                'fashion', 'style', 'trend', 'collection', 'design', 'wear',
                'dress', 'shirt', 'pants', 'jacket', 'shoes', 'accessories'
            ]):
                fashion_sentences.append(sentence)
        
        return '. '.join(fashion_sentences)
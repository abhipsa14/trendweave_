# pdf_extractors/pdf_parser.py
import PyPDF2
import pdfplumber
import re
from datetime import datetime
import os

class PDFProcessor:
    def __init__(self):
        self.fashion_keywords = [
            'fashion', 'trend', 'collection', 'design', 'style', 'wear', 'couture',
            'dress', 'shirt', 'pants', 'jeans', 'jacket', 'coat', 'skirt', 'blouse',
            'shoes', 'sneakers', 'boots', 'heels', 'accessories', 'jewelry', 'bag',
            'color', 'fabric', 'material', 'textile', 'print', 'pattern', 'silhouette',
            'season', 'spring', 'summer', 'fall', 'autumn', 'winter', 'resort', 'cruise',
            'luxury', 'premium', 'sustainable', 'organic', 'eco-friendly', 'vintage',
            'minimalist', 'bohemian', 'streetwear', 'avant-garde', 'haute-couture',
            'runway', 'catwalk', 'collection', 'lookbook', 'capsule', 'bespoke'
        ]
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF with fashion-specific processing"""
        text = ""
        
        try:
            # Try pdfplumber first for better text extraction
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            # Fallback to PyPDF2
            if not text.strip():
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                        
        except Exception as e:
            raise Exception(f"PDF extraction failed: {str(e)}")
        
        return self.clean_text(text)
    
    def clean_text(self, text):
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()-]', '', text)
        
        return text.strip()
    
    def is_fashion_related(self, text):
        """Check if text is fashion-related using fashion jargon"""
        text_lower = text.lower()
        fashion_score = sum(1 for keyword in self.fashion_keywords if keyword in text_lower)
        return fashion_score >= 3
    
    def extract_metadata(self, file_path):
        """Extract basic metadata from PDF"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata = pdf_reader.metadata
                
                return {
                    'title': getattr(metadata, 'title', 'Unknown'),
                    'author': getattr(metadata, 'author', 'Unknown'),
                    'pages': len(pdf_reader.pages),
                    'extraction_date': datetime.now().strftime("%Y-%m-%d")
                }
        except:
            return {
                'title': 'Unknown',
                'author': 'Unknown', 
                'pages': 0,
                'extraction_date': datetime.now().strftime("%Y-%m-%d")
            }
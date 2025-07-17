#!/usr/bin/env python3
"""
Setup script for Fashion Trend Analysis Project
Installs required packages and downloads necessary models
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Required packages installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error installing packages. Please check requirements.txt")
        return False
    return True

def download_spacy_model():
    """Download spaCy English model"""
    print("Downloading spaCy English model...")
    try:
        subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
        print("✅ spaCy model downloaded successfully!")
    except subprocess.CalledProcessError:
        print("❌ Error downloading spaCy model. Please run manually: python -m spacy download en_core_web_sm")
        return False
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('vader_lexicon', quiet=True)
        print("✅ NLTK data downloaded successfully!")
    except Exception as e:
        print(f"❌ Error downloading NLTK data: {e}")
        return False
    return True

def check_directories():
    """Check if required directories exist"""
    print("Checking directories...")
    
    required_dirs = ['extracted_content', 'nlp_pipeline']
    
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"❌ Directory '{dir_name}' not found!")
            return False
        else:
            print(f"✅ Directory '{dir_name}' found")
    
    return True

def main():
    """Main setup function"""
    print("Fashion Trend Analysis Project Setup")
    print("=" * 50)
    
    # Check directories
    if not check_directories():
        print("❌ Setup failed: Required directories not found")
        return
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed: Could not install requirements")
        return
    
    # Download spaCy model
    if not download_spacy_model():
        print("❌ Setup failed: Could not download spaCy model")
        return
    
    # Download NLTK data
    if not download_nltk_data():
        print("❌ Setup failed: Could not download NLTK data")
        return
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("You can now run the analysis with: python fashion_trend_analysis.py")
    print("=" * 50)

if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
import re
import string
import os # Import os module for directory operations
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Core NLP libraries
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

# Sentiment Analysis
from textblob import TextBlob
from transformers import pipeline

# Topic Modeling
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# Keyword Extraction
from rake_nltk import Rake
from keybert import KeyBERT

# Visualization (not directly used in pipeline logic, but good to keep for future)
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger') # Ensure this is also checked
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    print("NLTK data download complete.")


class NLPPipeline:
    def __init__(self, language='en'):
        self.language = language
        self.results = {}

        # Load spaCy model
        try:
            self.nlp = spacy.load("en_core_web_sm")
            # Increase spaCy's max_length to handle longer texts
            # Default is 1,000,000 characters. Setting to 2,000,000 for robustness.
            self.nlp.max_length = 2000000 # Increased limit
            print("spaCy model 'en_core_web_sm' loaded and max_length set to 2,000,000.")
        except OSError:
            print("Please install spaCy English model: python -m spacy download en_core_web_sm")
            raise

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        # Sentiment analyzer: BERT model (1–5 stars)
        # Suppress potential warnings from Hugging Face transformers pipeline loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
        print("Sentiment analyzer loaded.")

        self.keybert_model = KeyBERT()
        self.rake = Rake()
        print("Keyword extraction models initialized.")

        print("NLP Pipeline initialized successfully!")

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses a single string of text by:
        - Removing HTML tags
        - Removing non-alphabetic characters
        - Converting to lowercase
        - Removing extra whitespace
        - Tokenizing, lemmatizing, and removing stop words and short tokens.
        """
        text = re.sub(r'<[^>]+>', '', text) # Remove HTML tags
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove non-alphabetic characters
        text = text.lower() # Convert to lowercase
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace and strip

        tokens = word_tokenize(text)
        # Lemmatize and remove stop words and tokens shorter than 3 characters
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts named entities (PERSON, ORG, GPE, etc.) from text using spaCy.
        Returns a dictionary where keys are entity types and values are lists of unique entities.
        """
        doc = self.nlp(text)
        entities = {}
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        # Ensure uniqueness of entities within each category
        for key in entities:
            entities[key] = list(set(entities[key]))
        return entities

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Performs sentiment analysis using TextBlob for polarity/subjectivity
        and a BERT-based transformer model for star ratings and labels.
        """
        blob = TextBlob(text)
        textblob_sentiment = {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
            'label': 'positive' if blob.sentiment.polarity > 0 else 'negative' if blob.sentiment.polarity < 0 else 'neutral'
        }

        sentences = sent_tokenize(text)
        transformer_sentiments = []
        # Analyze up to the first 5 meaningful sentences to avoid excessive computation
        for sentence in sentences[:5]:
            if len(sentence.split()) > 5: # Only analyze sentences with more than 5 words
                try:
                    result = self.sentiment_analyzer(sentence)
                    # Extract star rating from label (e.g., "1 star")
                    stars = int(re.search(r'\d', result[0]['label']).group())
                    transformer_sentiments.append({
                        'sentence': sentence,
                        'stars': stars,
                        'label': result[0]['label'],
                        'score': result[0]['score']
                    })
                except Exception as e:
                    # print(f"Could not analyze sentence '{sentence[:50]}...': {e}")
                    continue # Skip sentences that cause errors

        if transformer_sentiments:
            star_scores = [s['stars'] for s in transformer_sentiments]
            avg_star = np.mean(star_scores)
            overall_label = (
                'very positive' if avg_star >= 4.5 else
                'positive' if avg_star >= 3.5 else
                'neutral' if avg_star >= 2.5 else
                'negative' if avg_star >= 1.5 else
                'very negative'
            )
        else:
            avg_star = 3.0 # Default to neutral if no sentences were analyzed
            overall_label = 'neutral'

        return {
            'textblob': textblob_sentiment,
            'transformer': {
                'overall_label': overall_label,
                'average_star_rating': avg_star,
                'sentence_sentiments': transformer_sentiments
            }
        }

    def extract_topics_bertopic(self, texts: List[str], num_topics: int = 5) -> Dict[str, Any]:
        """
        Extracts topics using BERTopic, which leverages sentence embeddings and UMAP.
        Handles cases with empty processed texts or embeddings gracefully.
        """
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Filter out empty strings after preprocessing
        processed_texts = [text for text in processed_texts if text]

        if not processed_texts:
            print("Warning: No valid texts after preprocessing for BERTopic. Skipping topic extraction.")
            return {
                'topics': [],
                'model': None,
                'topics_raw': pd.DataFrame() # Return an empty DataFrame for consistency
            }

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        topic_model = BERTopic(top_n_words=10, nr_topics=num_topics, calculate_probabilities=True, verbose=False)
        
        try:
            embeddings = embedding_model.encode(processed_texts)
            # Check if embeddings are empty (e.g., if processed_texts was very short/filtered)
            if embeddings.size == 0:
                print("Warning: SentenceTransformer produced empty embeddings. Skipping BERTopic topic extraction.")
                return {
                    'topics': [],
                    'model': None,
                    'topics_raw': pd.DataFrame()
                }

            topics, probs = topic_model.fit_transform(processed_texts, embeddings=embeddings)
            topic_info = topic_model.get_topic_info()
            extracted_topics = []
            for i in range(len(topic_info)):
                if topic_info.iloc[i]["Topic"] != -1: # -1 is the outlier topic, typically ignored
                    extracted_topics.append({
                        "topic_id": int(topic_info.iloc[i]["Topic"]),
                        "top_words": topic_model.get_topic(int(topic_info.iloc[i]["Topic"]))
                    })
            return {
                'topics': extracted_topics,
                'model': topic_model,
                'topics_raw': topic_info
            }
        except Exception as e:
            print(f"Error during BERTopic topic extraction: {e}")
            return {
                'topics': [],
                'model': None,
                'topics_raw': pd.DataFrame()
            }

    def extract_topics_nmf(self, texts: List[str], num_topics: int = 5) -> Dict[str, Any]:
        """
        Extracts topics using Non-negative Matrix Factorization (NMF) on TF-IDF features.
        Adjusts min_df and max_df for small datasets and handles cases with no features.
        """
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Filter out empty strings after preprocessing
        processed_texts = [text for text in processed_texts if text]

        if not processed_texts:
            print("Warning: No valid texts after preprocessing for NMF. Skipping topic extraction.")
            return {
                'topics': [],
                'model': None,
                'vectorizer': None,
                'feature_names': []
            }

        # Adjust min_df and max_df dynamically for robustness with small datasets
        current_min_df = 1 # Always allow terms appearing in at least one document
        # If there's only one document, max_df should not filter it out (i.e., be 1.0)
        # Otherwise, use 0.8 as a default to filter very common words.
        current_max_df = 1.0 if len(processed_texts) == 1 else 0.8 

        vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=current_min_df,
            max_df=current_max_df,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_texts)

            # Check if the TF-IDF matrix has any features (columns)
            if tfidf_matrix.shape[1] == 0:
                print("Warning: TF-IDF matrix has no features after vectorization. Skipping NMF.")
                return {
                    'topics': [],
                    'model': None,
                    'vectorizer': vectorizer,
                    'feature_names': []
                }
            
            feature_names = vectorizer.get_feature_names_out()
            
            # Ensure num_topics is not greater than the number of features or documents
            # NMF requires n_components < min(n_samples, n_features)
            effective_num_topics = min(num_topics, tfidf_matrix.shape[1] -1, len(processed_texts) - 1)
            # Ensure effective_num_topics is at least 1 if possible
            if effective_num_topics <= 0 and (tfidf_matrix.shape[1] > 0 and len(processed_texts) > 0):
                effective_num_topics = 1
            elif effective_num_topics <= 0: # If still <= 0, means no valid topics can be extracted
                print(f"Warning: Not enough features or documents for NMF. Cannot extract {num_topics} topics. Effective topics: {effective_num_topics}")
                return {
                    'topics': [],
                    'model': None,
                    'vectorizer': vectorizer,
                    'feature_names': feature_names
                }


            nmf_model = NMF(n_components=effective_num_topics, random_state=42, max_iter=100)
            nmf_model.fit(tfidf_matrix)
            topics = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_words_idx = topic.argsort()[-10:][::-1] # Get indices of top 10 words
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append({
                    'topic_id': topic_idx,
                    'top_words': top_words,
                    'weights': topic[top_words_idx]
                })
            return {
                'topics': topics,
                'model': nmf_model,
                'vectorizer': vectorizer,
                'feature_names': feature_names
            }
        
        except ValueError as e:
            print(f"Error during NMF topic extraction (ValueError): {e}")
            return {
                'topics': [],
                'model': None,
                'vectorizer': vectorizer,
                'feature_names': []
            }
        except Exception as e:
            print(f"An unexpected error occurred during NMF topic extraction: {e}")
            return {
                'topics': [],
                'model': None,
                'vectorizer': vectorizer,
                'feature_names': []
            }

    def extract_keywords_rake(self, text: str, num_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Extracts keywords using RAKE (Rapid Automatic Keyword Extraction algorithm).
        Returns a list of (keyword, score) tuples.
        """
        self.rake.extract_keywords_from_text(text)
        # Rake returns (score, phrase), so we need to reorder for consistency
        keywords = [(phrase, score) for score, phrase in self.rake.get_ranked_phrases_with_scores()]
        return keywords[:num_keywords]

    def extract_keywords_keybert(self, text: str, num_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Extracts keywords using KeyBERT, which leverages BERT embeddings.
        Returns a list of (keyword, score) tuples.
        """
        try:
            # Using use_maxsum and diversity for more diverse keywords
            keywords = self.keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=num_keywords,
                use_maxsum=True,
                nr_candidates=20,
                diversity=0.5
            )
            return keywords
        except Exception as e:
            print(f"KeyBERT extraction failed: {e}")
            return []

    def extract_keywords_tfidf(self, texts: List[str], num_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Extracts keywords based on TF-IDF scores.
        Calculates mean TF-IDF scores across documents for each term.
        """
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Skip if all processed texts are empty
        if not any(processed_texts):
            print("Warning: No valid texts after preprocessing for TF-IDF keyword extraction.")
            return []

        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            min_df=1, # Keep min_df=1 for keyword extraction
            max_df=0.8
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            if tfidf_matrix.shape[1] == 0: # Check if any features were extracted
                print("Warning: TF-IDF matrix has no features for keyword extraction.")
                return []
            
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0) # Average TF-IDF score for each term
            top_indices = mean_scores.argsort()[-num_keywords:][::-1] # Get indices of top keywords
            keywords = [(feature_names[i], mean_scores[i]) for i in top_indices]
            return keywords
        
        except ValueError as e:
            print(f"TF-IDF keyword extraction failed (ValueError): {e}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred during TF-IDF keyword extraction: {e}")
            return []

    def load_texts_from_directory(self, directory_path: str) -> List[str]:
        """
        Loads all text content from .txt files within a specified directory and its subdirectories.
        """
        if not os.path.isdir(directory_path):
            print(f"Error: Directory not found at '{directory_path}'. Please provide a valid path.")
            return []

        all_texts = []
        print(f"Loading texts from directory and its subdirectories: {directory_path}")
        # os.walk generates the file names in a directory tree by walking the tree top-down
        for root, _, files in os.walk(directory_path):
            for filename in files:
                if filename.endswith(".txt"):
                    file_path = os.path.join(root, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            all_texts.append(content)
                            print(f"  Loaded: {file_path}") # Print full path to show subdirectory loading
                    except Exception as e:
                        print(f"  Error reading '{file_path}': {e}")
        
        if not all_texts:
            print(f"No .txt files found or readable in '{directory_path}' or its subdirectories.")
        return all_texts


    def run_pipeline(self, text_data: List[str], num_topics: int = 5, num_keywords: int = 10) -> Dict[str, Any]:
        """
        Runs the complete NLP pipeline on a list of text documents.
        """
        if not text_data:
            print("No text data provided to the pipeline. Returning empty results.")
            return {}

        print("Starting NLP Pipeline...")
        combined_text = ' '.join(text_data)
        processed_texts_for_stats = [self.preprocess_text(text) for text in text_data] # For accurate stats
        
        # Calculate stats from original combined text before extensive preprocessing for analysis
        total_characters = len(combined_text)
        total_words = len(combined_text.split())

        print("1. Preprocessing complete.")

        print("2. Extracting named entities...")
        entities = self.extract_entities(combined_text)

        print("3. Performing sentiment analysis...")
        sentiment_results = self.analyze_sentiment(combined_text)

        print("4. Extracting topics...")
        if len(text_data) > 1:
            bertopic_topics = self.extract_topics_bertopic(text_data, num_topics)
            nmf_topics = self.extract_topics_nmf(text_data, num_topics)
        else:
            # For single document, split into sentences for topic modeling if enough sentences
            sentences = sent_tokenize(combined_text)
            # Filter out very short sentences for topic modeling
            meaningful_sentences = [s for s in sentences if len(s.split()) > 5] 
            
            if len(meaningful_sentences) >= 2: # Need at least 2 "documents" (sentences) for NMF/BERTopic to work well
                print(f"Single document, splitting into {len(meaningful_sentences)} sentences for topic modeling.")
                bertopic_topics = self.extract_topics_bertopic(meaningful_sentences, min(num_topics, max(1, len(meaningful_sentences) // 2)))
                nmf_topics = self.extract_topics_nmf(meaningful_sentences, min(num_topics, max(1, len(meaningful_sentences) // 2)))
            else:
                print("Not enough sentences in single document for robust topic modeling. Skipping.")
                bertopic_topics = {'topics': []}
                nmf_topics = {'topics': []}

        print("5. Extracting keywords...")
        # RAKE and KeyBERT work best on a single combined text
        rake_keywords = self.extract_keywords_rake(combined_text, num_keywords)
        keybert_keywords = self.extract_keywords_keybert(combined_text, num_keywords)
        # TF-IDF keyword extraction benefits from multiple documents, so pass original list
        tfidf_keywords = self.extract_keywords_tfidf(text_data, num_keywords)

        self.results = {
            'preprocessing': {
                'original_texts': text_data,
                'processed_texts_for_stats': processed_texts_for_stats, # Store for reference
                'total_characters': total_characters,
                'total_words': total_words
            },
            'entities': entities,
            'sentiment': sentiment_results,
            'topics': {
                'bertopic': bertopic_topics,
                'nmf': nmf_topics
            },
            'keywords': {
                'rake': rake_keywords,
                'keybert': keybert_keywords,
                'tfidf': tfidf_keywords
            }
        }

        print("Pipeline completed successfully!")
        return self.results

    def generate_report(self, results: Dict[str, Any] = None, output_file_path: str = None) -> str:
        """
        Generates a human-readable report from the NLP analysis results.
        If output_file_path is provided, writes the report to that file.
        Otherwise, returns the report as a string.
        """
        if results is None: 
            results = self.results
        if not results:
            report_content = "No analysis results available. Please run the pipeline first."
            if output_file_path:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"Report saved to: {output_file_path}")
                return "" 
            return report_content

        report_lines = [] #
        report_lines.append("=" * 60)
        report_lines.append("           NLP ANALYSIS REPORT")
        report_lines.append("=" * 60)

        report_lines.append(f"\n📊 DOCUMENT STATISTICS")
        report_lines.append(f"Total Characters: {results['preprocessing']['total_characters']:,}")
        report_lines.append(f"Total Words: {results['preprocessing']['total_words']:,}")
        report_lines.append(f"Number of Documents: {len(results['preprocessing']['original_texts'])}")

        report_lines.append(f"\n😊 SENTIMENT ANALYSIS")
        sentiment = results['sentiment']
        report_lines.append(f"TextBlob Sentiment: {sentiment['textblob']['label'].title()}")
        report_lines.append(f"Polarity: {sentiment['textblob']['polarity']:.3f} (-1 to 1)")
        report_lines.append(f"Subjectivity: {sentiment['textblob']['subjectivity']:.3f} (0 to 1)")
        report_lines.append(f"BERT Sentiment: {sentiment['transformer']['overall_label'].title()}")
        report_lines.append(f"Average Star Rating: {sentiment['transformer']['average_star_rating']:.2f} ⭐")

        report_lines.append(f"\n🏷️  NAMED ENTITIES")
        entities = results['entities']
        if entities:
            for entity_type, entity_list in entities.items():
                report_lines.append(f"{entity_type}: {', '.join(entity_list[:5])}")
                if len(entity_list) > 5:
                    report_lines.append(f"  (... and {len(entity_list) - 5} more)")
        else:
            report_lines.append("No named entities found.")

        report_lines.append(f"\n📋 TOPIC ANALYSIS")
        bertopic_topics = results['topics']['bertopic']
        if bertopic_topics.get('topics'):
            report_lines.append("BERTopic Extracted Topics:")
            for topic in bertopic_topics['topics']:
                # top_words is a list of tuples (word, score) from BERTopic
                top_words_only = [w[0] for w in topic['top_words']] 
                report_lines.append(f"  Topic {topic['topic_id']}: {', '.join(top_words_only[:5])}")
        else:
            report_lines.append("No BERTopic topics extracted.")


        nmf_topics = results['topics']['nmf']
        if nmf_topics.get('topics'):
            report_lines.append("NMF Topics:")
            for topic in nmf_topics['topics']:
                report_lines.append(f"  Topic {topic['topic_id']}: {', '.join(topic['top_words'][:5])}")
        else:
            report_lines.append("No NMF topics extracted.")


        report_lines.append(f"\n🔑 KEYWORDS")
        rake_keywords = results['keywords']['rake']
        if rake_keywords:
            report_lines.append("RAKE Keywords:")
            for keyword, score in rake_keywords[:5]: # RAKE keywords are now (phrase, score)
                report_lines.append(f"  {keyword}: {float(score):.3f}")
        else:
            report_lines.append("No RAKE keywords extracted.")


        keybert_keywords = results['keywords']['keybert']
        if keybert_keywords:
            report_lines.append("KeyBERT Keywords:")
            for keyword, score in keybert_keywords[:5]:
                report_lines.append(f"  {keyword}: {float(score):.3f}")
        else:
            report_lines.append("No KeyBERT keywords extracted.")


        tfidf_keywords = results['keywords']['tfidf']
        if tfidf_keywords:
            report_lines.append("TF-IDF Keywords:")
            for keyword, score in tfidf_keywords[:5]:
                report_lines.append(f"  {keyword}: {float(score):.3f}")
        else:
            report_lines.append("No TF-IDF keywords extracted.")

        report_content = "\n".join(report_lines)

        if output_file_path:
            try:
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                print(f"Report successfully saved to: {output_file_path}")
                return "" # Return empty string as report is saved to file
            except Exception as e:
                print(f"Error saving report to file '{output_file_path}': {e}")
                return report_content # Return content if file saving fails
        else:
            return report_content # Return content if no file path is specified

if __name__ == "__main__":
    # Example 1: Using hardcoded texts
    pipeline_hardcoded = NLPPipeline()

    # Example 2: Loading texts from a directory
    print("\n--- Running Pipeline with Texts from Directory ---")
    # Create a dummy directory and some text files for demonstration
    dummy_dir = "extracted_content"

    # Initialize a new pipeline instance or reuse the existing one
    pipeline_directory = NLPPipeline() 
    
    # Load texts from the dummy directory
    documents_from_dir = pipeline_directory.load_texts_from_directory(dummy_dir)
    
    if documents_from_dir:
        results_from_dir = pipeline_directory.run_pipeline(documents_from_dir, num_topics=2, num_keywords=5)
        # Generate report and save to a file
        report_file_directory = "nlp_report_from_directory.txt"
        pipeline_directory.generate_report(results_from_dir, output_file_path=report_file_directory)
        print(f"Check '{report_file_directory}' for the analysis report.")
    else:
        print("No documents were loaded from the directory. Cannot run pipeline.")


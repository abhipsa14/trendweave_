from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
import glob
import pandas as pd
from tqdm import tqdm

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """Initialize the sentiment analyzer with a pre-trained model."""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ['negative', 'positive']
        
    def analyze_single_text(self, text):
        """Analyze sentiment of a single text."""
        # Preprocess input
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        # print(self.model.accuracy())
        # Get predicted class and confidence
        predicted_class = torch.argmax(probs).item()
        confidence = probs[0][predicted_class].item()
        return {
            'sentiment': self.labels[predicted_class],
            'confidence': confidence,
            'probabilities': {
    'negative': probs[0][0].item(),
    'positive': probs[0][1].item()
}

        }
    
    def analyze_text_files(self, dataset_path, output_file=None):
        """
        Analyze sentiment for all .txt files in a dataset directory (including subdirectories).
        
        Args:
            dataset_path (str): Path to the directory containing folders with .txt files
            output_file (str, optional): Path to save results as CSV
            
        Returns:
            list: List of dictionaries containing analysis results
        """
        # Find all .txt files recursively in the dataset directory and subdirectories
        txt_files = glob.glob(os.path.join(dataset_path, "**", "*.txt"), recursive=True)
        
        if not txt_files:
            print(f"No .txt files found in {dataset_path} or its subdirectories")
            return []
        
        print(f"Found {len(txt_files)} .txt files to analyze...")
        
        results = []
        
        # Process each file with progress bar
        for file_path in tqdm(txt_files, desc="Analyzing files"):
            try:
                # Read the file content
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                
                # Skip empty files
                if not content:
                    print(f"Warning: Empty file skipped - {file_path}")
                    continue
                
                # Analyze sentiment
                analysis = self.analyze_single_text(content)
                
                # Store results with folder information
                result = {
                    'filename': os.path.basename(file_path),
                    'folder': os.path.basename(os.path.dirname(file_path)),
                    'filepath': file_path,
                    'relative_path': os.path.relpath(file_path, dataset_path),
                    'text_preview': content[:100] + "..." if len(content) > 100 else content,
                    'sentiment': analysis['sentiment'],
                    'confidence': analysis['confidence'],
                    'negative_prob': analysis['probabilities']['negative'],
                    'positive_prob': analysis['probabilities']['positive']
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
        
        # Save results to CSV if specified
        if output_file:
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        
        return results
    
    def print_summary(self, results):
        """Print a summary of the sentiment analysis results."""
        if not results:
            print("No results to summarize.")
            return
        
        # Count sentiments
        sentiment_counts = {}
        folder_counts = {}
        total_confidence = 0
        
        for result in results:
            sentiment = result['sentiment']
            folder = result['folder']
            
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            folder_counts[folder] = folder_counts.get(folder, 0) + 1
            total_confidence += result['confidence']
        
        # Print summary
        print("\n" + "="*50)
        print("SENTIMENT ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total files analyzed: {len(results)}")
        print(f"Number of folders processed: {len(folder_counts)}")
        print(f"Average confidence: {total_confidence/len(results):.4f}")
        
        print("\nFiles per folder:")
        for folder, count in sorted(folder_counts.items()):
            print(f"  {folder}: {count} files")
        
        print("\nSentiment Distribution:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {sentiment.capitalize()}: {count} files ({percentage:.1f}%)")
        
        # Show top confident predictions
        print("\nTop 5 Most Confident Predictions:")
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        for i, result in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {result['folder']}/{result['filename']}: {result['sentiment']} ({result['confidence']:.4f})")
        
        # Show sentiment distribution by folder
        print("\nSentiment by Folder:")
        folder_sentiments = {}
        for result in results:
            folder = result['folder']
            sentiment = result['sentiment']
            if folder not in folder_sentiments:
                folder_sentiments[folder] = {'negative': 0, 'neutral': 0, 'positive': 0}
            folder_sentiments[folder][sentiment] += 1
        
        for folder, sentiments in sorted(folder_sentiments.items()):
            total = sum(sentiments.values())
            print(f"  {folder}:")
            for sentiment, count in sentiments.items():
                if count > 0:
                    percentage = (count / total) * 100
                    print(f"    {sentiment}: {count} ({percentage:.1f}%)")


def main():
    """Example usage of the sentiment analyzer."""
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Analyze all txt files in nested directories
    dataset_path = "extracted_content"  
    
    # Run batch analysis on nested directories
    print(f"\nAnalyzing all .txt files in '{dataset_path}' and its subdirectories...")
    results = analyzer.analyze_text_files(dataset_path, output_file="sentiment_results.csv")
    
    # Print detailed summary
    analyzer.print_summary(results)
    
    # Optional: Filter results by folder
    # if results:
    #     print(f"\nExample: Files from first folder ({results[0]['folder']}):")
    #     folder_results = [r for r in results if r['folder'] == results[0]['folder']]
    #     for result in folder_results[:3]:  # Show first 3 files from first folder
    #         print(f"  {result['filename']}: {result['sentiment']} ({result['confidence']:.4f})")
    

if __name__ == "__main__":
    main()
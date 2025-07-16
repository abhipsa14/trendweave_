# Fashion Trend Analysis Pipeline

A comprehensive NLP pipeline for analyzing fashion trends from extracted PDF content. This project combines sentiment analysis, topic modeling, keyword extraction, named entity recognition, and machine learning predictions to generate detailed fashion trend reports.

## Project Structure

```
trendweaveexp/
├── fashion_trend_analysis.py      # Main analysis script
├── extraction.py                  # PDF extraction script
├── setup.py                      # Setup and installation script
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── dataset/                      # Input PDF files
├── extracted_content/            # Extracted text content from PDFs
├── nlp_pipeline/                 # NLP analysis modules
│   ├── __init__.py
│   ├── data_loader.py           # Data loading and preprocessing
│   ├── sentiment_analyzer.py    # Sentiment analysis
│   ├── topic_modeling.py        # Topic modeling (BERTopic)
│   ├── keyword_extraction.py    # Keyword extraction (RAKE, KeyBERT)
│   ├── ner_analyzer.py          # Named entity recognition
│   ├── trend_predictor.py       # Prediction and forecasting
│   └── report_generator.py      # Report generation
└── Tesseract-OCR/               # OCR engine
```

## Features

### 1. Data Loading (`data_loader.py`)
- Loads text files from extracted PDF content
- Preprocesses text for analysis
- Categorizes documents by type, season, and location
- Provides summary statistics

### 2. Sentiment Analysis (`sentiment_analyzer.py`)
- **TextBlob**: Polarity and subjectivity analysis
- **VADER**: Detailed sentiment scoring
- Category-wise sentiment analysis
- Fashion-specific sentiment insights

### 3. Topic Modeling (`topic_modeling.py`)
- **BERTopic**: Advanced topic modeling with transformers
- **LDA**: Traditional topic modeling
- Topic trend analysis across categories and seasons
- Topic evolution tracking

### 4. Keyword Extraction (`keyword_extraction.py`)
- **RAKE**: Rapid Automatic Keyword Extraction
- **KeyBERT**: BERT-based keyword extraction
- **TF-IDF**: Statistical keyword extraction
- Fashion-specific keyword analysis

### 5. Named Entity Recognition (`ner_analyzer.py`)
- **spaCy**: Standard NER for persons, organizations, locations
- **Fashion-specific entities**: Brands, colors, materials, styles, garments
- Brand analysis and trends
- Color and material trend analysis

### 6. Prediction & Forecasting (`trend_predictor.py`)
- **Machine Learning Models**: Random Forest, Gradient Boosting, Logistic Regression
- **Multi-task prediction**: Sentiment trends, popularity, innovation levels
- **Forecasting**: Category-specific and market-wide predictions
- **Accuracy reporting**: Cross-validation and performance metrics

### 7. Report Generation (`report_generator.py`)
- Comprehensive HTML/text reports
- Executive summaries
- Detailed analysis sections
- Recommendations and insights
- JSON export for further analysis

## Installation

1. **Clone or download the project**
2. **Install dependencies:**
   ```bash
   python setup.py
   ```
   
   Or manually:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

## Usage

### Basic Usage
```bash
python fashion_trend_analysis.py
```

### Expected Output
The script will generate:
- **Text Report**: `fashion_trend_analysis_report_YYYYMMDD_HHMMSS.txt`
- **JSON Results**: `fashion_trend_analysis_results_YYYYMMDD_HHMMSS.json`
- **Log File**: `fashion_trend_analysis.log`

### Sample Report Sections

#### Executive Summary
- Data overview and statistics
- Category and season distribution
- Overall sentiment analysis
- Key forecasts and model performance

#### Sentiment Analysis
- Overall sentiment metrics
- Category-wise sentiment breakdown
- Fashion-specific sentiment insights
- Most positive/negative documents

#### Topic Modeling
- Identified topics with keywords
- Topic trends by category and season
- Topic evolution analysis

#### Keyword Analysis
- Top keywords by extraction method
- Category and season-wise keyword trends
- Fashion-specific keyword insights

#### Named Entity Recognition
- Brand mention analysis
- Color and material trends
- Entity distribution by category
- Geographic fashion center analysis

#### Predictions & Forecasting
- Model performance metrics
- Trend predictions with confidence scores
- Category-specific forecasts
- Market-wide predictions

#### Recommendations
- Strategic recommendations based on analysis
- Category-specific action items
- Color and material focus areas
- Brand positioning insights

## Model Performance

The pipeline uses multiple machine learning models with cross-validation:

- **Random Forest**: Ensemble method for robust predictions
- **Gradient Boosting**: Advanced ensemble with sequential learning
- **Logistic Regression**: Linear model for interpretable results

**Accuracy Metrics**:
- Test set accuracy
- Cross-validation mean ± standard deviation
- Feature importance analysis
- Confidence scores for predictions

## Customization

### Adding New Analysis Methods
1. Create new module in `nlp_pipeline/`
2. Follow the existing pattern with batch processing
3. Add to `__init__.py` exports
4. Integrate into main pipeline

### Modifying Categories
Edit the categorization logic in `data_loader.py`:
```python
def get_category(filename):
    # Add your categorization logic here
    pass
```

### Adding Fashion Entities
Extend the entity lists in `ner_analyzer.py`:
```python
self.fashion_brands.update(['new_brand1', 'new_brand2'])
self.fashion_colors.update(['new_color1', 'new_color2'])
```

## Dependencies

Main dependencies include:
- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning
- **spacy**: NLP and NER
- **transformers**: BERT models
- **bertopic**: Topic modeling
- **keybert**: Keyword extraction
- **rake-nltk**: Keyword extraction
- **textblob**: Sentiment analysis
- **nltk**: Natural language processing
- **matplotlib/seaborn**: Visualization (optional)

## Troubleshooting

### Common Issues

1. **spaCy model not found**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **NLTK data missing**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('vader_lexicon')
   ```

3. **Memory issues with large datasets**:
   - Process documents in smaller batches
   - Reduce BERTopic model complexity
   - Use text preprocessing to reduce document size

4. **Low prediction accuracy**:
   - Increase training data
   - Adjust feature engineering
   - Tune model hyperparameters
   - Check data quality and preprocessing

## Performance Tips

- **Preprocessing**: Clean and normalize text thoroughly
- **Feature Engineering**: Create domain-specific features
- **Model Selection**: Use ensemble methods for better accuracy
- **Validation**: Use stratified cross-validation
- **Monitoring**: Check logs for processing status

## License

This project is developed for fashion trend analysis and research purposes.

## Contact

For questions or issues, please check the logs and ensure all dependencies are properly installed.

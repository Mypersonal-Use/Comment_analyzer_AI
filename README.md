# eConsultation AI

**AI-powered sentiment analysis and comment processing for eConsultation module**

## Overview

The eConsultation AI module is a comprehensive solution designed to analyze stakeholder comments and feedback received during public consultation processes. It leverages advanced AI techniques to provide:

- **Sentiment Analysis**: Automated classification of comments as positive, negative, or neutral
- **Text Summarization**: Generation of concise summaries for individual comments and global overviews
- **Word Cloud Visualization**: Visual representation of key themes and frequently used terms
- **Comprehensive Reporting**: Detailed analysis reports with charts and statistics

## Problem Statement

When substantial volumes of comments are received on draft legislation during public consultation, there exists a risk of certain observations being inadvertently overlooked or inadequately analyzed. This AI-powered system ensures that all remarks are systematically analyzed and duly considered.

## Features

### Core Functionality
- ✅ **Multi-method Sentiment Analysis** (VADER, TextBlob, Transformer models)
- ✅ **Text Summarization** (Extractive and Abstractive methods)
- ✅ **Word Cloud Generation** with customizable themes
- ✅ **Batch Processing** for large datasets
- ✅ **Multiple Input Formats** (CSV, JSON, TXT, Excel)
- ✅ **Comprehensive Reporting** with visualizations

### Advanced Features
- 🔧 **Configurable Analysis Parameters**
- 📊 **Interactive Dashboards** (with Plotly)
- 🎨 **Themed Visualizations** (Government, Professional, Modern)
- 📈 **Performance Monitoring**
- 🔄 **Ensemble Predictions**
- 📝 **Export Capabilities** (JSON, CSV, HTML)

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Dependencies Installation

```bash
# Clone or download the project
cd econsultation-ai

# Install required packages
pip install -r requirements.txt

# Download NLTK data (if needed)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Optional Dependencies

For enhanced functionality:
```bash
# For transformer-based models (optional but recommended)
pip install torch transformers

# For interactive visualizations
pip install plotly

# For web interface (future enhancement)
pip install streamlit flask
```

## Quick Start

### Basic Usage

```python
from src.econsultation_ai import EConsultationAI

# Initialize the system
ai_system = EConsultationAI()

# Analyze a single comment
result = ai_system.analyze_single_comment(
    text="I strongly support these proposed amendments.",
    comment_id="comment_001"
)

print(f"Sentiment: {result.sentiment.overall_sentiment}")
print(f"Summary: {result.summary.summary}")
```

### Batch Processing

```python
# Analyze multiple comments
comments = [
    "These amendments are excellent and well thought out.",
    "I have concerns about the implementation timeline.",
    "The approach seems balanced and reasonable."
]

batch_result = ai_system.analyze_batch(
    comments=comments,
    output_dir="results/analysis",
    generate_reports=True
)

print(f"Sentiment Distribution: {batch_result.sentiment_distribution}")
```

### File Processing

```python
# Analyze comments from a file
result = ai_system.analyze_from_file(
    file_path="data/stakeholder_comments.csv",
    output_dir="results/file_analysis"
)
```

## Configuration

The system supports extensive configuration for customizing analysis behavior:

```python
config = {
    # Sentiment Analysis
    'sentiment_methods': ['vader', 'textblob'],
    'default_sentiment_method': 'vader',
    
    # Text Summarization
    'default_summarization_method': 'textrank',
    'max_summary_sentences': 3,
    'max_global_summary_sentences': 5,
    
    # Word Cloud
    'wordcloud_language': 'english',
    'wordcloud_theme': 'government',
    'custom_stopwords': ['amendment', 'provision'],
    
    # Data Processing
    'min_comment_length': 20,
    'max_comment_length': 2000,
    'remove_duplicates': True,
    
    # Output
    'encoding': 'utf-8'
}

ai_system = EConsultationAI(config=config)
```

## Input Formats

The system supports multiple input formats:

### CSV Format
```csv
id,text,author,category,timestamp
1,"Comment text here","Author Name","Category","2024-01-15 10:30:00"
```

### JSON Format
```json
[
    {
        "id": "1",
        "text": "Comment text here",
        "author": "Author Name",
        "category": "Category",
        "timestamp": "2024-01-15 10:30:00"
    }
]
```

### Text Format
Simple text file with comments separated by double newlines or delimiter markers.

## Output and Reports

The system generates comprehensive reports including:

### 1. Analysis Results (JSON)
Detailed results with sentiment scores, summaries, and metadata.

### 2. Visual Reports
- Sentiment distribution charts
- Word clouds with customizable themes
- Keyword frequency charts
- Processing timeline visualizations

### 3. Summary Reports (Text)
Human-readable summary with key insights and statistics.

### 4. Interactive Dashboards (HTML)
Interactive visualizations using Plotly (when available).

## Examples

### Running Examples

```bash
# Basic usage example
python examples/basic_usage.py

# Advanced features demonstration
python examples/advanced_usage.py
```

### Sample Output Structure
```
results/
├── analysis_results.json          # Detailed analysis results
├── analysis_summary.txt           # Human-readable summary
├── stakeholder_wordcloud.png      # Word cloud visualization
├── keyword_frequency_chart.png    # Top keywords chart
├── sentiment_distribution.png     # Sentiment analysis chart
├── top_keywords.csv              # Keywords export
└── interactive_dashboard.html     # Interactive dashboard
```

## API Reference

### Main Classes

#### EConsultationAI
Main application interface integrating all components.

**Key Methods:**
- `analyze_single_comment(text, comment_id)` - Analyze individual comment
- `analyze_batch(comments, output_dir)` - Batch processing
- `analyze_from_file(file_path, output_dir)` - File processing
- `validate_input_file(file_path)` - Input validation

#### SentimentAnalyzer
Handles sentiment analysis using multiple methods.

**Methods:**
- `analyze_single_comment(text, method)` - Single comment analysis
- `get_ensemble_prediction(text)` - Multi-method ensemble
- `get_sentiment_distribution(results)` - Aggregate statistics

#### TextSummarizer
Provides text summarization capabilities.

**Methods:**
- `summarize_single_text(text, method, max_sentences)` - Text summarization
- `create_global_summary(texts, method)` - Global summary generation

#### WordCloudGenerator
Generates word clouds and keyword analysis.

**Methods:**
- `generate_wordcloud(texts, save_path)` - Word cloud creation
- `extract_keywords(texts, top_n)` - Keyword extraction
- `generate_frequency_chart(result, save_path)` - Frequency visualization

## Performance

### Benchmarks
- **Single Comment**: ~0.1-0.5 seconds (depending on method)
- **Batch Processing**: ~0.1 seconds per comment average
- **File Processing**: Linear scaling with comment count

### Optimization Tips
1. Use VADER for faster sentiment analysis
2. Limit transformer models for large batches
3. Adjust summary sentence limits for performance
4. Use batch processing for multiple comments

## Troubleshooting

### Common Issues

1. **ImportError: transformers not found**
   ```bash
   pip install transformers torch
   ```

2. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

3. **Memory Issues with Large Files**
   - Process files in smaller batches
   - Reduce max_words in word cloud generation
   - Use lighter sentiment analysis methods

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

### Development Setup
1. Fork the repository
2. Install development dependencies: `pip install -r requirements.txt`
3. Run tests: `python -m pytest tests/`
4. Follow PEP 8 style guidelines

### Adding New Features
- Sentiment analysis methods in `sentiment_analyzer.py`
- Summarization methods in `text_summarizer.py`
- Visualization themes in `visualization_reporter.py`

## License

This project is developed for the Ministry of Corporate Affairs (MCA) eConsultation module. 
Please refer to the specific licensing terms provided by MCA.

## Contact and Support

For technical support or feature requests related to the eConsultation AI module, please contact the development team through the appropriate MCA channels.

## Version History

- **v1.0.0** - Initial release with core functionality
  - Sentiment analysis (VADER, TextBlob)
  - Text summarization (multiple methods)
  - Word cloud generation
  - Batch processing capabilities
  - Comprehensive reporting

## Acknowledgments

This solution leverages several open-source libraries:
- **NLTK** for natural language processing
- **TextBlob** for sentiment analysis
- **VADER Sentiment** for lexicon-based analysis
- **Sumy** for text summarization
- **WordCloud** for visualization
- **Matplotlib/Seaborn** for charts
- **Plotly** for interactive dashboards
- **Transformers** for advanced NLP models

---

**eConsultation AI** - Enhancing public consultation through intelligent comment analysis.

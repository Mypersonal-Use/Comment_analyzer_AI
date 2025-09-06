"""
Word Cloud Generation Module for eConsultation AI
Creates visual representations of keyword density and frequency
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import Counter
import numpy as np

# Core NLP libraries
import nltk
from textblob import TextBlob

# Visualization libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from wordcloud import WordCloud

# Optional seaborn import
try:
    import seaborn as sns
except ImportError:
    sns = None

# Data processing
import pandas as pd

# Download required NLTK data with error handling
def ensure_nltk_data():
    """Ensure required NLTK data is available"""
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            logging.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logging.warning(f"Could not download stopwords: {e}")
    
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            logging.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logging.warning(f"Could not download punkt: {e}")

# Ensure NLTK data is available
ensure_nltk_data()

# Import NLTK functions with fallback handling
try:
    from nltk.corpus import stopwords
    STOPWORDS_AVAILABLE = True
except ImportError:
    STOPWORDS_AVAILABLE = False
    logging.warning("NLTK stopwords not available")

# Define our own tokenization function to avoid NLTK dependency issues
def safe_word_tokenize(text):
    """Safe word tokenization that doesn't depend on NLTK punkt data"""
    # Simple but effective word tokenization using regex
    import re
    # Split on whitespace and punctuation but keep alphanumeric words
    words = re.findall(r'\b\w+\b', text.lower())
    return words

@dataclass
class WordCloudResult:
    """Data class to store word cloud generation results"""
    word_frequencies: Dict[str, int]
    top_keywords: List[Tuple[str, int]]
    total_words: int
    unique_words: int
    avg_word_frequency: float
    image_path: Optional[str] = None

class WordCloudGenerator:
    """
    Comprehensive word cloud generation class with customization options
    """
    
    def __init__(self, language: str = 'english', custom_stopwords: List[str] = None):
        """
        Initialize the word cloud generator
        
        Args:
            language: Language for stopwords
            custom_stopwords: Additional custom stopwords to exclude
        """
        self.language = language
        
        # Load stopwords with robust error handling
        self.stopwords = set()
        if STOPWORDS_AVAILABLE:
            try:
                self.stopwords = set(stopwords.words(language))
            except LookupError:
                logging.warning(f"NLTK stopwords data not available for language: {language}")
            except Exception as e:
                logging.warning(f"Could not load stopwords for language {language}: {e}")
        else:
            logging.warning("NLTK stopwords module not available")
        
        # Add common generic stopwords
        generic_stopwords = {
            'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'will',
            'also', 'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
            'said', 'says', 'say', 'think', 'thought', 'believe', 'feel', 'seems',
            'one', 'two', 'three', 'first', 'second', 'third', 'last', 'next',
            'much', 'many', 'more', 'most', 'less', 'least', 'very', 'quite', 'really',
            'get', 'got', 'getting', 'goes', 'went', 'going', 'come', 'came', 'coming',
            'make', 'made', 'making', 'take', 'took', 'taken', 'taking',
            'see', 'saw', 'seen', 'seeing', 'look', 'looked', 'looking',
            'way', 'ways', 'time', 'times', 'year', 'years', 'day', 'days',
            'good', 'better', 'best', 'bad', 'worse', 'worst', 'new', 'old'
        }
        
        self.stopwords.update(generic_stopwords)
        
        # Add custom stopwords if provided
        if custom_stopwords:
            self.stopwords.update([word.lower() for word in custom_stopwords])
            
        # Default word cloud parameters
        self.default_params = {
            'width': 1200,
            'height': 800,
            'max_words': 100,
            'background_color': 'white',
            'colormap': 'viridis',
            'relative_scaling': 0.5,
            'min_font_size': 10
        }
    
    def preprocess_text(self, text: str, remove_numbers: bool = True, 
                       min_word_length: int = 3) -> List[str]:
        """
        Preprocess text for word cloud generation
        
        Args:
            text: Input text to preprocess
            remove_numbers: Whether to remove numeric values
            min_word_length: Minimum length for words to include
            
        Returns:
            List of cleaned words
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Use our safe tokenization that doesn't depend on NLTK data
        words = safe_word_tokenize(text)
        
        # Filter words
        filtered_words = []
        for word in words:
            # Skip if in stopwords
            if word in self.stopwords:
                continue
                
            # Skip if too short
            if len(word) < min_word_length:
                continue
                
            # Skip if numeric and remove_numbers is True
            if remove_numbers and word.isdigit():
                continue
                
            # Skip if contains only numbers and special characters
            if re.match(r'^[\d\W]+$', word):
                continue
                
            filtered_words.append(word)
        
        return filtered_words
    
    def extract_keywords(self, texts: Union[str, List[str]], top_n: int = 50) -> WordCloudResult:
        """
        Extract keywords and their frequencies from text(s)
        
        Args:
            texts: Single text string or list of text strings
            top_n: Number of top keywords to return
            
        Returns:
            WordCloudResult with keyword analysis
        """
        # Handle input format
        if isinstance(texts, str):
            all_text = texts
        else:
            all_text = ' '.join(texts)
        
        # Preprocess text
        words = self.preprocess_text(all_text)
        
        # Calculate frequencies
        word_frequencies = Counter(words)
        
        # Get top keywords
        top_keywords = word_frequencies.most_common(top_n)
        
        # Calculate statistics
        total_words = len(words)
        unique_words = len(word_frequencies)
        avg_frequency = total_words / unique_words if unique_words > 0 else 0
        
        return WordCloudResult(
            word_frequencies=dict(word_frequencies),
            top_keywords=top_keywords,
            total_words=total_words,
            unique_words=unique_words,
            avg_word_frequency=avg_frequency
        )
    
    def generate_wordcloud(self, texts: Union[str, List[str]], 
                          save_path: str = None,
                          custom_params: Dict = None) -> WordCloudResult:
        """
        Generate word cloud from text(s)
        
        Args:
            texts: Single text string or list of text strings
            save_path: Path to save the word cloud image
            custom_params: Custom parameters for word cloud generation
            
        Returns:
            WordCloudResult with generated word cloud
        """
        # Extract keywords first
        result = self.extract_keywords(texts)
        
        if not result.word_frequencies:
            logging.warning("No words found for word cloud generation")
            return result
        
        # Merge default and custom parameters
        params = self.default_params.copy()
        if custom_params:
            params.update(custom_params)
        
        # Create word cloud
        wordcloud = WordCloud(**params).generate_from_frequencies(result.word_frequencies)
        
        # Create visualization
        plt.figure(figsize=(params['width']/100, params['height']/100))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Stakeholder Comments - Word Cloud', fontsize=16, fontweight='bold')
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300, 
                       facecolor='white', edgecolor='none')
            result.image_path = save_path
            logging.info(f"Word cloud saved to: {save_path}")
        
        plt.show()
        
        return result
    
    def generate_frequency_chart(self, result: WordCloudResult, 
                               top_n: int = 20, save_path: str = None) -> str:
        """
        Generate frequency bar chart for top keywords
        
        Args:
            result: WordCloudResult object with keyword data
            top_n: Number of top words to display
            save_path: Path to save the chart
            
        Returns:
            Path to saved chart or empty string
        """
        top_words = result.top_keywords[:top_n]
        
        if not top_words:
            logging.warning("No words available for frequency chart")
            return ""
        
        words, frequencies = zip(*top_words)
        
        # Create horizontal bar chart
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(words)), frequencies, color='steelblue', alpha=0.7)
        
        # Customize chart
        plt.yticks(range(len(words)), words)
        plt.xlabel('Frequency', fontsize=12)
        plt.title(f'Top {top_n} Keywords by Frequency', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()  # Highest frequency at top
        
        # Add value labels on bars
        for i, (bar, freq) in enumerate(zip(bars, frequencies)):
            plt.text(freq + max(frequencies) * 0.01, i, str(freq), 
                    va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300,
                       facecolor='white', edgecolor='none')
            logging.info(f"Frequency chart saved to: {save_path}")
            
        plt.show()
        
        return save_path or ""
    
    def generate_themed_wordcloud(self, texts: Union[str, List[str]], 
                                 theme: str = 'government',
                                 save_path: str = None) -> WordCloudResult:
        """
        Generate themed word cloud with specific color schemes and styling
        
        Args:
            texts: Single text string or list of text strings  
            theme: Theme for styling ('government', 'legal', 'business', 'modern')
            save_path: Path to save the word cloud image
            
        Returns:
            WordCloudResult with themed word cloud
        """
        # Define theme parameters
        themes = {
            'government': {
                'colormap': 'Blues',
                'background_color': '#f8f9fa',
                'width': 1400,
                'height': 900
            },
            'legal': {
                'colormap': 'Greys',
                'background_color': 'white',
                'width': 1200,
                'height': 800
            },
            'business': {
                'colormap': 'Set2',
                'background_color': '#ffffff',
                'width': 1300,
                'height': 850
            },
            'modern': {
                'colormap': 'plasma',
                'background_color': '#1e1e1e',
                'width': 1500,
                'height': 1000
            }
        }
        
        theme_params = themes.get(theme, themes['government'])
        
        return self.generate_wordcloud(texts, save_path, theme_params)
    
    def create_comparative_wordcloud(self, text_groups: Dict[str, List[str]], 
                                   save_path: str = None) -> Dict[str, WordCloudResult]:
        """
        Create comparative word clouds for different groups of texts
        
        Args:
            text_groups: Dictionary with group names as keys and text lists as values
            save_path: Base path for saving images (group names will be appended)
            
        Returns:
            Dictionary of WordCloudResult objects for each group
        """
        results = {}
        
        fig, axes = plt.subplots(1, len(text_groups), figsize=(6*len(text_groups), 8))
        if len(text_groups) == 1:
            axes = [axes]
        
        for idx, (group_name, texts) in enumerate(text_groups.items()):
            # Generate word cloud for this group
            result = self.extract_keywords(texts)
            results[group_name] = result
            
            if result.word_frequencies:
                # Create word cloud
                wordcloud = WordCloud(**self.default_params).generate_from_frequencies(
                    result.word_frequencies
                )
                
                # Display in subplot
                axes[idx].imshow(wordcloud, interpolation='bilinear')
                axes[idx].set_title(f'{group_name}\n({result.total_words} words)', 
                                  fontsize=12, fontweight='bold')
                axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, f'No words found\nfor {group_name}', 
                             ha='center', va='center', fontsize=12)
                axes[idx].set_xlim(0, 1)
                axes[idx].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300,
                       facecolor='white', edgecolor='none')
            logging.info(f"Comparative word cloud saved to: {save_path}")
        
        plt.show()
        
        return results
    
    def get_keyword_statistics(self, results: Union[WordCloudResult, List[WordCloudResult]]) -> Dict:
        """
        Get comprehensive statistics about keywords
        
        Args:
            results: Single WordCloudResult or list of results
            
        Returns:
            Dictionary with keyword statistics
        """
        if isinstance(results, WordCloudResult):
            results = [results]
        
        # Aggregate statistics
        total_documents = len(results)
        all_word_counts = []
        all_unique_counts = []
        combined_frequencies = Counter()
        
        for result in results:
            all_word_counts.append(result.total_words)
            all_unique_counts.append(result.unique_words)
            combined_frequencies.update(result.word_frequencies)
        
        # Calculate statistics
        stats = {
            'total_documents': total_documents,
            'avg_words_per_document': np.mean(all_word_counts) if all_word_counts else 0,
            'avg_unique_words_per_document': np.mean(all_unique_counts) if all_unique_counts else 0,
            'total_unique_words': len(combined_frequencies),
            'most_common_words': combined_frequencies.most_common(10),
            'word_diversity': len(combined_frequencies) / sum(all_word_counts) if sum(all_word_counts) > 0 else 0
        }
        
        return stats
    
    def export_keywords_to_csv(self, result: WordCloudResult, filepath: str) -> str:
        """
        Export keyword frequencies to CSV file
        
        Args:
            result: WordCloudResult object
            filepath: Path to save CSV file
            
        Returns:
            Path to saved file
        """
        # Create DataFrame
        df = pd.DataFrame(result.top_keywords, columns=['Word', 'Frequency'])
        df['Rank'] = range(1, len(df) + 1)
        df = df[['Rank', 'Word', 'Frequency']]  # Reorder columns
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logging.info(f"Keywords exported to CSV: {filepath}")
        
        return filepath

"""
Text Summarization Module for eConsultation AI
Supports both extractive and abstractive summarization techniques
"""

import logging
import re
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np

# Core NLP libraries
import nltk
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer

# TF-IDF based summarization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional transformer support
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available for abstractive summarization.")

# Download required NLTK data
# Download required NLTK data with better error handling
def ensure_nltk_data():
    """Ensure required NLTK data is available"""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            logging.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logging.warning(f"Could not download punkt tokenizer: {e}")
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        try:
            logging.info("Downloading NLTK stopwords...")
            nltk.download('stopwords', quiet=True)
        except Exception as e:
            logging.warning(f"Could not download stopwords: {e}")

# Ensure NLTK data is available
ensure_nltk_data()

@dataclass
class SummaryResult:
    """Data class to store summarization results"""
    original_text: str
    summary: str
    method: str
    compression_ratio: float
    key_sentences: List[str]
    word_count_original: int
    word_count_summary: int

class TextSummarizer:
    """
    Comprehensive text summarization class supporting multiple methods
    """
    
    def __init__(self, language: str = 'english'):
        """
        Initialize the text summarizer
        
        Args:
            language: Language for tokenization and processing
        """
        self.language = language
        self.available_methods = ['textrank', 'lsa', 'lexrank', 'luhn', 'tfidf']
        
        if TRANSFORMERS_AVAILABLE:
            self.available_methods.append('transformer')
            self._initialize_transformer()
            
        # Initialize tokenizer with error handling
        try:
            self.tokenizer = Tokenizer(language)
        except Exception as e:
            logging.warning(f"Could not initialize tokenizer for {language}: {e}")
            # Fallback to basic english tokenizer or skip language-specific features
            try:
                self.tokenizer = Tokenizer('english')
                logging.info("Fallback to English tokenizer")
            except Exception as fallback_error:
                logging.error(f"Could not initialize any tokenizer: {fallback_error}")
                # Set tokenizer to None and handle this in methods that use it
                self.tokenizer = None
                # Remove summarization methods that require tokenizer
                self.available_methods = ['tfidf']  # TF-IDF doesn't need sumy tokenizer
        
    def _initialize_transformer(self):
        """Initialize transformer-based summarizer"""
        try:
            self.transformer_summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                tokenizer="facebook/bart-large-cnn"
            )
        except Exception as e:
            logging.warning(f"Could not load transformer summarization model: {e}")
            if 'transformer' in self.available_methods:
                self.available_methods.remove('transformer')
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better summarization
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Cleaned and preprocessed text
        """
        # Remove extra whitespaces and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', ' ', text)
        
        # Normalize spacing around punctuation
        text = re.sub(r'\s*([\.!\?])\s*', r'\1 ', text)
        
        return text
    
    def summarize_single_text(self, text: str, method: str = 'textrank', 
                            max_sentences: int = 3, max_words: int = None) -> SummaryResult:
        """
        Summarize a single text using specified method
        
        Args:
            text: Text to summarize
            method: Summarization method to use
            max_sentences: Maximum number of sentences in summary
            max_words: Maximum number of words in summary (for transformer)
            
        Returns:
            SummaryResult object with summarization results
        """
        if method not in self.available_methods:
            raise ValueError(f"Method '{method}' not available. Use one of: {self.available_methods}")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if method == 'transformer':
            return self._summarize_with_transformer(processed_text, max_words or 50)
        else:
            return self._summarize_extractive(processed_text, method, max_sentences)
    
    def _summarize_extractive(self, text: str, method: str, max_sentences: int) -> SummaryResult:
        """Perform extractive summarization using specified method"""
        
        # Check if tokenizer is available
        if self.tokenizer is None:
            logging.warning("Tokenizer not available, falling back to TF-IDF method")
            return self._summarize_with_tfidf(text, max_sentences)
        
        # Parse text
        parser = PlaintextParser.from_string(text, self.tokenizer)
        
        # Initialize summarizer based on method
        if method == 'textrank':
            summarizer = TextRankSummarizer()
        elif method == 'lsa':
            summarizer = LsaSummarizer()
        elif method == 'lexrank':
            summarizer = LexRankSummarizer()
        elif method == 'luhn':
            summarizer = LuhnSummarizer()
        elif method == 'tfidf':
            return self._summarize_with_tfidf(text, max_sentences)
        else:
            raise ValueError(f"Unknown extractive method: {method}")
        
        # Generate summary
        sentences = summarizer(parser.document, max_sentences)
        summary_sentences = [str(sentence) for sentence in sentences]
        summary = ' '.join(summary_sentences)
        
        # Calculate metrics
        original_words = len(text.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            method=method,
            compression_ratio=compression_ratio,
            key_sentences=summary_sentences,
            word_count_original=original_words,
            word_count_summary=summary_words
        )
    
    def _summarize_with_tfidf(self, text: str, max_sentences: int) -> SummaryResult:
        """Perform TF-IDF based extractive summarization"""
        
        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            summary = text
            key_sentences = sentences
        else:
            # Calculate TF-IDF scores
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate sentence scores (sum of TF-IDF scores)
            sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
            # Get top sentences
            top_indices = sentence_scores.argsort()[-max_sentences:][::-1]
            top_indices.sort()  # Maintain original order
            
            key_sentences = [sentences[i] for i in top_indices]
            summary = ' '.join(key_sentences)
        
        # Calculate metrics
        original_words = len(text.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            method='tfidf',
            compression_ratio=compression_ratio,
            key_sentences=key_sentences,
            word_count_original=original_words,
            word_count_summary=summary_words
        )
    
    def _summarize_with_transformer(self, text: str, max_length: int) -> SummaryResult:
        """Perform abstractive summarization using transformer model"""
        
        if not hasattr(self, 'transformer_summarizer'):
            raise ValueError("Transformer summarizer not available")
        
        # Truncate text if too long (BART has token limits)
        max_input_length = 1024
        words = text.split()
        if len(words) > max_input_length:
            text = ' '.join(words[:max_input_length])
        
        try:
            # Generate summary
            result = self.transformer_summarizer(
                text, 
                max_length=max_length, 
                min_length=max_length//3, 
                do_sample=False
            )
            summary = result[0]['summary_text']
            
        except Exception as e:
            logging.error(f"Error in transformer summarization: {e}")
            # Fallback to extractive summarization
            return self._summarize_extractive(text, 'textrank', 3)
        
        # Extract key sentences (approximate from original text)
        key_sentences = self._extract_key_sentences(text, summary, 3)
        
        # Calculate metrics
        original_words = len(text.split())
        summary_words = len(summary.split())
        compression_ratio = summary_words / original_words if original_words > 0 else 0
        
        return SummaryResult(
            original_text=text,
            summary=summary,
            method='transformer',
            compression_ratio=compression_ratio,
            key_sentences=key_sentences,
            word_count_original=original_words,
            word_count_summary=summary_words
        )
    
    def _extract_key_sentences(self, original_text: str, summary: str, num_sentences: int) -> List[str]:
        """Extract key sentences from original text that are most similar to summary"""
        
        sentences = nltk.sent_tokenize(original_text)
        if len(sentences) <= num_sentences:
            return sentences
            
        # Use TF-IDF to find most similar sentences to summary
        all_texts = sentences + [summary]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate similarity between each sentence and summary
        summary_vector = tfidf_matrix[-1]
        sentence_similarities = cosine_similarity(tfidf_matrix[:-1], summary_vector).flatten()
        
        # Get top sentences
        top_indices = sentence_similarities.argsort()[-num_sentences:][::-1]
        top_indices.sort()  # Maintain original order
        
        return [sentences[i] for i in top_indices]
    
    def summarize_multiple_texts(self, texts: List[str], method: str = 'textrank',
                                max_sentences: int = 3) -> List[SummaryResult]:
        """
        Summarize multiple texts
        
        Args:
            texts: List of texts to summarize
            method: Summarization method
            max_sentences: Maximum sentences per summary
            
        Returns:
            List of SummaryResult objects
        """
        return [self.summarize_single_text(text, method, max_sentences) for text in texts]
    
    def create_global_summary(self, texts: List[str], method: str = 'textrank',
                             max_sentences: int = 5) -> SummaryResult:
        """
        Create a global summary from multiple texts
        
        Args:
            texts: List of texts to create global summary from
            method: Summarization method
            max_sentences: Maximum sentences in global summary
            
        Returns:
            SummaryResult with global summary
        """
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create summary
        return self.summarize_single_text(combined_text, method, max_sentences)
    
    def get_summary_statistics(self, results: List[SummaryResult]) -> Dict[str, float]:
        """
        Get statistics from multiple summary results
        
        Args:
            results: List of summary results
            
        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}
        
        compression_ratios = [r.compression_ratio for r in results]
        original_lengths = [r.word_count_original for r in results]
        summary_lengths = [r.word_count_summary for r in results]
        
        stats = {
            'avg_compression_ratio': np.mean(compression_ratios),
            'avg_original_length': np.mean(original_lengths),
            'avg_summary_length': np.mean(summary_lengths),
            'total_original_words': sum(original_lengths),
            'total_summary_words': sum(summary_lengths),
            'total_compression_ratio': sum(summary_lengths) / sum(original_lengths) if sum(original_lengths) > 0 else 0
        }
        
        return stats

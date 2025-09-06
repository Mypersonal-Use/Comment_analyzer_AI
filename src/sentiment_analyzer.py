"""
Sentiment Analysis Module for eConsultation AI
Supports multiple sentiment analysis approaches including VADER, TextBlob, and Transformer models
"""

import logging
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np

# Core NLP libraries
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Optional transformer support
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers not available. Some features will be limited.")

@dataclass
class SentimentResult:
    """Data class to store sentiment analysis results"""
    text: str
    overall_sentiment: str  # 'positive', 'negative', 'neutral'
    confidence: float
    scores: Dict[str, float]
    method: str

class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis class supporting multiple methods
    """
    
    def __init__(self, methods: List[str] = None):
        """
        Initialize the sentiment analyzer
        
        Args:
            methods: List of methods to use ['vader', 'textblob', 'transformer']
        """
        self.available_methods = ['vader', 'textblob']
        if TRANSFORMERS_AVAILABLE:
            self.available_methods.append('transformer')
            
        if methods is None:
            self.methods = ['vader', 'textblob']
        else:
            self.methods = [m for m in methods if m in self.available_methods]
            
        if not self.methods:
            raise ValueError(f"No valid methods specified. Available: {self.available_methods}")
            
        # Initialize analyzers
        self._initialize_analyzers()
        
    def _initialize_analyzers(self):
        """Initialize the sentiment analysis models"""
        self.analyzers = {}
        
        # Initialize VADER
        if 'vader' in self.methods:
            self.analyzers['vader'] = SentimentIntensityAnalyzer()
            
        # Initialize TextBlob (no explicit initialization needed)
        if 'textblob' in self.methods:
            self.analyzers['textblob'] = True
            
        # Initialize Transformer model
        if 'transformer' in self.methods and TRANSFORMERS_AVAILABLE:
            try:
                # Using a pre-trained model for sentiment analysis
                self.analyzers['transformer'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
            except Exception as e:
                logging.warning(f"Could not load transformer model: {e}")
                self.methods.remove('transformer')
                
    def analyze_single_comment(self, text: str, method: str = 'vader') -> SentimentResult:
        """
        Analyze sentiment of a single comment
        
        Args:
            text: The text to analyze
            method: The method to use ('vader', 'textblob', 'transformer')
            
        Returns:
            SentimentResult object with analysis results
        """
        if method not in self.methods:
            raise ValueError(f"Method '{method}' not available. Use one of: {self.methods}")
            
        if method == 'vader':
            return self._analyze_with_vader(text)
        elif method == 'textblob':
            return self._analyze_with_textblob(text)
        elif method == 'transformer':
            return self._analyze_with_transformer(text)
            
    def _analyze_with_vader(self, text: str) -> SentimentResult:
        """Analyze sentiment using VADER"""
        scores = self.analyzers['vader'].polarity_scores(text)
        
        # Determine overall sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
            confidence = scores['pos']
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
            confidence = scores['neg']
        else:
            sentiment = 'neutral'
            confidence = scores['neu']
            
        return SentimentResult(
            text=text,
            overall_sentiment=sentiment,
            confidence=confidence,
            scores=scores,
            method='vader'
        )
        
    def _analyze_with_textblob(self, text: str) -> SentimentResult:
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Convert polarity to sentiment categories
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        scores = {
            'polarity': polarity,
            'subjectivity': subjectivity,
            'positive': max(0, polarity),
            'negative': abs(min(0, polarity)),
            'neutral': 1 - abs(polarity)
        }
        
        confidence = abs(polarity)
        
        return SentimentResult(
            text=text,
            overall_sentiment=sentiment,
            confidence=confidence,
            scores=scores,
            method='textblob'
        )
        
    def _analyze_with_transformer(self, text: str) -> SentimentResult:
        """Analyze sentiment using Transformer model"""
        if 'transformer' not in self.analyzers:
            raise ValueError("Transformer model not available")
            
        results = self.analyzers['transformer'](text)[0]
        
        # Convert results to standard format
        scores = {}
        max_score = 0
        predicted_sentiment = 'neutral'
        
        for result in results:
            label = result['label'].lower()
            score = result['score']
            
            if 'positive' in label or 'pos' in label:
                scores['positive'] = score
                if score > max_score:
                    max_score = score
                    predicted_sentiment = 'positive'
            elif 'negative' in label or 'neg' in label:
                scores['negative'] = score
                if score > max_score:
                    max_score = score
                    predicted_sentiment = 'negative'
            else:
                scores['neutral'] = score
                if score > max_score:
                    max_score = score
                    predicted_sentiment = 'neutral'
                    
        return SentimentResult(
            text=text,
            overall_sentiment=predicted_sentiment,
            confidence=max_score,
            scores=scores,
            method='transformer'
        )
        
    def analyze_batch(self, texts: List[str], method: str = 'vader') -> List[SentimentResult]:
        """
        Analyze sentiment for a batch of texts
        
        Args:
            texts: List of texts to analyze
            method: The method to use
            
        Returns:
            List of SentimentResult objects
        """
        return [self.analyze_single_comment(text, method) for text in texts]
        
    def get_ensemble_prediction(self, text: str) -> SentimentResult:
        """
        Get ensemble prediction using all available methods
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with ensemble prediction
        """
        results = []
        for method in self.methods:
            try:
                result = self.analyze_single_comment(text, method)
                results.append(result)
            except Exception as e:
                logging.warning(f"Error with method {method}: {e}")
                
        if not results:
            raise ValueError("No methods produced valid results")
            
        # Combine results using weighted voting
        sentiment_votes = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_confidence = 0
        all_scores = {}
        
        for result in results:
            weight = result.confidence
            sentiment_votes[result.overall_sentiment] += weight
            total_confidence += weight
            
            # Aggregate scores
            for key, value in result.scores.items():
                if key not in all_scores:
                    all_scores[key] = []
                all_scores[key].append(value)
                
        # Determine final sentiment
        final_sentiment = max(sentiment_votes.keys(), key=sentiment_votes.get)
        final_confidence = sentiment_votes[final_sentiment] / total_confidence if total_confidence > 0 else 0
        
        # Average scores
        averaged_scores = {key: np.mean(values) for key, values in all_scores.items()}
        
        return SentimentResult(
            text=text,
            overall_sentiment=final_sentiment,
            confidence=final_confidence,
            scores=averaged_scores,
            method='ensemble'
        )
        
    def get_sentiment_distribution(self, results: List[SentimentResult]) -> Dict[str, float]:
        """
        Get distribution of sentiments from a list of results
        
        Args:
            results: List of sentiment analysis results
            
        Returns:
            Dictionary with sentiment distribution
        """
        total = len(results)
        if total == 0:
            return {'positive': 0, 'negative': 0, 'neutral': 0}
            
        distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for result in results:
            distribution[result.overall_sentiment] += 1
            
        # Convert to percentages
        for key in distribution:
            distribution[key] = (distribution[key] / total) * 100
            
        return distribution

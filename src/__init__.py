"""
eConsultation AI Package

AI-powered sentiment analysis and comment processing for eConsultation module
"""

from .econsultation_ai import EConsultationAI, AnalysisResult, BatchAnalysisResult
from .sentiment_analyzer import SentimentAnalyzer, SentimentResult
from .text_summarizer import TextSummarizer, SummaryResult
from .wordcloud_generator import WordCloudGenerator, WordCloudResult
from .data_processor import DataProcessor, CommentData, ProcessingStats
from .visualization_reporter import VisualizationReporter

__version__ = "1.0.0"
__author__ = "MCA eConsultation Team"
__email__ = "support@mca.gov.in"

__all__ = [
    # Main classes
    'EConsultationAI',
    'SentimentAnalyzer', 
    'TextSummarizer',
    'WordCloudGenerator',
    'DataProcessor',
    'VisualizationReporter',
    
    # Data classes
    'AnalysisResult',
    'BatchAnalysisResult', 
    'SentimentResult',
    'SummaryResult',
    'WordCloudResult',
    'CommentData',
    'ProcessingStats'
]

"""
eConsultation AI - Main Application Interface
Integrates sentiment analysis, summarization, and word cloud generation
"""

import logging
import time
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json

# Import custom modules
from sentiment_analyzer import SentimentAnalyzer, SentimentResult
from text_summarizer import TextSummarizer, SummaryResult
from wordcloud_generator import WordCloudGenerator, WordCloudResult
from data_processor import DataProcessor, CommentData

@dataclass
class AnalysisResult:
    """Comprehensive analysis result for a single comment or batch"""
    comment_id: str
    original_text: str
    sentiment: SentimentResult
    summary: SummaryResult
    word_analysis: Optional[WordCloudResult] = None
    processing_time: float = 0.0

@dataclass
class BatchAnalysisResult:
    """Results for batch processing of multiple comments"""
    total_comments: int
    results: List[AnalysisResult]
    sentiment_distribution: Dict[str, float]
    global_summary: SummaryResult
    global_wordcloud: WordCloudResult
    processing_stats: Dict[str, Any]
    total_processing_time: float

class EConsultationAI:
    """
    Main application class integrating all AI components for eConsultation analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the eConsultation AI system
        
        Args:
            config: Configuration dictionary with component settings
        """
        self.config = config or {}
        
        # Setup logging EARLY (before initializing components that may log)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all AI components"""
        
        # Sentiment Analysis
        sentiment_methods = self.config.get('sentiment_methods', ['vader', 'textblob'])
        self.sentiment_analyzer = SentimentAnalyzer(methods=sentiment_methods)
        
        # Text Summarization
        summarization_language = self.config.get('summarization_language', 'english')
        self.text_summarizer = TextSummarizer(language=summarization_language)
        
        # Word Cloud Generation
        wordcloud_language = self.config.get('wordcloud_language', 'english')
        custom_stopwords = self.config.get('custom_stopwords', [])
        self.wordcloud_generator = WordCloudGenerator(
            language=wordcloud_language, 
            custom_stopwords=custom_stopwords
        )
        
        # Data Processing
        encoding = self.config.get('encoding', 'utf-8')
        self.data_processor = DataProcessor(encoding=encoding)
        
        self.logger.info("eConsultation AI components initialized successfully")
    
    def analyze_single_comment(self, text: str, comment_id: str = None,
                              include_word_analysis: bool = False) -> AnalysisResult:
        """
        Perform comprehensive analysis on a single comment
        
        Args:
            text: Comment text to analyze
            comment_id: Optional ID for the comment
            include_word_analysis: Whether to include word cloud analysis
            
        Returns:
            AnalysisResult with comprehensive analysis
        """
        start_time = time.time()
        
        if not comment_id:
            comment_id = f"comment_{int(time.time())}"
        
        try:
            # Sentiment Analysis
            sentiment_method = self.config.get('default_sentiment_method', 'vader')
            sentiment_result = self.sentiment_analyzer.analyze_single_comment(
                text, method=sentiment_method
            )
            
            # Text Summarization
            summarization_method = self.config.get('default_summarization_method', 'textrank')
            max_sentences = self.config.get('max_summary_sentences', 2)
            summary_result = self.text_summarizer.summarize_single_text(
                text, method=summarization_method, max_sentences=max_sentences
            )
            
            # Word Analysis (optional)
            word_result = None
            if include_word_analysis:
                word_result = self.wordcloud_generator.extract_keywords(text, top_n=20)
            
            processing_time = time.time() - start_time
            
            result = AnalysisResult(
                comment_id=comment_id,
                original_text=text,
                sentiment=sentiment_result,
                summary=summary_result,
                word_analysis=word_result,
                processing_time=processing_time
            )
            
            self.logger.info(f"Analyzed comment {comment_id} in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing comment {comment_id}: {e}")
            raise
    
    def analyze_batch(self, comments: List[Union[str, CommentData]], 
                     output_dir: str = None, 
                     generate_reports: bool = True) -> BatchAnalysisResult:
        """
        Perform batch analysis on multiple comments
        
        Args:
            comments: List of comment strings or CommentData objects
            output_dir: Optional directory to save results
            generate_reports: Whether to generate visual reports
            
        Returns:
            BatchAnalysisResult with comprehensive batch analysis
        """
        start_time = time.time()
        
        # Convert to CommentData objects if necessary
        comment_data = []
        for i, comment in enumerate(comments):
            if isinstance(comment, str):
                comment_data.append(CommentData(id=str(i), text=comment))
            else:
                comment_data.append(comment)
        
        # Process individual comments
        individual_results = []
        for comment in comment_data:
            try:
                result = self.analyze_single_comment(
                    comment.text, 
                    comment.id, 
                    include_word_analysis=False
                )
                individual_results.append(result)
            except Exception as e:
                self.logger.warning(f"Failed to process comment {comment.id}: {e}")
                continue
        
        if not individual_results:
            raise ValueError("No comments were successfully processed")
        
        # Aggregate analysis
        all_texts = [comment.text for comment in comment_data]
        
        # Sentiment distribution
        sentiment_results = [result.sentiment for result in individual_results]
        sentiment_distribution = self.sentiment_analyzer.get_sentiment_distribution(sentiment_results)
        
        # Global summary
        global_summary = self.text_summarizer.create_global_summary(
            all_texts, 
            method=self.config.get('default_summarization_method', 'textrank'),
            max_sentences=self.config.get('max_global_summary_sentences', 5)
        )
        
        # Global word cloud
        global_wordcloud = self.wordcloud_generator.extract_keywords(
            all_texts, 
            top_n=self.config.get('max_keywords', 50)
        )
        
        # Processing statistics
        processing_stats = {
            'total_comments_processed': len(individual_results),
            'total_comments_input': len(comments),
            'success_rate': len(individual_results) / len(comments) * 100,
            'avg_sentiment_confidence': sum(r.sentiment.confidence for r in individual_results) / len(individual_results),
            'avg_summary_compression': sum(r.summary.compression_ratio for r in individual_results) / len(individual_results),
            'total_words_analyzed': sum(r.word_analysis.total_words for r in individual_results if r.word_analysis),
        }
        
        total_processing_time = time.time() - start_time
        
        # Create batch result
        batch_result = BatchAnalysisResult(
            total_comments=len(comments),
            results=individual_results,
            sentiment_distribution=sentiment_distribution,
            global_summary=global_summary,
            global_wordcloud=global_wordcloud,
            processing_stats=processing_stats,
            total_processing_time=total_processing_time
        )
        
        # Generate reports if requested
        if generate_reports and output_dir:
            self._generate_batch_reports(batch_result, output_dir)
        
        self.logger.info(f"Batch analysis completed: {len(individual_results)}/{len(comments)} comments processed in {total_processing_time:.2f}s")
        
        return batch_result
    
    def analyze_from_file(self, file_path: str, output_dir: str = None) -> BatchAnalysisResult:
        """
        Analyze comments from a file
        
        Args:
            file_path: Path to input file (CSV, JSON, TXT, Excel)
            output_dir: Optional directory to save results
            
        Returns:
            BatchAnalysisResult with analysis results
        """
        try:
            # Load data
            comments = self.data_processor.load_data(file_path)
            
            # Filter and clean comments
            filtered_comments = self.data_processor.filter_comments(
                comments,
                min_length=self.config.get('min_comment_length', 10),
                max_length=self.config.get('max_comment_length'),
                remove_duplicates=self.config.get('remove_duplicates', True)
            )
            
            self.logger.info(f"Loaded {len(filtered_comments)} comments from {file_path}")
            
            # Perform analysis
            return self.analyze_batch(filtered_comments, output_dir)
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {file_path}: {e}")
            raise
    
    def _generate_batch_reports(self, batch_result: BatchAnalysisResult, output_dir: str):
        """Generate comprehensive reports for batch analysis"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 1. Generate word cloud
            wordcloud_path = output_path / "stakeholder_wordcloud.png"
            all_texts = [result.original_text for result in batch_result.results]
            
            # Use themed wordcloud generation instead of passing theme as parameter
            theme = self.config.get('wordcloud_theme', 'government')
            self.wordcloud_generator.generate_themed_wordcloud(
                all_texts, 
                theme=theme,
                save_path=str(wordcloud_path)
            )
            
            # 2. Generate frequency chart
            freq_chart_path = output_path / "keyword_frequency_chart.png"
            self.wordcloud_generator.generate_frequency_chart(
                batch_result.global_wordcloud,
                top_n=20,
                save_path=str(freq_chart_path)
            )
            
            # 3. Export keywords to CSV
            keywords_csv_path = output_path / "top_keywords.csv"
            self.wordcloud_generator.export_keywords_to_csv(
                batch_result.global_wordcloud,
                str(keywords_csv_path)
            )
            
            # 4. Save detailed results to JSON
            results_json_path = output_path / "analysis_results.json"
            self._save_results_to_json(batch_result, str(results_json_path))
            
            # 5. Generate summary report
            summary_report_path = output_path / "analysis_summary.txt"
            self._generate_summary_report(batch_result, str(summary_report_path))
            
            self.logger.info(f"Reports generated successfully in {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
            raise
    
    def _save_results_to_json(self, batch_result: BatchAnalysisResult, file_path: str):
        """Save batch results to JSON file"""
        
        # Convert results to serializable format
        json_data = {
            'total_comments': batch_result.total_comments,
            'processing_stats': batch_result.processing_stats,
            'total_processing_time': batch_result.total_processing_time,
            'sentiment_distribution': batch_result.sentiment_distribution,
            'global_summary': {
                'summary_text': batch_result.global_summary.summary,
                'method': batch_result.global_summary.method,
                'compression_ratio': batch_result.global_summary.compression_ratio,
                'word_count_original': batch_result.global_summary.word_count_original,
                'word_count_summary': batch_result.global_summary.word_count_summary
            },
            'global_keywords': {
                'top_keywords': batch_result.global_wordcloud.top_keywords[:20],
                'total_words': batch_result.global_wordcloud.total_words,
                'unique_words': batch_result.global_wordcloud.unique_words
            },
            'individual_results': []
        }
        
        # Add individual results
        for result in batch_result.results:
            individual_data = {
                'comment_id': result.comment_id,
                'original_text': result.original_text,
                'processing_time': result.processing_time,
                'sentiment': {
                    'overall_sentiment': result.sentiment.overall_sentiment,
                    'confidence': result.sentiment.confidence,
                    'method': result.sentiment.method
                },
                'summary': {
                    'summary_text': result.summary.summary,
                    'method': result.summary.method,
                    'compression_ratio': result.summary.compression_ratio
                }
            }
            json_data['individual_results'].append(individual_data)
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    def _generate_summary_report(self, batch_result: BatchAnalysisResult, file_path: str):
        """Generate a human-readable summary report"""
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("eConsultation AI - Analysis Summary Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overview
            f.write("OVERVIEW\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Comments Analyzed: {batch_result.total_comments}\n")
            f.write(f"Processing Time: {batch_result.total_processing_time:.2f} seconds\n")
            f.write(f"Success Rate: {batch_result.processing_stats['success_rate']:.1f}%\n\n")
            
            # Sentiment Analysis
            f.write("SENTIMENT ANALYSIS\n")
            f.write("-" * 30 + "\n")
            for sentiment, percentage in batch_result.sentiment_distribution.items():
                f.write(f"{sentiment.capitalize()}: {percentage:.1f}%\n")
            f.write(f"\nAverage Confidence: {batch_result.processing_stats['avg_sentiment_confidence']:.3f}\n\n")
            
            # Summary
            f.write("GLOBAL SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Method: {batch_result.global_summary.method}\n")
            f.write(f"Compression Ratio: {batch_result.global_summary.compression_ratio:.3f}\n")
            f.write("Summary Text:\n")
            f.write(batch_result.global_summary.summary + "\n\n")
            
            # Keywords
            f.write("TOP KEYWORDS\n")
            f.write("-" * 20 + "\n")
            for i, (word, freq) in enumerate(batch_result.global_wordcloud.top_keywords[:10], 1):
                f.write(f"{i:2d}. {word} ({freq} occurrences)\n")
            f.write("\n")
            
            # Processing Statistics
            f.write("PROCESSING STATISTICS\n")
            f.write("-" * 30 + "\n")
            for key, value in batch_result.processing_stats.items():
                if isinstance(value, float):
                    f.write(f"{key.replace('_', ' ').title()}: {value:.3f}\n")
                else:
                    f.write(f"{key.replace('_', ' ').title()}: {value}\n")
    
    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.copy()
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update configuration and reinitialize components if necessary"""
        self.config.update(new_config)
        self._initialize_components()
        self.logger.info("Configuration updated and components reinitialized")
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported input file formats"""
        return self.data_processor.supported_formats
    
    def validate_input_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate input file and provide information about its structure
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Dictionary with file information and validation results
        """
        try:
            comments = self.data_processor.load_data(file_path)
            
            validation_result = {
                'valid': True,
                'total_comments': len(comments),
                'sample_comments': [comment.text[:100] + "..." for comment in comments[:3]],
                'file_format': Path(file_path).suffix.lower(),
                'estimated_processing_time': len(comments) * 0.1,  # Rough estimate
                'warnings': []
            }
            
            # Add warnings
            if len(comments) == 0:
                validation_result['warnings'].append("No valid comments found in file")
            elif len(comments) > 1000:
                validation_result['warnings'].append("Large number of comments - processing may take significant time")
                
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'file_format': Path(file_path).suffix.lower()
            }

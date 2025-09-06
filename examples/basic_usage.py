#!/usr/bin/env python3
"""
Basic Usage Example for eConsultation AI

This script demonstrates how to use the eConsultation AI system for
analyzing stakeholder comments with sentiment analysis, summarization,
and word cloud generation.
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import the modules directly
from sentiment_analyzer import SentimentAnalyzer, SentimentResult
from text_summarizer import TextSummarizer, SummaryResult
from wordcloud_generator import WordCloudGenerator, WordCloudResult
from data_processor import DataProcessor, CommentData
from econsultation_ai import EConsultationAI

def main():
    """Demonstrate basic usage of eConsultation AI"""
    
    print("=" * 60)
    print("eConsultation AI - Basic Usage Example")
    print("=" * 60)
    
    # Initialize the AI system with custom configuration
    config = {
        'sentiment_methods': ['vader', 'textblob'],
        'default_sentiment_method': 'vader',
        'default_summarization_method': 'textrank',
        'max_summary_sentences': 2,
        'max_global_summary_sentences': 4,
        'wordcloud_language': 'english',
        'custom_stopwords': ['amendment', 'provision', 'section', 'clause'],
        'min_comment_length': 20,
        'remove_duplicates': True
    }
    
    try:
        # Initialize the system
        print("\\n1. Initializing eConsultation AI system...")
        ai_system = EConsultationAI(config=config)
        print("✓ System initialized successfully")
        
        # Example 1: Analyze a single comment
        print("\\n2. Analyzing a single comment...")
        
        sample_comment = (
            "I strongly support the proposed amendments to the corporate governance framework. "
            "These changes will significantly enhance transparency and accountability in our "
            "business environment. The mandatory disclosure requirements are particularly welcome "
            "and will help build investor confidence."
        )
        
        result = ai_system.analyze_single_comment(
            text=sample_comment,
            comment_id="example_001",
            include_word_analysis=True
        )
        
        print(f"\\nSingle Comment Analysis Results:")
        print(f"- Sentiment: {result.sentiment.overall_sentiment} (confidence: {result.sentiment.confidence:.3f})")
        print(f"- Summary: {result.summary.summary}")
        print(f"- Processing Time: {result.processing_time:.3f} seconds")
        print(f"- Top Keywords: {result.word_analysis.top_keywords[:5] if result.word_analysis else 'N/A'}")
        
        # Example 2: Analyze multiple comments
        print("\\n3. Analyzing multiple comments...")
        
        sample_comments = [
            "I fully endorse these amendments. They align perfectly with international best practices.",
            "While I appreciate the intent, I have concerns about the implementation timeline being too short.",
            "The proposed penalties appear excessive and may impact smaller enterprises disproportionately.",
            "These amendments are long overdue and will ensure better governance standards.",
            "The consultation process has been thorough, but some clauses need clarification."
        ]
        
        batch_result = ai_system.analyze_batch(
            comments=sample_comments,
            output_dir="results/basic_example",
            generate_reports=True
        )
        
        print(f"\\nBatch Analysis Results:")
        print(f"- Total Comments: {batch_result.total_comments}")
        print(f"- Processing Time: {batch_result.total_processing_time:.2f} seconds")
        print(f"- Success Rate: {batch_result.processing_stats['success_rate']:.1f}%")
        print(f"- Sentiment Distribution:")
        for sentiment, percentage in batch_result.sentiment_distribution.items():
            print(f"  * {sentiment.capitalize()}: {percentage:.1f}%")
        
        print(f"\\n- Global Summary:")
        print(f"  {batch_result.global_summary.summary}")
        
        print(f"\\n- Top 5 Keywords:")
        for i, (word, freq) in enumerate(batch_result.global_wordcloud.top_keywords[:5], 1):
            print(f"  {i}. {word} ({freq} occurrences)")
        
        # Example 3: Analyze from file
        print("\\n4. Analyzing comments from CSV file...")
        
        # Check if sample data file exists
        data_file = Path(__file__).parent.parent / 'data' / 'sample_comments.csv'
        if data_file.exists():
            file_result = ai_system.analyze_from_file(
                file_path=str(data_file),
                output_dir="results/file_analysis"
            )
            
            print(f"\\nFile Analysis Results:")
            print(f"- Comments Processed: {len(file_result.results)}")
            print(f"- Total Processing Time: {file_result.total_processing_time:.2f} seconds")
            print(f"- Average Processing Time per Comment: {file_result.total_processing_time/len(file_result.results):.3f} seconds")
            
            print(f"\\n- Sentiment Distribution:")
            for sentiment, percentage in file_result.sentiment_distribution.items():
                print(f"  * {sentiment.capitalize()}: {percentage:.1f}%")
            
            print(f"\\n- Reports Generated in: results/file_analysis/")
            print("  * Word cloud visualization")
            print("  * Keyword frequency chart") 
            print("  * Analysis summary report")
            print("  * Detailed results (JSON)")
            
        else:
            print("Sample data file not found. Skipping file analysis example.")
        
        # Example 4: Configuration management
        print("\\n5. Configuration management example...")
        
        current_config = ai_system.get_configuration()
        print(f"Current configuration has {len(current_config)} settings")
        
        # Update configuration
        new_settings = {
            'max_summary_sentences': 3,
            'wordcloud_theme': 'professional'
        }
        
        ai_system.update_configuration(new_settings)
        print("✓ Configuration updated successfully")
        
        # Show supported formats
        print(f"\\nSupported file formats: {ai_system.get_supported_formats()}")
        
        print("\\n" + "=" * 60)
        print("Basic usage demonstration completed successfully!")
        print("Check the 'results/' directory for generated reports.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\\nError during execution: {e}")
        print("Please ensure all dependencies are installed and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

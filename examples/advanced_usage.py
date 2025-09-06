#!/usr/bin/env python3
"""
Advanced Usage Example for eConsultation AI

This script demonstrates advanced features including:
- Custom configuration management
- Batch file processing
- Multiple analysis methods comparison
- Custom visualization generation
- Performance monitoring
"""

import sys
import os
import time
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import the modules directly
from sentiment_analyzer import SentimentAnalyzer, SentimentResult
from text_summarizer import TextSummarizer, SummaryResult
from wordcloud_generator import WordCloudGenerator, WordCloudResult
from data_processor import DataProcessor, CommentData
from econsultation_ai import EConsultationAI
from visualization_reporter import VisualizationReporter

def compare_sentiment_methods():
    """Compare different sentiment analysis methods"""
    
    print("\\n" + "="*50)
    print("SENTIMENT ANALYSIS METHOD COMPARISON")
    print("="*50)
    
    test_comments = [
        "I strongly support these excellent amendments and commend the comprehensive approach.",
        "These proposed changes are terrible and will destroy business efficiency completely.",
        "The amendments are reasonable but need some minor adjustments for better implementation.",
        "While I appreciate the intent, the execution seems flawed and potentially problematic.",
        "This is a balanced approach that addresses most stakeholder concerns effectively."
    ]
    
    methods = ['vader', 'textblob']
    results = {}
    
    for method in methods:
        print(f"\\nTesting {method.upper()} method...")
        
        config = {
            'sentiment_methods': [method],
            'default_sentiment_method': method
        }
        
        ai_system = EConsultationAI(config=config)
        
        method_results = []
        start_time = time.time()
        
        for i, comment in enumerate(test_comments, 1):
            result = ai_system.analyze_single_comment(comment, f"test_{i}")
            method_results.append({
                'comment': comment[:50] + "...",
                'sentiment': result.sentiment.overall_sentiment,
                'confidence': result.sentiment.confidence,
                'processing_time': result.processing_time
            })
        
        total_time = time.time() - start_time
        results[method] = {
            'results': method_results,
            'total_time': total_time,
            'avg_time': total_time / len(test_comments)
        }
        
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Average time per comment: {total_time/len(test_comments):.3f}s")
    
    # Display comparison
    print(f"\\nMETHOD COMPARISON RESULTS:")
    print("-" * 40)
    
    for i, comment in enumerate(test_comments):
        print(f"\\nComment {i+1}: {comment[:60]}...")
        for method in methods:
            result = results[method]['results'][i]
            print(f"  {method.upper():10}: {result['sentiment']:8} (conf: {result['confidence']:.3f})")
    
    return results

def advanced_batch_processing():
    """Demonstrate advanced batch processing with custom configuration"""
    
    print("\\n" + "="*50)
    print("ADVANCED BATCH PROCESSING")
    print("="*50)
    
    # Advanced configuration
    advanced_config = {
        'sentiment_methods': ['vader', 'textblob'],
        'default_sentiment_method': 'vader',
        'default_summarization_method': 'textrank',
        'max_summary_sentences': 3,
        'max_global_summary_sentences': 6,
        'wordcloud_language': 'english',
        'custom_stopwords': [
            'amendment', 'provision', 'section', 'clause', 'proposed',
            'legislation', 'regulation', 'framework', 'requirement'
        ],
        'min_comment_length': 30,
        'max_comment_length': 2000,
        'remove_duplicates': True,
        'wordcloud_theme': 'government'
    }
    
    ai_system = EConsultationAI(config=advanced_config)
    
    # Check for sample data
    data_file = Path(__file__).parent.parent / 'data' / 'sample_comments.csv'
    
    if not data_file.exists():
        print("Sample data file not found. Creating synthetic data...")
        return create_synthetic_analysis()
    
    print(f"\\n1. Validating input file...")
    validation_result = ai_system.validate_input_file(str(data_file))
    
    if validation_result['valid']:
        print(f"✓ File validation successful:")
        print(f"  - Format: {validation_result['file_format']}")
        print(f"  - Comments: {validation_result['total_comments']}")
        print(f"  - Estimated time: {validation_result['estimated_processing_time']:.1f}s")
        
        if validation_result['warnings']:
            print("  - Warnings:")
            for warning in validation_result['warnings']:
                print(f"    * {warning}")
    else:
        print(f"✗ File validation failed: {validation_result['error']}")
        return
    
    print(f"\\n2. Processing file with advanced configuration...")
    
    start_time = time.time()
    batch_result = ai_system.analyze_from_file(
        file_path=str(data_file),
        output_dir="results/advanced_analysis"
    )
    processing_time = time.time() - start_time
    
    print(f"\\n3. Analysis completed in {processing_time:.2f} seconds")
    
    # Display detailed results
    print(f"\\nDETAILED ANALYSIS RESULTS:")
    print("-" * 40)
    print(f"Total Comments Processed: {len(batch_result.results)}")
    print(f"Success Rate: {batch_result.processing_stats['success_rate']:.1f}%")
    print(f"Average Processing Time: {batch_result.total_processing_time/len(batch_result.results):.3f}s per comment")
    
    print(f"\\nSentiment Distribution:")
    for sentiment, percentage in sorted(batch_result.sentiment_distribution.items()):
        bar_length = int(percentage / 2)  # Scale for display
        bar = "█" * bar_length
        print(f"  {sentiment.capitalize():8}: {percentage:5.1f}% {bar}")
    
    print(f"\\nTop Keywords (Global):")
    for i, (word, freq) in enumerate(batch_result.global_wordcloud.top_keywords[:10], 1):
        print(f"  {i:2d}. {word:15} ({freq:2d} occurrences)")
    
    print(f"\\nGlobal Summary ({batch_result.global_summary.method}):")
    print(f"  Compression Ratio: {batch_result.global_summary.compression_ratio:.3f}")
    print(f"  Summary: {batch_result.global_summary.summary}")
    
    return batch_result

def create_synthetic_analysis():
    """Create analysis with synthetic data when sample file is not available"""
    
    print("Creating synthetic stakeholder comments for demonstration...")
    
    synthetic_comments = [
        "The proposed regulatory framework represents a significant improvement in corporate governance standards and will enhance investor confidence.",
        "While I appreciate the comprehensive nature of these amendments, the implementation timeline appears unrealistic for smaller organizations.",
        "These changes are absolutely necessary and long overdue. The enhanced transparency requirements will benefit all stakeholders.",
        "I have serious concerns about the compliance costs associated with these amendments, particularly for emerging companies.",
        "The consultation process has been thorough and the final amendments reflect a balanced approach to regulatory reform.",
        "The proposed penalties seem excessive and may discourage legitimate business activities and innovation.",
        "This is exactly what the industry needs - clear guidelines and robust enforcement mechanisms for better governance.",
        "The amendments fail to address the unique challenges faced by startups and may stifle entrepreneurial growth.",
        "I strongly support the digital filing requirements which will modernize the regulatory compliance process.",
        "The retrospective application of certain provisions creates legal uncertainty and should be reconsidered."
    ]
    
    config = {
        'default_sentiment_method': 'vader',
        'default_summarization_method': 'textrank',
        'max_global_summary_sentences': 4,
        'wordcloud_theme': 'government'
    }
    
    ai_system = EConsultationAI(config=config)
    
    batch_result = ai_system.analyze_batch(
        comments=synthetic_comments,
        output_dir="results/synthetic_analysis",
        generate_reports=True
    )
    
    print(f"\\nSynthetic Analysis Results:")
    print(f"- Comments: {len(batch_result.results)}")
    print(f"- Processing Time: {batch_result.total_processing_time:.2f}s")
    print(f"- Sentiment Distribution: {batch_result.sentiment_distribution}")
    
    return batch_result

def custom_visualization_demo(batch_result):
    """Demonstrate custom visualization generation"""
    
    print("\\n" + "="*50)
    print("CUSTOM VISUALIZATION GENERATION")
    print("="*50)
    
    # Initialize custom visualization reporter
    viz_reporter = VisualizationReporter(style='government')
    
    output_dir = "results/custom_visualizations"
    
    print(f"\\n1. Generating comprehensive visualization report...")
    
    try:
        reports = viz_reporter.generate_comprehensive_report(
            batch_result=batch_result,
            output_dir=output_dir
        )
        
        print(f"\\n✓ Custom visualizations generated:")
        for report_type, file_path in reports.items():
            if file_path:
                print(f"  - {report_type.replace('_', ' ').title()}: {Path(file_path).name}")
        
        print(f"\\nAll visualizations saved to: {output_dir}/")
        
    except Exception as e:
        print(f"\\n✗ Error generating custom visualizations: {e}")

def performance_monitoring():
    """Monitor and display performance metrics"""
    
    print("\\n" + "="*50)
    print("PERFORMANCE MONITORING")
    print("="*50)
    
    # Test with different comment sizes
    test_sizes = [1, 5, 10, 20]
    performance_data = []
    
    base_comment = (
        "The proposed amendments to the corporate governance framework represent "
        "a significant step forward in regulatory modernization. These changes "
        "will enhance transparency, improve accountability, and strengthen "
        "stakeholder protection mechanisms."
    )
    
    config = {'default_sentiment_method': 'vader'}
    ai_system = EConsultationAI(config=config)
    
    for size in test_sizes:
        print(f"\\nTesting with {size} comment(s)...")
        
        comments = [f"{base_comment} Comment {i+1} variant." for i in range(size)]
        
        start_time = time.time()
        result = ai_system.analyze_batch(comments, generate_reports=False)
        total_time = time.time() - start_time
        
        perf_data = {
            'comment_count': size,
            'total_time': total_time,
            'avg_time_per_comment': total_time / size,
            'success_rate': result.processing_stats['success_rate']
        }
        
        performance_data.append(perf_data)
        
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Avg per comment: {total_time/size:.3f}s")
        print(f"  Success rate: {perf_data['success_rate']:.1f}%")
    
    # Display performance summary
    print(f"\\nPERFORMANCE SUMMARY:")
    print("-" * 40)
    print(f"{'Comments':>8} {'Total Time':>12} {'Avg/Comment':>12} {'Success':>8}")
    print("-" * 40)
    
    for data in performance_data:
        print(f"{data['comment_count']:>8} {data['total_time']:>10.3f}s "
              f"{data['avg_time_per_comment']:>10.3f}s {data['success_rate']:>6.1f}%")

def main():
    """Run advanced usage demonstration"""
    
    print("="*60)
    print("eConsultation AI - Advanced Usage Example")
    print("="*60)
    
    try:
        # 1. Compare sentiment analysis methods
        comparison_results = compare_sentiment_methods()
        
        # 2. Advanced batch processing
        batch_result = advanced_batch_processing()
        
        # 3. Custom visualization generation
        if batch_result:
            custom_visualization_demo(batch_result)
        
        # 4. Performance monitoring
        performance_monitoring()
        
        print("\\n" + "="*60)
        print("Advanced usage demonstration completed successfully!")
        print("Check the 'results/' directory for all generated outputs.")
        print("="*60)
        
    except Exception as e:
        print(f"\\nError during advanced demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

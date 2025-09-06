#!/usr/bin/env python3
"""
Main Entry Point for eConsultation AI

This script provides a command-line interface for the eConsultation AI system.
"""

import argparse
import sys
import logging
from pathlib import Path
import json

from econsultation_ai import EConsultationAI

def setup_logging(verbose=False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def load_config(config_file):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}

def create_default_config():
    """Create default configuration"""
    return {
        'sentiment_methods': ['vader', 'textblob'],
        'default_sentiment_method': 'vader',
        'default_summarization_method': 'textrank',
        'max_summary_sentences': 3,
        'max_global_summary_sentences': 5,
        'wordcloud_language': 'english',
        'wordcloud_theme': 'government',
        'min_comment_length': 20,
        'remove_duplicates': True
    }

def analyze_single(args):
    """Handle single comment analysis"""
    config = load_config(args.config) if args.config else create_default_config()
    ai_system = EConsultationAI(config=config)
    
    result = ai_system.analyze_single_comment(
        text=args.text,
        comment_id=args.id or "single_comment",
        include_word_analysis=args.include_keywords
    )
    
    print("\\nSINGLE COMMENT ANALYSIS RESULTS")
    print("=" * 40)
    print(f"Comment ID: {result.comment_id}")
    print(f"Sentiment: {result.sentiment.overall_sentiment} (confidence: {result.sentiment.confidence:.3f})")
    print(f"Method: {result.sentiment.method}")
    print(f"Summary: {result.summary.summary}")
    print(f"Processing Time: {result.processing_time:.3f} seconds")
    
    if result.word_analysis:
        print(f"\\nTop Keywords:")
        for i, (word, freq) in enumerate(result.word_analysis.top_keywords[:10], 1):
            print(f"  {i:2d}. {word} ({freq} occurrences)")

def analyze_file(args):
    """Handle file analysis"""
    config = load_config(args.config) if args.config else create_default_config()
    ai_system = EConsultationAI(config=config)
    
    # Validate input file
    if args.validate:
        print("Validating input file...")
        validation = ai_system.validate_input_file(args.input)
        
        if validation['valid']:
            print("✓ File validation successful")
            print(f"  Format: {validation['file_format']}")
            print(f"  Comments: {validation['total_comments']}")
            print(f"  Estimated time: {validation['estimated_processing_time']:.1f}s")
        else:
            print(f"✗ File validation failed: {validation['error']}")
            return 1
    
    # Process file
    print(f"\\nProcessing file: {args.input}")
    
    try:
        batch_result = ai_system.analyze_from_file(
            file_path=args.input,
            output_dir=args.output
        )
        
        print(f"\\nFILE ANALYSIS COMPLETED")
        print("=" * 40)
        print(f"Comments Processed: {len(batch_result.results)}")
        print(f"Total Processing Time: {batch_result.total_processing_time:.2f} seconds")
        print(f"Success Rate: {batch_result.processing_stats['success_rate']:.1f}%")
        
        print(f"\\nSentiment Distribution:")
        for sentiment, percentage in batch_result.sentiment_distribution.items():
            print(f"  {sentiment.capitalize()}: {percentage:.1f}%")
        
        print(f"\\nTop 5 Keywords:")
        for i, (word, freq) in enumerate(batch_result.global_wordcloud.top_keywords[:5], 1):
            print(f"  {i}. {word} ({freq} occurrences)")
        
        print(f"\\nGlobal Summary:")
        print(f"  {batch_result.global_summary.summary}")
        
        print(f"\\nResults saved to: {args.output}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        return 1
    
    return 0

def main():
    """Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="eConsultation AI - Analyze stakeholder comments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a single comment
  python main.py single "I support these amendments" --id comment_001
  
  # Analyze comments from a file
  python main.py file input.csv --output results/
  
  # Use custom configuration
  python main.py file input.csv --config config.json --output results/
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--config', '-c', type=str,
                       help='Path to configuration JSON file')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single comment analysis
    single_parser = subparsers.add_parser('single', help='Analyze a single comment')
    single_parser.add_argument('text', type=str, help='Comment text to analyze')
    single_parser.add_argument('--id', type=str, help='Comment ID')
    single_parser.add_argument('--include-keywords', action='store_true',
                              help='Include keyword analysis')
    
    # File analysis
    file_parser = subparsers.add_parser('file', help='Analyze comments from file')
    file_parser.add_argument('input', type=str, help='Input file path')
    file_parser.add_argument('--output', '-o', type=str, default='results',
                            help='Output directory (default: results)')
    file_parser.add_argument('--validate', action='store_true',
                            help='Validate input file before processing')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'single':
            return analyze_single(args)
        elif args.command == 'file':
            return analyze_file(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except KeyboardInterrupt:
        print("\\nOperation cancelled by user")
        return 130
    except Exception as e:
        print(f"\\nUnexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Simple test script for eConsultation AI
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_basic_import():
    """Test basic imports"""
    try:
        print("Testing basic imports...")
        
        print("  - Importing sentiment_analyzer...")
        from sentiment_analyzer import SentimentAnalyzer
        
        print("  - Importing text_summarizer...")
        from text_summarizer import TextSummarizer
        
        print("  - Importing wordcloud_generator...")
        from wordcloud_generator import WordCloudGenerator
        
        print("  - Importing data_processor...")
        from data_processor import DataProcessor
        
        print("  - Importing main AI class...")
        from econsultation_ai import EConsultationAI
        
        print("✓ All imports successful!")
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality"""
    try:
        print("\nTesting basic functionality...")
        
        # Import the AI system
        from econsultation_ai import EConsultationAI
        
        # Initialize with minimal config
        config = {
            'sentiment_methods': ['vader'],
            'default_sentiment_method': 'vader',
            'default_summarization_method': 'textrank',
            'max_summary_sentences': 2,
            'wordcloud_language': 'english'
        }
        
        print("  - Initializing AI system...")
        ai_system = EConsultationAI(config=config)
        
        print("  - Testing single comment analysis...")
        test_comment = "I support these amendments. They are necessary for better governance."
        
        result = ai_system.analyze_single_comment(
            text=test_comment,
            comment_id="test_001"
        )
        
        print(f"  - Sentiment: {result.sentiment.overall_sentiment} (confidence: {result.sentiment.confidence:.3f})")
        print(f"  - Summary: {result.summary.summary}")
        print(f"  - Processing time: {result.processing_time:.3f}s")
        
        print("✓ Basic functionality test successful!")
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("eConsultation AI - System Test")
    print("=" * 50)
    
    # Test imports
    import_success = test_basic_import()
    
    if import_success:
        # Test functionality
        func_success = test_basic_functionality()
        
        if func_success:
            print("\n" + "=" * 50)
            print("All tests passed! The system is working correctly.")
            return 0
        else:
            print("\n" + "=" * 50)
            print("Functionality test failed.")
            return 1
    else:
        print("\n" + "=" * 50)
        print("Import test failed.")
        return 1

if __name__ == "__main__":
    exit(main())

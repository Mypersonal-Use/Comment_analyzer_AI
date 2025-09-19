#!/usr/bin/env python3
"""
Deployment test script for eConsultation AI
Verifies all dependencies and modules are properly installed
"""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic Python imports"""
    print("Testing basic imports...")
    
    try:
        import streamlit
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy
        print("✅ NumPy imported successfully")  
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import matplotlib
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
        
    return True

def test_nlp_imports():
    """Test NLP library imports"""
    print("\nTesting NLP imports...")
    
    try:
        import nltk
        print("✅ NLTK imported successfully")
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
        return False
    
    try:
        import textblob
        print("✅ TextBlob imported successfully")
    except ImportError as e:
        print(f"❌ TextBlob import failed: {e}")
        return False
        
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        print("✅ VADER Sentiment imported successfully")
    except ImportError as e:
        print(f"❌ VADER Sentiment import failed: {e}")
        return False
        
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
        
    try:
        import sumy
        print("✅ Sumy imported successfully")
    except ImportError as e:
        print(f"❌ Sumy import failed: {e}")
        return False
        
    return True

def test_project_structure():
    """Test project structure and custom modules"""
    print("\nTesting project structure...")
    
    # Check if src directory exists
    src_path = Path(__file__).parent / 'src'
    if not src_path.exists():
        print(f"❌ src directory not found at: {src_path}")
        return False
    else:
        print(f"✅ src directory found at: {src_path}")
    
    # List src directory contents
    try:
        src_files = list(src_path.glob('*.py'))
        print(f"📁 src directory contents: {[f.name for f in src_files]}")
        
        required_files = ['econsultation_ai.py', 'data_processor.py', 'text_summarizer.py', 'sentiment_analyzer.py']
        missing_files = []
        
        for required_file in required_files:
            if not (src_path / required_file).exists():
                missing_files.append(required_file)
        
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
        else:
            print("✅ All required files found in src directory")
            
    except Exception as e:
        print(f"❌ Error checking src directory: {e}")
        return False
        
    return True

def test_custom_imports():
    """Test custom module imports"""
    print("\nTesting custom module imports...")
    
    # Add src to path
    src_path = str(Path(__file__).parent / 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        print(f"📍 Added to Python path: {src_path}")
    
    try:
        from econsultation_ai import EConsultationAI
        print("✅ EConsultationAI imported successfully")
    except ImportError as e:
        print(f"❌ EConsultationAI import failed: {e}")
        return False
        
    try:
        from data_processor import CommentData
        print("✅ CommentData imported successfully")
    except ImportError as e:
        print(f"❌ CommentData import failed: {e}")
        return False
        
    try:
        from text_summarizer import TextSummarizer
        print("✅ TextSummarizer imported successfully") 
    except ImportError as e:
        print(f"❌ TextSummarizer import failed: {e}")
        return False
        
    return True

def main():
    """Main test function"""
    print("🚀 eConsultation AI Deployment Test")
    print("=" * 50)
    
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {Path(__file__).parent}")
    
    # Run all tests
    tests = [
        test_basic_imports,
        test_nlp_imports, 
        test_project_structure,
        test_custom_imports
    ]
    
    all_passed = True
    for test in tests:
        try:
            if not test():
                all_passed = False
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Your deployment should work correctly.")
    else:
        print("⚠️  Some tests failed. Please fix the issues before deploying.")
        
    return all_passed

if __name__ == "__main__":
    main()

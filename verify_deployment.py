#!/usr/bin/env python3
"""
Verification script for eConsultation AI deployment
Run this to ensure all components work properly before deploying
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all required imports work"""
    print("🔍 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit")
    except ImportError as e:
        print(f"❌ Streamlit: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas")
    except ImportError as e:
        print(f"❌ Pandas: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib")
    except ImportError as e:
        print(f"❌ Matplotlib: {e}")
        return False
    
    try:
        from econsultation_ai import EConsultationAI
        print("✅ eConsultation AI Core")
    except ImportError as e:
        print(f"❌ eConsultation AI Core: {e}")
        return False
    
    return True

def test_ai_functionality():
    """Test basic AI functionality"""
    print("\n🧠 Testing AI functionality...")
    
    try:
        from econsultation_ai import EConsultationAI
        
        # Create AI system
        ai = EConsultationAI()
        print("✅ AI system initialization")
        
        # Test single comment analysis
        result = ai.analyze_single_comment(
            text="This is a test comment for deployment verification.",
            comment_id="test_001"
        )
        
        print("✅ Single comment analysis")
        print(f"   - Sentiment: {result.sentiment.overall_sentiment}")
        print(f"   - Confidence: {result.sentiment.confidence:.3f}")
        print(f"   - Summary: {result.summary.summary[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ AI functionality test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\n📁 Testing file structure...")
    
    required_files = [
        'app.py',
        'requirements.txt',
        'src/econsultation_ai.py',
        'src/sentiment_analyzer.py',
        'src/text_summarizer.py',
        'src/data_processor.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING!")
            all_exist = False
    
    return all_exist

def main():
    """Run all verification tests"""
    print("="*60)
    print("🚀 eConsultation AI - Deployment Verification")
    print("="*60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Imports", test_imports),
        ("AI Functionality", test_ai_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: Unexpected error - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Ready for deployment!")
        print("🚀 You can now deploy to Streamlit Cloud with confidence!")
    else:
        print("⚠️  Some tests failed - Please fix issues before deployment")
    print("="*60)
    
    return all_passed

if __name__ == "__main__":
    main()

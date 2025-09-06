#!/usr/bin/env python3
"""
Streamlit Setup Script for eConsultation AI
Ensures all required dependencies and data are properly set up
"""

import os
import sys
import logging
from pathlib import Path

def setup_nltk_data():
    """Setup NLTK data for Streamlit deployment"""
    try:
        import nltk
        
        # Set NLTK data path to a writable location in Streamlit
        if 'STREAMLIT' in os.environ or 'streamlit' in sys.modules:
            nltk_data_path = os.path.expanduser('~/nltk_data')
            if nltk_data_path not in nltk.data.path:
                nltk.data.path.append(nltk_data_path)
        
        # Download required data
        required_downloads = [
            ('punkt', 'tokenizers/punkt'),
            ('stopwords', 'corpora/stopwords')
        ]
        
        for download_name, data_path in required_downloads:
            try:
                nltk.data.find(data_path)
                print(f"✓ NLTK {download_name} already available")
            except LookupError:
                try:
                    print(f"Downloading NLTK {download_name}...")
                    nltk.download(download_name, quiet=True)
                    print(f"✓ NLTK {download_name} downloaded successfully")
                except Exception as e:
                    print(f"⚠ Warning: Could not download NLTK {download_name}: {e}")
        
        return True
        
    except ImportError:
        print("⚠ Warning: NLTK not available")
        return False
    except Exception as e:
        print(f"⚠ Warning: Error setting up NLTK: {e}")
        return False

def verify_dependencies():
    """Verify all required dependencies are available"""
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'matplotlib',
        'nltk',
        'textblob',
        'vaderSentiment',
        'sklearn',
        'sumy',
        'wordcloud'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} missing")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements_web.txt")
        return False
    
    print("\n✓ All required packages are available")
    return True

def main():
    """Main setup function"""
    print("=" * 50)
    print("eConsultation AI - Streamlit Setup")
    print("=" * 50)
    
    # Verify dependencies
    print("\n1. Verifying dependencies...")
    deps_ok = verify_dependencies()
    
    # Setup NLTK data
    print("\n2. Setting up NLTK data...")
    nltk_ok = setup_nltk_data()
    
    # Summary
    print("\n" + "=" * 50)
    if deps_ok and nltk_ok:
        print("✓ Setup completed successfully!")
        print("You can now run: streamlit run app.py")
    else:
        print("⚠ Setup completed with warnings")
        print("The application may still work with limited functionality")
    print("=" * 50)

if __name__ == "__main__":
    main()

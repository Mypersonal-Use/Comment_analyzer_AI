#!/usr/bin/env python3
"""
eConsultation AI - Web Application
Streamlit-based web interface for analyzing stakeholder comments
"""

import streamlit as st
import sys
import os
import time
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web deployment
import matplotlib.pyplot as plt

# Optional seaborn import
try:
    import seaborn as sns
except ImportError:
    sns = None
from pathlib import Path
import io
import base64
from typing import Dict, List
import tempfile

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Setup NLTK data for Streamlit deployment
@st.cache_resource
def setup_nltk():
    """Setup NLTK data with caching to avoid repeated downloads"""
    import nltk
    import logging
    import os
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Ensure NLTK data directory exists
    try:
        nltk_data_dir = os.path.expanduser('~/nltk_data')
        os.makedirs(nltk_data_dir, exist_ok=True)
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.append(nltk_data_dir)
        logger.info(f"NLTK data directory: {nltk_data_dir}")
    except Exception as e:
        logger.warning(f"Could not setup NLTK data directory: {e}")
    
    # Download required data more aggressively
    nltk_downloads = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords')
    ]
    
    for download_name, data_path in nltk_downloads:
        try:
            nltk.data.find(data_path)
            logger.info(f"NLTK {download_name} already available")
        except LookupError:
            try:
                logger.info(f"Downloading NLTK {download_name}...")
                nltk.download(download_name, quiet=False)  # Show download progress
                logger.info(f"Successfully downloaded NLTK {download_name}")
            except Exception as e:
                logger.error(f"Failed to download NLTK {download_name}: {e}")
                try:
                    # Try alternative download method
                    nltk.download(download_name, download_dir=nltk_data_dir, quiet=False)
                    logger.info(f"Successfully downloaded NLTK {download_name} to {nltk_data_dir}")
                except Exception as e2:
                    logger.error(f"Alternative download also failed for {download_name}: {e2}")
    
    return True

# Initialize NLTK data
setup_nltk()

# Import our AI system
from econsultation_ai import EConsultationAI
from data_processor import CommentData

# Configure page
st.set_page_config(
    page_title="eConsultation AI",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64748b;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
    }
    .sentiment-positive { color: #059669; font-weight: bold; }
    .sentiment-negative { color: #dc2626; font-weight: bold; }
    .sentiment-neutral { color: #d97706; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'ai_system' not in st.session_state:
        st.session_state.ai_system = None

def get_ai_system():
    """Get or create AI system with caching"""
    if st.session_state.ai_system is None:
        config = {
            'sentiment_methods': ['vader', 'textblob'],
            'default_sentiment_method': 'vader',
            'default_summarization_method': 'textrank',
            'max_summary_sentences': 3,
            'max_global_summary_sentences': 5,
            'wordcloud_language': 'english',
            'custom_stopwords': ['amendment', 'provision', 'section', 'clause'],
            'min_comment_length': 20,
            'remove_duplicates': True,
            'wordcloud_theme': 'government'
        }
        st.session_state.ai_system = EConsultationAI(config=config)
    return st.session_state.ai_system

def analyze_single_comment(comment_text, comment_id):
    """Analyze a single comment"""
    ai_system = get_ai_system()
    
    with st.spinner("Analyzing comment..."):
        result = ai_system.analyze_single_comment(
            text=comment_text,
            comment_id=comment_id,
            include_word_analysis=True
        )
    
    return result

def analyze_batch_comments(comments_data, analysis_type):
    """Analyze batch of comments"""
    ai_system = get_ai_system()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("Processing comments...")
        progress_bar.progress(25)
        
        if analysis_type == "text_list":
            # Convert text list to CommentData objects
            comment_objects = [
                CommentData(id=str(i), text=text.strip()) 
                for i, text in enumerate(comments_data, 1) 
                if text.strip()
            ]
        elif analysis_type == "file_upload":
            comment_objects = comments_data
        else:
            comment_objects = comments_data
        
        progress_bar.progress(50)
        status_text.text("Running AI analysis...")
        
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            batch_result = ai_system.analyze_batch(
                comments=comment_objects,
                output_dir=temp_dir,
                generate_reports=True
            )
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        return batch_result
        
    except Exception as e:
        st.error(f"Error during analysis: {str(e)}")
        return None
    finally:
        progress_bar.empty()
        status_text.empty()

def display_sentiment_metrics(results):
    """Display sentiment analysis metrics"""
    if hasattr(results, 'sentiment_distribution'):
        # Batch results
        sentiment_dist = results.sentiment_distribution
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="😊 Positive Sentiment",
                value=f"{sentiment_dist.get('positive', 0):.1f}%",
                delta=None
            )
        
        with col2:
            st.metric(
                label="😐 Neutral Sentiment", 
                value=f"{sentiment_dist.get('neutral', 0):.1f}%",
                delta=None
            )
            
        with col3:
            st.metric(
                label="😟 Negative Sentiment",
                value=f"{sentiment_dist.get('negative', 0):.1f}%",
                delta=None
            )
    else:
        # Single comment result
        sentiment = results.sentiment.overall_sentiment
        confidence = results.sentiment.confidence
        
        col1, col2 = st.columns(2)
        
        with col1:
            sentiment_class = f"sentiment-{sentiment}"
            st.markdown(f"**Sentiment:** <span class='{sentiment_class}'>{sentiment.title()}</span>", 
                       unsafe_allow_html=True)
        
        with col2:
            st.metric("Confidence", f"{confidence:.3f}")

def display_wordcloud_viz(result):
    """Create and display word cloud visualization"""
    if not hasattr(result, 'global_wordcloud'):
        return
    
    wordcloud_data = result.global_wordcloud
    if not wordcloud_data.top_keywords:
        st.warning("No keywords found for word cloud generation.")
        return
    
    # Create word cloud visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get top 20 keywords
    top_words = wordcloud_data.top_keywords[:20]
    words, frequencies = zip(*top_words)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(words))
    bars = ax.barh(y_pos, frequencies, color='steelblue', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency')
    ax.set_title('Top Keywords by Frequency', fontsize=16, fontweight='bold')
    
    # Add value labels
    for i, (bar, freq) in enumerate(zip(bars, frequencies)):
        ax.text(freq + max(frequencies) * 0.01, i, str(freq), 
               va='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)

def create_download_button(data, filename, label):
    """Create a download button for data"""
    if isinstance(data, dict):
        # JSON data
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        b64 = base64.b64encode(json_str.encode()).decode()
        href = f'<a href="data:application/json;base64,{b64}" download="{filename}">{label}</a>'
    elif isinstance(data, str):
        # Text data
        b64 = base64.b64encode(data.encode()).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="{filename}">{label}</a>'
    else:
        return None
    
    st.markdown(href, unsafe_allow_html=True)

def main():
    """Main application function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">🏛️ eConsultation AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered analysis for stakeholder comments and public consultation feedback</p>', 
               unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Analysis Options")
    analysis_mode = st.sidebar.selectbox(
        "Choose Analysis Mode",
        ["Single Comment", "Multiple Comments", "File Upload"]
    )
    
    # Main content based on mode
    if analysis_mode == "Single Comment":
        st.header("📝 Single Comment Analysis")
        
        comment_text = st.text_area(
            "Enter your comment for analysis:",
            height=150,
            placeholder="I strongly support the proposed amendments to improve governance standards..."
        )
        
        comment_id = st.text_input("Comment ID (optional):", placeholder="comment_001")
        
        if st.button("🔍 Analyze Comment", type="primary"):
            if comment_text.strip():
                result = analyze_single_comment(
                    comment_text, 
                    comment_id or f"comment_{int(time.time())}"
                )
                
                # Display results
                st.success("Analysis Complete!")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("💭 Summary")
                    st.write(result.summary.summary)
                    
                with col2:
                    st.subheader("📊 Sentiment Analysis")
                    display_sentiment_metrics(result)
                
                # Keywords
                if result.word_analysis and result.word_analysis.top_keywords:
                    st.subheader("🔑 Top Keywords")
                    keywords_df = pd.DataFrame(
                        result.word_analysis.top_keywords[:10],
                        columns=['Keyword', 'Frequency']
                    )
                    keywords_df.index = keywords_df.index + 1
                    st.dataframe(keywords_df, width='stretch')
                
                # Processing info
                st.info(f"⏱️ Processing time: {result.processing_time:.3f} seconds")
                
            else:
                st.error("Please enter a comment to analyze.")
    
    elif analysis_mode == "Multiple Comments":
        st.header("📚 Multiple Comments Analysis")
        
        st.write("Enter multiple comments (one per line):")
        comments_text = st.text_area(
            "Comments:",
            height=200,
            placeholder="I support these amendments.\nThe implementation timeline seems too short.\nThese changes are necessary for better governance."
        )
        
        if st.button("🔍 Analyze All Comments", type="primary"):
            if comments_text.strip():
                comments_list = [
                    line.strip() for line in comments_text.split('\n') 
                    if line.strip()
                ]
                
                if len(comments_list) > 0:
                    result = analyze_batch_comments(comments_list, "text_list")
                    
                    if result:
                        st.success(f"Successfully analyzed {len(result.results)} comments!")
                        
                        # Overview metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Comments", len(result.results))
                        with col2:
                            st.metric("Processing Time", f"{result.total_processing_time:.2f}s")
                        with col3:
                            st.metric("Success Rate", f"{result.processing_stats['success_rate']:.1f}%")
                        with col4:
                            st.metric("Avg Confidence", f"{result.processing_stats['avg_sentiment_confidence']:.3f}")
                        
                        # Sentiment distribution
                        st.subheader("📊 Sentiment Analysis")
                        display_sentiment_metrics(result)
                        
                        # Global summary
                        st.subheader("📄 Global Summary")
                        st.write(result.global_summary.summary)
                        
                        # Keywords visualization
                        st.subheader("🔑 Keyword Analysis")
                        display_wordcloud_viz(result)
                        
                        # Top keywords table
                        if result.global_wordcloud.top_keywords:
                            keywords_df = pd.DataFrame(
                                result.global_wordcloud.top_keywords[:15],
                                columns=['Keyword', 'Frequency']
                            )
                            keywords_df.index = keywords_df.index + 1
                            st.dataframe(keywords_df, width='stretch')
                        
                        # Store results for download
                        st.session_state.analysis_results = result
                        
                else:
                    st.error("Please enter at least one comment.")
            else:
                st.error("Please enter comments to analyze.")
    
    else:  # File Upload
        st.header("📁 File Upload Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a file", 
            type=['csv', 'json', 'txt', 'xlsx'],
            help="Upload CSV, JSON, TXT, or Excel files with stakeholder comments"
        )
        
        if uploaded_file is not None:
            st.success(f"File uploaded: {uploaded_file.name}")
            
            # Show file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size} bytes",
                "File type": uploaded_file.type
            }
            st.json(file_details)
            
            if st.button("🔍 Analyze File", type="primary"):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name
                    
                    # Load and analyze
                    ai_system = get_ai_system()
                    
                    with st.spinner("Loading and validating file..."):
                        comments = ai_system.data_processor.load_data(tmp_file_path)
                    
                    st.info(f"Found {len(comments)} comments in the file")
                    
                    if len(comments) > 0:
                        result = analyze_batch_comments(comments, "file_upload")
                        
                        if result:
                            st.success(f"Successfully analyzed {len(result.results)} comments!")
                            
                            # Overview metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Total Comments", len(result.results))
                            with col2:
                                st.metric("Processing Time", f"{result.total_processing_time:.2f}s")
                            with col3:
                                st.metric("Success Rate", f"{result.processing_stats['success_rate']:.1f}%")
                            with col4:
                                st.metric("Avg Confidence", f"{result.processing_stats['avg_sentiment_confidence']:.3f}")
                            
                            # Sentiment distribution
                            st.subheader("📊 Sentiment Analysis")
                            display_sentiment_metrics(result)
                            
                            # Global summary
                            st.subheader("📄 Global Summary")
                            st.write(result.global_summary.summary)
                            
                            # Keywords visualization  
                            st.subheader("🔑 Keyword Analysis")
                            display_wordcloud_viz(result)
                            
                            # Individual results preview
                            st.subheader("📋 Individual Results Preview")
                            preview_data = []
                            for i, res in enumerate(result.results[:10], 1):
                                preview_data.append({
                                    "Comment ID": res.comment_id,
                                    "Sentiment": res.sentiment.overall_sentiment,
                                    "Confidence": f"{res.sentiment.confidence:.3f}",
                                    "Summary": res.summary.summary[:100] + "..." if len(res.summary.summary) > 100 else res.summary.summary
                                })
                            
                            preview_df = pd.DataFrame(preview_data)
                            st.dataframe(preview_df, width='stretch')
                            
                            if len(result.results) > 10:
                                st.info(f"Showing first 10 results. Total: {len(result.results)} comments analyzed.")
                            
                            # Store results for download
                            st.session_state.analysis_results = result
                    
                    # Clean up temp file
                    os.unlink(tmp_file_path)
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
    
    # Download section
    if st.session_state.analysis_results:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📥 Download Results")
        
        result = st.session_state.analysis_results
        
        # Prepare downloadable data
        summary_text = f"""eConsultation AI - Analysis Summary
{'='*50}

OVERVIEW
Total Comments: {len(result.results) if hasattr(result, 'results') else 1}
Processing Time: {result.total_processing_time:.2f} seconds
Success Rate: {result.processing_stats['success_rate']:.1f}%

SENTIMENT ANALYSIS
{chr(10).join([f"{k.title()}: {v:.1f}%" for k, v in result.sentiment_distribution.items()])}

GLOBAL SUMMARY
{result.global_summary.summary}

TOP KEYWORDS
{chr(10).join([f"{i:2d}. {word} ({freq} occurrences)" for i, (word, freq) in enumerate(result.global_wordcloud.top_keywords[:10], 1)])}
"""
        
        if st.sidebar.button("📄 Download Summary Report"):
            st.sidebar.download_button(
                label="Click to Download",
                data=summary_text,
                file_name="eConsultation_analysis_summary.txt",
                mime="text/plain"
            )
        
        # Keywords CSV
        if hasattr(result, 'global_wordcloud') and result.global_wordcloud.top_keywords:
            keywords_df = pd.DataFrame(
                result.global_wordcloud.top_keywords,
                columns=['Keyword', 'Frequency']
            )
            
            if st.sidebar.button("📊 Download Keywords CSV"):
                csv = keywords_df.to_csv(index=False)
                st.sidebar.download_button(
                    label="Click to Download",
                    data=csv,
                    file_name="eConsultation_keywords.csv",
                    mime="text/csv"
                )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #64748b;'>
            <p>eConsultation AI - Powered by Advanced NLP & Machine Learning</p>
            <p>Built for analyzing stakeholder feedback and public consultation responses</p>
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

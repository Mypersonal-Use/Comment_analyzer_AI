"""
Data Processing Utilities for eConsultation AI
Handles various input formats and data preprocessing tasks
"""

import logging
import os
import json
import csv
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import re

import pandas as pd
import numpy as np

@dataclass
class CommentData:
    """Data class to store individual comment information"""
    id: str
    text: str
    timestamp: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ProcessingStats:
    """Data class to store data processing statistics"""
    total_records: int
    valid_records: int
    invalid_records: int
    empty_records: int
    duplicate_records: int
    avg_text_length: float
    processing_time: float

class DataProcessor:
    """
    Comprehensive data processing class for handling various input formats
    and preprocessing tasks
    """
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Initialize the data processor
        
        Args:
            encoding: Default encoding for file operations
        """
        self.encoding = encoding
        self.supported_formats = ['.csv', '.json', '.txt', '.xlsx', '.xls']
        
    def load_data(self, file_path: str, format_hint: str = None) -> List[CommentData]:
        """
        Load data from various file formats
        
        Args:
            file_path: Path to the data file
            format_hint: Hint about file format if extension is unclear
            
        Returns:
            List of CommentData objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Determine file format
        file_ext = Path(file_path).suffix.lower()
        if format_hint:
            file_ext = f".{format_hint.lower()}"
            
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported: {self.supported_formats}")
        
        # Load data based on format
        if file_ext == '.csv':
            return self._load_csv(file_path)
        elif file_ext == '.json':
            return self._load_json(file_path)
        elif file_ext == '.txt':
            return self._load_txt(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return self._load_excel(file_path)
        else:
            raise ValueError(f"Handler not implemented for format: {file_ext}")
    
    def _load_csv(self, file_path: str) -> List[CommentData]:
        """Load data from CSV file"""
        comments = []
        
        try:
            df = pd.read_csv(file_path, encoding=self.encoding)
            
            # Check for required columns
            required_cols = {'text', 'comment', 'content', 'message'}
            text_col = None
            
            for col in df.columns:
                if col.lower() in required_cols:
                    text_col = col
                    break
            
            if text_col is None:
                # Try to find any column that might contain text
                for col in df.columns:
                    if df[col].dtype == 'object' and df[col].str.len().mean() > 20:
                        text_col = col
                        break
                        
            if text_col is None:
                raise ValueError("No suitable text column found in CSV")
            
            # Process each row
            for idx, row in df.iterrows():
                comment_id = str(row.get('id', idx))
                text = str(row[text_col]) if pd.notna(row[text_col]) else ""
                
                if text.strip():  # Only process non-empty text
                    comment = CommentData(
                        id=comment_id,
                        text=text,
                        timestamp=str(row.get('timestamp', '')) if pd.notna(row.get('timestamp')) else None,
                        author=str(row.get('author', '')) if pd.notna(row.get('author')) else None,
                        category=str(row.get('category', '')) if pd.notna(row.get('category')) else None,
                        metadata={col: str(row[col]) for col in df.columns if col not in ['id', text_col, 'timestamp', 'author', 'category']}
                    )
                    comments.append(comment)
                    
        except Exception as e:
            logging.error(f"Error loading CSV file: {e}")
            raise
            
        return comments
    
    def _load_json(self, file_path: str) -> List[CommentData]:
        """Load data from JSON file"""
        comments = []
        
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                # List of comment objects
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        comment = self._dict_to_comment(item, idx)
                        if comment:
                            comments.append(comment)
            elif isinstance(data, dict):
                # Single object or nested structure
                if 'comments' in data:
                    # Nested structure with comments array
                    for idx, item in enumerate(data['comments']):
                        comment = self._dict_to_comment(item, idx)
                        if comment:
                            comments.append(comment)
                else:
                    # Single comment object
                    comment = self._dict_to_comment(data, 0)
                    if comment:
                        comments.append(comment)
                        
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON format: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading JSON file: {e}")
            raise
            
        return comments
    
    def _dict_to_comment(self, data: Dict[str, Any], default_id: int) -> Optional[CommentData]:
        """Convert dictionary to CommentData object"""
        # Find text field
        text_fields = ['text', 'comment', 'content', 'message', 'description']
        text = ""
        
        for field in text_fields:
            if field in data and data[field]:
                text = str(data[field])
                break
                
        if not text.strip():
            return None
            
        # Extract other fields
        comment_id = str(data.get('id', default_id))
        timestamp = data.get('timestamp') or data.get('created_at') or data.get('date')
        author = data.get('author') or data.get('user') or data.get('name')
        category = data.get('category') or data.get('type') or data.get('classification')
        
        # Collect metadata (remaining fields)
        excluded_fields = {'id', 'text', 'comment', 'content', 'message', 'description', 
                          'timestamp', 'created_at', 'date', 'author', 'user', 'name',
                          'category', 'type', 'classification'}
        metadata = {k: v for k, v in data.items() if k not in excluded_fields}
        
        return CommentData(
            id=comment_id,
            text=text,
            timestamp=str(timestamp) if timestamp else None,
            author=str(author) if author else None,
            category=str(category) if category else None,
            metadata=metadata if metadata else None
        )
    
    def _load_txt(self, file_path: str) -> List[CommentData]:
        """Load data from text file"""
        comments = []
        
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                content = f.read()
            
            # Split by common delimiters
            delimiters = ['\n\n', '\n---\n', '\n===\n', '\n***\n']
            
            # Try different delimiters
            parts = [content]  # Start with whole content
            for delimiter in delimiters:
                if delimiter in content:
                    parts = content.split(delimiter)
                    break
            
            # If no delimiter found, split by single newlines for short texts
            if len(parts) == 1 and '\n' in content:
                lines = content.split('\n')
                # Group consecutive non-empty lines
                parts = []
                current_part = []
                
                for line in lines:
                    if line.strip():
                        current_part.append(line.strip())
                    else:
                        if current_part:
                            parts.append(' '.join(current_part))
                            current_part = []
                            
                if current_part:
                    parts.append(' '.join(current_part))
            
            # Create comment objects
            for idx, part in enumerate(parts):
                text = part.strip()
                if text and len(text) > 10:  # Filter out very short text
                    comment = CommentData(
                        id=str(idx + 1),
                        text=text
                    )
                    comments.append(comment)
                    
        except Exception as e:
            logging.error(f"Error loading text file: {e}")
            raise
            
        return comments
    
    def _load_excel(self, file_path: str) -> List[CommentData]:
        """Load data from Excel file"""
        try:
            df = pd.read_excel(file_path)
            
            # Convert to CSV-like format and reuse CSV logic
            temp_data = []
            for idx, row in df.iterrows():
                temp_data.append(row.to_dict())
                
            # Find text column similar to CSV method
            text_cols = {'text', 'comment', 'content', 'message'}
            text_col = None
            
            for col in df.columns:
                if col.lower() in text_cols:
                    text_col = col
                    break
                    
            if text_col is None:
                for col in df.columns:
                    if df[col].dtype == 'object' and df[col].astype(str).str.len().mean() > 20:
                        text_col = col
                        break
                        
            if text_col is None:
                raise ValueError("No suitable text column found in Excel file")
            
            comments = []
            for idx, row_dict in enumerate(temp_data):
                text = str(row_dict.get(text_col, '')) if pd.notna(row_dict.get(text_col)) else ""
                
                if text.strip():
                    comment = CommentData(
                        id=str(row_dict.get('id', idx)),
                        text=text,
                        timestamp=str(row_dict.get('timestamp', '')) if pd.notna(row_dict.get('timestamp')) else None,
                        author=str(row_dict.get('author', '')) if pd.notna(row_dict.get('author')) else None,
                        category=str(row_dict.get('category', '')) if pd.notna(row_dict.get('category')) else None,
                        metadata={k: str(v) for k, v in row_dict.items() if k not in ['id', text_col, 'timestamp', 'author', 'category']}
                    )
                    comments.append(comment)
                    
            return comments
            
        except Exception as e:
            logging.error(f"Error loading Excel file: {e}")
            raise
    
    def clean_text(self, text: str, remove_urls: bool = True, 
                  remove_emails: bool = True, normalize_whitespace: bool = True) -> str:
        """
        Clean and normalize text content
        
        Args:
            text: Text to clean
            remove_urls: Whether to remove URLs
            remove_emails: Whether to remove email addresses
            normalize_whitespace: Whether to normalize whitespace
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        cleaned = text
        
        # Remove URLs
        if remove_urls:
            url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            cleaned = re.sub(url_pattern, '', cleaned)
            
        # Remove email addresses
        if remove_emails:
            email_pattern = r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b'
            cleaned = re.sub(email_pattern, '', cleaned)
            
        # Normalize whitespace
        if normalize_whitespace:
            cleaned = re.sub(r'\\s+', ' ', cleaned.strip())
            
        return cleaned
    
    def filter_comments(self, comments: List[CommentData], 
                       min_length: int = 10, max_length: int = None,
                       remove_duplicates: bool = True) -> List[CommentData]:
        """
        Filter comments based on various criteria
        
        Args:
            comments: List of comments to filter
            min_length: Minimum text length
            max_length: Maximum text length (None for no limit)
            remove_duplicates: Whether to remove duplicate texts
            
        Returns:
            Filtered list of comments
        """
        filtered = []
        seen_texts = set()
        
        for comment in comments:
            # Clean text
            cleaned_text = self.clean_text(comment.text)
            
            # Length check
            if len(cleaned_text) < min_length:
                continue
                
            if max_length and len(cleaned_text) > max_length:
                continue
                
            # Duplicate check
            if remove_duplicates:
                text_normalized = cleaned_text.lower().strip()
                if text_normalized in seen_texts:
                    continue
                seen_texts.add(text_normalized)
                
            # Update comment with cleaned text
            filtered_comment = CommentData(
                id=comment.id,
                text=cleaned_text,
                timestamp=comment.timestamp,
                author=comment.author,
                category=comment.category,
                metadata=comment.metadata
            )
            filtered.append(filtered_comment)
            
        return filtered
    
    def save_data(self, comments: List[CommentData], file_path: str, 
                 format_type: str = 'csv') -> str:
        """
        Save comments to file
        
        Args:
            comments: List of comments to save
            file_path: Output file path
            format_type: Output format ('csv', 'json', 'excel')
            
        Returns:
            Path to saved file
        """
        if format_type.lower() == 'csv':
            return self._save_csv(comments, file_path)
        elif format_type.lower() == 'json':
            return self._save_json(comments, file_path)
        elif format_type.lower() in ['excel', 'xlsx']:
            return self._save_excel(comments, file_path)
        else:
            raise ValueError(f"Unsupported output format: {format_type}")
    
    def _save_csv(self, comments: List[CommentData], file_path: str) -> str:
        """Save comments to CSV file"""
        try:
            # Convert to DataFrame
            data = []
            for comment in comments:
                row = {
                    'id': comment.id,
                    'text': comment.text,
                    'timestamp': comment.timestamp or '',
                    'author': comment.author or '',
                    'category': comment.category or ''
                }
                
                # Add metadata fields
                if comment.metadata:
                    row.update(comment.metadata)
                    
                data.append(row)
                
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False, encoding=self.encoding)
            
            logging.info(f"Saved {len(comments)} comments to CSV: {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving CSV file: {e}")
            raise
    
    def _save_json(self, comments: List[CommentData], file_path: str) -> str:
        """Save comments to JSON file"""
        try:
            data = [asdict(comment) for comment in comments]
            
            with open(file_path, 'w', encoding=self.encoding) as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logging.info(f"Saved {len(comments)} comments to JSON: {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving JSON file: {e}")
            raise
    
    def _save_excel(self, comments: List[CommentData], file_path: str) -> str:
        """Save comments to Excel file"""
        try:
            # Convert to DataFrame similar to CSV
            data = []
            for comment in comments:
                row = {
                    'id': comment.id,
                    'text': comment.text,
                    'timestamp': comment.timestamp or '',
                    'author': comment.author or '',
                    'category': comment.category or ''
                }
                
                if comment.metadata:
                    row.update(comment.metadata)
                    
                data.append(row)
                
            df = pd.DataFrame(data)
            df.to_excel(file_path, index=False)
            
            logging.info(f"Saved {len(comments)} comments to Excel: {file_path}")
            return file_path
            
        except Exception as e:
            logging.error(f"Error saving Excel file: {e}")
            raise
    
    def get_processing_stats(self, original_count: int, 
                           processed_comments: List[CommentData],
                           processing_time: float) -> ProcessingStats:
        """
        Generate processing statistics
        
        Args:
            original_count: Original number of records
            processed_comments: Final processed comments
            processing_time: Time taken for processing
            
        Returns:
            ProcessingStats object
        """
        valid_count = len(processed_comments)
        invalid_count = original_count - valid_count
        
        # Calculate average text length
        if processed_comments:
            avg_length = np.mean([len(comment.text) for comment in processed_comments])
        else:
            avg_length = 0
            
        return ProcessingStats(
            total_records=original_count,
            valid_records=valid_count,
            invalid_records=invalid_count,
            empty_records=0,  # This would need to be tracked during processing
            duplicate_records=0,  # This would need to be tracked during processing
            avg_text_length=avg_length,
            processing_time=processing_time
        )
    
    def batch_process(self, input_dir: str, output_dir: str, 
                     file_pattern: str = "*.csv") -> Dict[str, ProcessingStats]:
        """
        Process multiple files in batch
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory for output files
            file_pattern: Pattern to match files
            
        Returns:
            Dictionary with processing stats for each file
        """
        import glob
        import time
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        files = list(input_path.glob(file_pattern))
        results = {}
        
        for file_path in files:
            start_time = time.time()
            
            try:
                # Load data
                comments = self.load_data(str(file_path))
                original_count = len(comments)
                
                # Process data
                filtered_comments = self.filter_comments(comments)
                
                # Save processed data
                output_file = output_path / f"processed_{file_path.name}"
                self.save_data(filtered_comments, str(output_file))
                
                # Generate stats
                processing_time = time.time() - start_time
                stats = self.get_processing_stats(original_count, filtered_comments, processing_time)
                results[file_path.name] = stats
                
                logging.info(f"Processed {file_path.name}: {stats.valid_records}/{stats.total_records} records")
                
            except Exception as e:
                logging.error(f"Error processing {file_path.name}: {e}")
                results[file_path.name] = None
                
        return results

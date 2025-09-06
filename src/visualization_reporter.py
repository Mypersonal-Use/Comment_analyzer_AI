"""
Visualization and Reporting Module for eConsultation AI
Creates comprehensive visual reports and dashboards
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np

# Try importing plotly for interactive charts
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not available. Interactive charts will be disabled.")

from econsultation_ai import BatchAnalysisResult, AnalysisResult

class VisualizationReporter:
    """
    Comprehensive visualization and reporting class
    """
    
    def __init__(self, style: str = 'government'):
        """
        Initialize the visualization reporter
        
        Args:
            style: Visual style theme ('government', 'professional', 'modern')
        """
        self.style = style
        self.logger = logging.getLogger(__name__)
        
        # Set up matplotlib style
        self._setup_matplotlib_style()
        
        # Color palettes for different themes
        self.color_palettes = {
            'government': {
                'primary': '#1f4e79',
                'secondary': '#8bb6df',
                'positive': '#2d5016',
                'negative': '#8b1a1a',
                'neutral': '#666666',
                'background': '#f8f9fa'
            },
            'professional': {
                'primary': '#2c3e50',
                'secondary': '#7fb3d3',
                'positive': '#27ae60',
                'negative': '#e74c3c',
                'neutral': '#95a5a6',
                'background': '#ffffff'
            },
            'modern': {
                'primary': '#6a1b9a',
                'secondary': '#ab47bc',
                'positive': '#4caf50',
                'negative': '#f44336',
                'neutral': '#607d8b',
                'background': '#fafafa'
            }
        }
        
        self.current_palette = self.color_palettes.get(style, self.color_palettes['government'])
    
    def _setup_matplotlib_style(self):
        """Setup matplotlib styling"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Set default font sizes
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16
        })
    
    def create_sentiment_distribution_chart(self, batch_result: BatchAnalysisResult, 
                                          save_path: str = None, 
                                          chart_type: str = 'pie') -> str:
        """
        Create sentiment distribution visualization
        
        Args:
            batch_result: Analysis results
            save_path: Path to save the chart
            chart_type: Type of chart ('pie', 'bar', 'donut')
            
        Returns:
            Path to saved chart
        """
        sentiment_dist = batch_result.sentiment_distribution
        
        if chart_type == 'pie':
            fig, ax = plt.subplots(figsize=(10, 8))
            
            colors = [self.current_palette['positive'], self.current_palette['negative'], self.current_palette['neutral']]
            wedges, texts, autotexts = ax.pie(
                sentiment_dist.values(),
                labels=[s.capitalize() for s in sentiment_dist.keys()],
                autopct='%1.1f%%',
                colors=colors,
                startangle=90,
                explode=(0.05, 0.05, 0.05)
            )
            
            # Enhance text
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(12)
            
            ax.set_title('Sentiment Distribution of Stakeholder Comments', 
                        fontsize=16, fontweight='bold', pad=20)
            
        elif chart_type == 'bar':
            fig, ax = plt.subplots(figsize=(10, 6))
            
            sentiments = list(sentiment_dist.keys())
            percentages = list(sentiment_dist.values())
            colors = [self.current_palette['positive'] if s == 'positive' 
                     else self.current_palette['negative'] if s == 'negative' 
                     else self.current_palette['neutral'] for s in sentiments]
            
            bars = ax.bar(sentiments, percentages, color=colors, alpha=0.8)
            
            # Add value labels on bars
            for bar, pct in zip(bars, percentages):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Percentage of Comments')
            ax.set_title('Sentiment Distribution of Stakeholder Comments', 
                        fontsize=16, fontweight='bold')
            ax.set_ylim(0, max(percentages) * 1.2)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            self.logger.info(f"Sentiment distribution chart saved: {save_path}")
        
        plt.show()
        return save_path or ""
    
    def create_processing_timeline(self, batch_result: BatchAnalysisResult, 
                                 save_path: str = None) -> str:
        """
        Create processing timeline visualization
        
        Args:
            batch_result: Analysis results
            save_path: Path to save the chart
            
        Returns:
            Path to saved chart
        """
        # Extract processing times
        processing_times = [result.processing_time for result in batch_result.results]
        comment_ids = [result.comment_id for result in batch_result.results]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Timeline plot
        ax1.plot(range(len(processing_times)), processing_times, 
                color=self.current_palette['primary'], linewidth=2, alpha=0.7)
        ax1.fill_between(range(len(processing_times)), processing_times, 
                        alpha=0.3, color=self.current_palette['secondary'])
        ax1.set_xlabel('Comment Index')
        ax1.set_ylabel('Processing Time (seconds)')
        ax1.set_title('Individual Comment Processing Times', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Histogram
        ax2.hist(processing_times, bins=20, color=self.current_palette['primary'], 
                alpha=0.7, edgecolor='white')
        ax2.set_xlabel('Processing Time (seconds)')
        ax2.set_ylabel('Number of Comments')
        ax2.set_title('Distribution of Processing Times', fontweight='bold')
        ax2.axvline(np.mean(processing_times), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(processing_times):.3f}s')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            self.logger.info(f"Processing timeline saved: {save_path}")
        
        plt.show()
        return save_path or ""
    
    def create_sentiment_confidence_scatter(self, batch_result: BatchAnalysisResult, 
                                          save_path: str = None) -> str:
        """
        Create scatter plot of sentiment vs confidence
        
        Args:
            batch_result: Analysis results
            save_path: Path to save the chart
            
        Returns:
            Path to saved chart
        """
        # Extract data
        sentiments = []
        confidences = []
        colors = []
        
        for result in batch_result.results:
            sentiments.append(result.sentiment.overall_sentiment)
            confidences.append(result.sentiment.confidence)
            
            # Color mapping
            if result.sentiment.overall_sentiment == 'positive':
                colors.append(self.current_palette['positive'])
            elif result.sentiment.overall_sentiment == 'negative':
                colors.append(self.current_palette['negative'])
            else:
                colors.append(self.current_palette['neutral'])
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Group by sentiment for better visualization
        sentiment_types = ['positive', 'negative', 'neutral']
        for i, sent_type in enumerate(sentiment_types):
            mask = [s == sent_type for s in sentiments]
            conf_values = [c for c, m in zip(confidences, mask) if m]
            
            if conf_values:  # Only plot if there are values
                y_positions = [i] * len(conf_values)
                # Add some jitter to y positions
                y_jitter = np.random.normal(0, 0.1, len(conf_values))
                
                ax.scatter([c for c in conf_values], 
                          [y + jitter for y, jitter in zip(y_positions, y_jitter)],
                          c=self.current_palette[sent_type] if sent_type in self.current_palette else colors[0],
                          alpha=0.6, s=50, label=sent_type.capitalize())
        
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Sentiment Category')
        ax.set_yticks(range(len(sentiment_types)))
        ax.set_yticklabels([s.capitalize() for s in sentiment_types])
        ax.set_title('Sentiment Confidence Distribution', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            self.logger.info(f"Sentiment confidence scatter saved: {save_path}")
        
        plt.show()
        return save_path or ""
    
    def create_text_length_analysis(self, batch_result: BatchAnalysisResult, 
                                   save_path: str = None) -> str:
        """
        Analyze text lengths and their relationship to sentiment
        
        Args:
            batch_result: Analysis results
            save_path: Path to save the chart
            
        Returns:
            Path to saved chart
        """
        # Extract data
        text_lengths = [len(result.original_text) for result in batch_result.results]
        sentiments = [result.sentiment.overall_sentiment for result in batch_result.results]
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'text_length': text_lengths,
            'sentiment': sentiments
        })
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of text lengths by sentiment
        sentiment_order = ['negative', 'neutral', 'positive']
        colors = [self.current_palette['negative'], self.current_palette['neutral'], 
                 self.current_palette['positive']]
        
        box_plot = ax1.boxplot([df[df['sentiment'] == s]['text_length'].values 
                               for s in sentiment_order],
                              labels=[s.capitalize() for s in sentiment_order],
                              patch_artist=True)
        
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Text Length (characters)')
        ax1.set_title('Text Length Distribution by Sentiment', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of text lengths
        ax2.hist(text_lengths, bins=30, color=self.current_palette['primary'], 
                alpha=0.7, edgecolor='white')
        ax2.axvline(np.mean(text_lengths), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(text_lengths):.0f} chars')
        ax2.axvline(np.median(text_lengths), color='orange', linestyle='--', 
                   label=f'Median: {np.median(text_lengths):.0f} chars')
        ax2.set_xlabel('Text Length (characters)')
        ax2.set_ylabel('Number of Comments')
        ax2.set_title('Overall Text Length Distribution', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            self.logger.info(f"Text length analysis saved: {save_path}")
        
        plt.show()
        return save_path or ""
    
    def create_summary_metrics_dashboard(self, batch_result: BatchAnalysisResult, 
                                       save_path: str = None) -> str:
        """
        Create comprehensive metrics dashboard
        
        Args:
            batch_result: Analysis results
            save_path: Path to save the dashboard
            
        Returns:
            Path to saved dashboard
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
        
        # 1. Key Metrics (top row, spanning all columns)
        ax_metrics = fig.add_subplot(gs[0, :])
        ax_metrics.axis('off')
        
        # Create metric boxes
        metrics = [
            ('Total Comments', f"{batch_result.total_comments:,}"),
            ('Processing Time', f"{batch_result.total_processing_time:.1f}s"),
            ('Success Rate', f"{batch_result.processing_stats['success_rate']:.1f}%"),
            ('Avg Confidence', f"{batch_result.processing_stats['avg_sentiment_confidence']:.3f}"),
        ]
        
        for i, (label, value) in enumerate(metrics):
            x_pos = 0.1 + i * 0.2
            # Background box
            bbox = dict(boxstyle="round,pad=0.3", facecolor=self.current_palette['secondary'], alpha=0.3)
            ax_metrics.text(x_pos, 0.7, label, transform=ax_metrics.transAxes, 
                           fontsize=12, ha='center', fontweight='bold')
            ax_metrics.text(x_pos, 0.3, value, transform=ax_metrics.transAxes, 
                           fontsize=16, ha='center', fontweight='bold', 
                           color=self.current_palette['primary'], bbox=bbox)
        
        # 2. Sentiment pie chart
        ax_sentiment = fig.add_subplot(gs[1, 0])
        sentiment_dist = batch_result.sentiment_distribution
        colors = [self.current_palette['positive'], self.current_palette['negative'], 
                 self.current_palette['neutral']]
        wedges, texts, autotexts = ax_sentiment.pie(
            sentiment_dist.values(),
            labels=[s.capitalize() for s in sentiment_dist.keys()],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax_sentiment.set_title('Sentiment Distribution', fontweight='bold')
        
        # 3. Top keywords bar chart
        ax_keywords = fig.add_subplot(gs[1, 1])
        top_keywords = batch_result.global_wordcloud.top_keywords[:8]
        words, frequencies = zip(*top_keywords) if top_keywords else ([], [])
        
        if words:
            bars = ax_keywords.barh(range(len(words)), frequencies, 
                                   color=self.current_palette['primary'], alpha=0.7)
            ax_keywords.set_yticks(range(len(words)))
            ax_keywords.set_yticklabels(words)
            ax_keywords.set_xlabel('Frequency')
            ax_keywords.set_title('Top Keywords', fontweight='bold')
            ax_keywords.invert_yaxis()
        
        # 4. Processing time distribution
        ax_time = fig.add_subplot(gs[1, 2])
        processing_times = [result.processing_time for result in batch_result.results]
        ax_time.hist(processing_times, bins=15, color=self.current_palette['secondary'], 
                    alpha=0.7, edgecolor='white')
        ax_time.set_xlabel('Processing Time (s)')
        ax_time.set_ylabel('Count')
        ax_time.set_title('Processing Times', fontweight='bold')
        
        # 5. Summary compression ratios
        ax_compression = fig.add_subplot(gs[2, 0])
        compression_ratios = [result.summary.compression_ratio for result in batch_result.results]
        ax_compression.hist(compression_ratios, bins=15, color=self.current_palette['primary'], 
                           alpha=0.7, edgecolor='white')
        ax_compression.set_xlabel('Compression Ratio')
        ax_compression.set_ylabel('Count')
        ax_compression.set_title('Summary Compression', fontweight='bold')
        
        # 6. Global summary box
        ax_summary = fig.add_subplot(gs[2, 1:])
        ax_summary.axis('off')
        
        summary_text = batch_result.global_summary.summary
        if len(summary_text) > 400:  # Truncate long summaries
            summary_text = summary_text[:400] + "..."
        
        ax_summary.text(0.05, 0.95, "Global Summary:", transform=ax_summary.transAxes, 
                       fontsize=12, fontweight='bold', va='top')
        ax_summary.text(0.05, 0.85, summary_text, transform=ax_summary.transAxes, 
                       fontsize=10, va='top', wrap=True,
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                                edgecolor=self.current_palette['primary'], alpha=0.8))
        
        plt.suptitle('eConsultation AI - Analysis Dashboard', 
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            self.logger.info(f"Dashboard saved: {save_path}")
        
        plt.show()
        return save_path or ""
    
    def create_interactive_dashboard(self, batch_result: BatchAnalysisResult, 
                                   save_path: str = None) -> str:
        """
        Create interactive HTML dashboard using Plotly
        
        Args:
            batch_result: Analysis results
            save_path: Path to save the HTML file
            
        Returns:
            Path to saved HTML file
        """
        if not PLOTLY_AVAILABLE:
            self.logger.warning("Plotly not available. Cannot create interactive dashboard.")
            return ""
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Sentiment Distribution', 'Top Keywords',
                          'Processing Times', 'Text Length vs Sentiment',
                          'Confidence Distribution', 'Summary Statistics'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "histogram"}, {"type": "box"}],
                   [{"type": "violin"}, {"type": "table"}]]
        )
        
        # 1. Sentiment pie chart
        sentiment_dist = batch_result.sentiment_distribution
        fig.add_trace(
            go.Pie(labels=list(sentiment_dist.keys()), 
                   values=list(sentiment_dist.values()),
                   name="Sentiment"),
            row=1, col=1
        )
        
        # 2. Top keywords bar chart
        top_keywords = batch_result.global_wordcloud.top_keywords[:10]
        words, frequencies = zip(*top_keywords) if top_keywords else ([], [])
        fig.add_trace(
            go.Bar(x=list(frequencies), y=list(words), 
                   orientation='h', name="Keywords"),
            row=1, col=2
        )
        
        # 3. Processing times histogram
        processing_times = [result.processing_time for result in batch_result.results]
        fig.add_trace(
            go.Histogram(x=processing_times, name="Processing Times"),
            row=2, col=1
        )
        
        # 4. Text length by sentiment box plot
        text_lengths = [len(result.original_text) for result in batch_result.results]
        sentiments = [result.sentiment.overall_sentiment for result in batch_result.results]
        
        for sentiment in set(sentiments):
            lengths = [length for length, sent in zip(text_lengths, sentiments) if sent == sentiment]
            fig.add_trace(
                go.Box(y=lengths, name=sentiment, boxpoints='outliers'),
                row=2, col=2
            )
        
        # 5. Confidence distribution violin plot
        confidences = [result.sentiment.confidence for result in batch_result.results]
        sentiments = [result.sentiment.overall_sentiment for result in batch_result.results]
        
        for sentiment in set(sentiments):
            conf_vals = [conf for conf, sent in zip(confidences, sentiments) if sent == sentiment]
            fig.add_trace(
                go.Violin(y=conf_vals, name=sentiment, box_visible=True),
                row=3, col=1
            )
        
        # 6. Summary statistics table
        stats_data = [
            ['Total Comments', batch_result.total_comments],
            ['Processing Time', f"{batch_result.total_processing_time:.2f}s"],
            ['Success Rate', f"{batch_result.processing_stats['success_rate']:.1f}%"],
            ['Avg Confidence', f"{batch_result.processing_stats['avg_sentiment_confidence']:.3f}"],
            ['Unique Keywords', batch_result.global_wordcloud.unique_words],
            ['Compression Ratio', f"{batch_result.global_summary.compression_ratio:.3f}"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=[[row[0] for row in stats_data], 
                                  [row[1] for row in stats_data]])
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="eConsultation AI - Interactive Analysis Dashboard",
            showlegend=True,
            height=1200
        )
        
        # Save to HTML
        if save_path:
            pyo.plot(fig, filename=save_path, auto_open=False)
            self.logger.info(f"Interactive dashboard saved: {save_path}")
        
        return save_path or ""
    
    def generate_comprehensive_report(self, batch_result: BatchAnalysisResult, 
                                    output_dir: str) -> Dict[str, str]:
        """
        Generate all visualizations and reports
        
        Args:
            batch_result: Analysis results
            output_dir: Directory to save all outputs
            
        Returns:
            Dictionary mapping report type to file path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        reports = {}
        
        try:
            # 1. Sentiment distribution chart
            reports['sentiment_distribution'] = self.create_sentiment_distribution_chart(
                batch_result, str(output_path / "sentiment_distribution.png")
            )
            
            # 2. Processing timeline
            reports['processing_timeline'] = self.create_processing_timeline(
                batch_result, str(output_path / "processing_timeline.png")
            )
            
            # 3. Sentiment confidence scatter
            reports['sentiment_confidence'] = self.create_sentiment_confidence_scatter(
                batch_result, str(output_path / "sentiment_confidence.png")
            )
            
            # 4. Text length analysis
            reports['text_length_analysis'] = self.create_text_length_analysis(
                batch_result, str(output_path / "text_length_analysis.png")
            )
            
            # 5. Summary metrics dashboard
            reports['metrics_dashboard'] = self.create_summary_metrics_dashboard(
                batch_result, str(output_path / "metrics_dashboard.png")
            )
            
            # 6. Interactive dashboard (if Plotly available)
            if PLOTLY_AVAILABLE:
                reports['interactive_dashboard'] = self.create_interactive_dashboard(
                    batch_result, str(output_path / "interactive_dashboard.html")
                )
            
            self.logger.info(f"Comprehensive report generated in {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            raise
        
        return reports

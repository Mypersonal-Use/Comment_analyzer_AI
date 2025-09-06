from flask import Flask, request, jsonify, render_template_string
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from econsultation_ai import EConsultationAI

app = Flask(__name__)

# Initialize AI system
config = {
    'sentiment_methods': ['vader', 'textblob'],
    'default_sentiment_method': 'vader',
    'default_summarization_method': 'textrank',
    'max_summary_sentences': 3,
    'wordcloud_language': 'english',
}

ai_system = EConsultationAI(config=config)

@app.route('/')
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>eConsultation AI</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .header { text-align: center; color: #1e3a8a; }
            .form { background: #f8f9fa; padding: 20px; border-radius: 10px; margin: 20px 0; }
            textarea { width: 100%; height: 150px; padding: 10px; border-radius: 5px; border: 1px solid #ccc; }
            button { background: #3b82f6; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; }
            .result { background: #e7f3ff; padding: 15px; border-radius: 5px; margin: 10px 0; }
        </style>
    </head>
    <body>
        <h1 class="header">🏛️ eConsultation AI</h1>
        <p style="text-align: center;">AI-powered analysis for stakeholder comments</p>
        
        <div class="form">
            <h3>Analyze Your Comment</h3>
            <textarea id="comment" placeholder="Enter your comment here..."></textarea>
            <br><br>
            <button onclick="analyzeComment()">🔍 Analyze Comment</button>
        </div>
        
        <div id="results"></div>
        
        <script>
        async function analyzeComment() {
            const comment = document.getElementById('comment').value;
            if (!comment.trim()) {
                alert('Please enter a comment to analyze');
                return;
            }
            
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: comment})
            });
            
            const result = await response.json();
            
            document.getElementById('results').innerHTML = `
                <div class="result">
                    <h3>📊 Analysis Results</h3>
                    <p><strong>Sentiment:</strong> ${result.sentiment} (${(result.confidence * 100).toFixed(1)}% confidence)</p>
                    <p><strong>Summary:</strong> ${result.summary}</p>
                    <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(3)} seconds</p>
                </div>
            `;
        }
        </script>
    </body>
    </html>
    ''')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400
        
        result = ai_system.analyze_single_comment(text, 'web_comment')
        
        return jsonify({
            'sentiment': result.sentiment.overall_sentiment,
            'confidence': result.sentiment.confidence,
            'summary': result.summary.summary,
            'processing_time': result.processing_time
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Vercel serverless function handler
def handler(event, context):
    return app(event, context)

if __name__ == '__main__':
    app.run(debug=True)

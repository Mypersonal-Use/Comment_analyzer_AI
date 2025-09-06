# eConsultation AI - Deployment Guide

## 🚀 Deploy to Vercel

### Quick Deployment Steps:

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - eConsultation AI"
   git remote add origin https://github.com/YOUR_USERNAME/econsultation-ai
   git push -u origin main
   ```

2. **Deploy to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub
   - Click "New Project"
   - Import your `econsultation-ai` repository
   - Vercel will automatically detect the configuration

### Manual Deployment Configuration:

If automatic detection doesn't work:
- **Build Command**: Leave empty
- **Output Directory**: Leave empty  
- **Install Command**: `pip install -r requirements_web.txt`
- **Framework Preset**: Other

### Environment Variables (if needed):
- Set `PYTHONPATH` to `./src`

## 🖥️ Run Locally

### Web Application:
```bash
# Install web dependencies
pip install -r requirements_web.txt

# Run Streamlit app
streamlit run app.py
```

### Desktop Application:
```bash
# Install all dependencies
pip install -r requirements.txt

# Run examples
python examples/basic_usage.py
python examples/advanced_usage.py

# Use CLI
python src/main.py single "Your comment here"
python src/main.py file data/sample_comments.csv --output results/
```

## 📁 Project Structure

```
econsultation-ai/
├── app.py                 # Web application (Streamlit)
├── src/                   # Core AI modules
│   ├── econsultation_ai.py
│   ├── sentiment_analyzer.py
│   ├── text_summarizer.py
│   ├── wordcloud_generator.py
│   └── data_processor.py
├── examples/              # Usage examples
├── data/                  # Sample data
├── results/               # Generated outputs
├── requirements.txt       # Desktop dependencies
├── requirements_web.txt   # Web dependencies
└── vercel.json           # Vercel configuration
```

## 🌐 Features

### Web Interface:
- ✅ Single comment analysis
- ✅ Multiple comments batch processing
- ✅ File upload (CSV, JSON, TXT, Excel)
- ✅ Interactive sentiment analysis
- ✅ Keyword visualizations
- ✅ Downloadable reports

### Desktop Interface:
- ✅ Command-line interface
- ✅ Python API
- ✅ Batch processing
- ✅ Multiple file formats
- ✅ Advanced reporting

## 🔧 Troubleshooting

### Common Issues:

1. **NLTK Data Missing**:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

2. **Memory Issues on Vercel**:
   - Limit file sizes to < 10MB
   - Process files in smaller batches

3. **Display Issues**:
   - Disable matplotlib GUI: `plt.switch_backend('Agg')`

## 📊 Sample Usage

### Upload a CSV file with columns:
- `id` - Comment identifier
- `text` - Comment content
- `author` - Author name (optional)
- `category` - Category (optional)
- `timestamp` - Date/time (optional)

### Expected Output:
- Sentiment distribution (Positive/Negative/Neutral percentages)
- Global summary of all comments
- Top keywords with frequencies
- Individual comment analysis
- Downloadable reports

## 🎯 Use Cases

- **Government Consultations**: Analyze public feedback on policy changes
- **Corporate Governance**: Process stakeholder comments on amendments
- **Public Surveys**: Understand sentiment in large-scale feedback
- **Legislative Reviews**: Summarize consultation responses

---

**Ready to deploy!** 🚀 Follow the steps above to get your eConsultation AI live on the web.

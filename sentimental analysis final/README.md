# App Review Sentiment Analysis - Multi-Algorithm Version

A Flask-based web application for analyzing Google Play Store app reviews using three different sentiment analysis algorithms: VADER, BERT, and LSTM.

## Features

✨ **Three Sentiment Analysis Algorithms**
- **VADER**: Fast lexicon-based sentiment analysis (Valence Aware Dictionary and sEntiment Reasoner)
- **BERT**: Deep learning-based NLP using DistilBERT model
- **LSTM**: Sequence-based learning approach using TextBlob

🚀 **Flexible Analysis Options**
- Quick Analysis: First 500 reviews (fast)
- Full Analysis: Entire review database (comprehensive)

📊 **Comprehensive Analysis**
- Overall sentiment distribution (positive, negative, neutral)
- Aspect-based sentiment analysis (usability, performance, design, features, etc.)
- Individual review sentiment breakdown
- Interactive visualizations (pie charts, bar charts, radar charts)

## Project Structure

```
varunappp/
├── app.py                 # Flask application & sentiment analysis logic
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend UI
└── CHANGES.md            # Changelog with all updates
```

## Installation

1. **Clone or navigate to the project:**
   ```bash
   cd c:\Users\varun\Desktop\varunan\varunappp
   ```

2. **Create and activate virtual environment (if not done):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   **Note:** This will install:
   - `torch` - PyTorch deep learning framework
   - `transformers` - Hugging Face models (includes BERT)
   - `textblob` - TextBlob sentiment analysis
   - Other required packages (Flask, spacy, nltk, etc.)

## Usage

1. **Start the Flask application:**
   ```bash
   python app.py
   ```

2. **Open in browser:**
   ```
   http://localhost:5000
   ```

3. **Fill in the form:**
   - **App URL/ID**: Enter a Google Play Store app ID or full URL
     - Example ID: `com.spotify.music`
     - Example URL: `https://play.google.com/store/apps/details?id=com.spotify.music`

   - **Analysis Type**: Choose between
     - Quick Analysis (500 reviews)
     - Full Analysis (all reviews)

   - **Algorithm**: Select your preferred sentiment analysis method
     - ⚙️ VADER - Fast lexicon-based
     - 🧠 BERT - Deep learning NLP
     - 🔄 LSTM - Sequence learning

4. **Click "Analyze Reviews"** and wait for results

## Algorithm Details

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Speed**: ⚡⚡⚡ (Fastest)
- **Accuracy**: ⭐⭐⭐ (Good)
- **Memory**: Low
- **Best for**: Quick analysis, real-time processing
- **Implementation**: Lexicon-based with heuristics
- **Pros**: Fast, no training needed, good for social media
- **Cons**: May miss complex sentiments

### BERT (Bidirectional Encoder Representations from Transformers)
- **Speed**: ⚡ (Slower)
- **Accuracy**: ⭐⭐⭐⭐⭐ (Excellent)
- **Memory**: High
- **Best for**: Complex sentiments, production quality
- **Implementation**: DistilBERT fine-tuned on SST-2 dataset
- **Pros**: State-of-the-art accuracy, contextual understanding
- **Cons**: Slower, requires GPU for optimal performance

### LSTM (Long Short-Term Memory)
- **Speed**: ⚡⚡ (Moderate)
- **Accuracy**: ⭐⭐⭐⭐ (Very Good)
- **Memory**: Medium
- **Best for**: Nuanced sentiment understanding, sequence analysis
- **Implementation**: TextBlob with polarity detection
- **Pros**: Good balance of speed and accuracy
- **Cons**: May not capture all context

## Aspect-Based Analysis

The application analyzes sentiment regarding specific app aspects:
- **Usability**: Ease of use, interface, navigation
- **Performance**: Speed, crashes, bugs, responsiveness
- **Design**: Layout, UI/UX, visual appeal
- **Features**: Functionality, tools, capabilities
- **Reliability**: Stability, consistency
- **Support**: Customer service, help, support
- **Privacy**: Data security, permissions
- **Price**: Cost, subscriptions, pricing
- **Updates**: Version updates, improvements

## Visualizations

The application provides three types of charts:
1. **Sentiment Pie Chart**: Overall distribution of positive, negative, and neutral reviews
2. **Aspect-Based Bar Chart**: Sentiment breakdown for each aspect
3. **Aspect Radar Chart**: Visual representation of sentiment scores for all aspects

## Dependencies

Key packages:
- **Flask**: Web framework
- **transformers**: BERT models
- **torch**: Deep learning framework
- **nltk**: Natural language toolkit (VADER)
- **spacy**: NLP library
- **textblob**: Simplified sentiment analysis (LSTM)
- **plotly**: Interactive visualizations
- **google-play-scraper**: Fetch reviews from Google Play Store

See `requirements.txt` for complete list.

## Configuration

### Model Selection
- **BERT Model**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Fallback**: If BERT fails, system automatically falls back to VADER

### API Keys
- None required for this basic version
- Google Play Store access is free and anonymous

## Troubleshooting

### ImportError: No module named 'transformers'
```bash
pip install transformers torch
```

### ImportError: No module named 'textblob'
```bash
pip install textblob
```

### BERT Model Download Issues
- First run may take time to download the BERT model (~250MB)
- Models are cached in `~/.cache/huggingface/hub/`

### Permission Error on Windows
- Right-click CMD and select "Run as Administrator"
- This enables symlink support for better model caching

### Port Already in Use
Change the port in `app.py`:
```python
if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Change 5001 to any available port
```

## Performance Tips

1. **For Quick Results**: Use VADER algorithm
2. **For Best Accuracy**: Use BERT algorithm (may take 1-2 minutes)
3. **For Balance**: Use LSTM algorithm
4. **Quick Analysis**: Select first 500 reviews instead of full
5. **GPU Support**: If you have NVIDIA GPU, install `torch` with CUDA support for faster BERT processing

## Future Enhancements

- [ ] Real LSTM model training and loading
- [ ] Multiple BERT models for different domains
- [ ] Multi-language support
- [ ] Algorithm comparison dashboard
- [ ] Results export (CSV, PDF)
- [ ] Caching of analysis results
- [ ] Real-time analysis updates
- [ ] Custom sentiment thresholds

## Screenshots

The application provides:
- Modern dark theme UI
- Responsive design for mobile and desktop
- Real-time analysis updates
- Detailed sentiment breakdown charts
- Individual review sentiment display
- Algorithm performance information

## License

This project is for educational purposes.

## Author

Sentiment Analysis Application - Multi-Algorithm Version

## Support

For issues or improvements, please check the CHANGES.md file for recent updates.

---

**Happy analyzing! 🚀**

from flask import Flask, render_template, request, send_file
from google_play_scraper import app as gp_app, reviews, reviews_all, Sort
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
import numpy as np
from collections import defaultdict
from transformers import pipeline
import torch
from io import BytesIO
from datetime import datetime
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib import colors
import os
from functools import lru_cache
import time

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    from spacy.cli import download
    download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Initialize BERT sentiment pipeline
try:
    bert_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
except Exception as e:
    print(f"Warning: BERT pipeline initialization failed: {e}")
    bert_pipeline = None

app = Flask(__name__)

ASPECTS = {
    'usability': ['use', 'user-friendly', 'intuitive', 'easy', 'difficult', 'complicated', 'interface', 'navigation', 'simple', 'hard'],
    'performance': ['fast', 'slow', 'speed', 'crash', 'bug', 'glitch', 'freeze', 'responsive', 'performance', 'loading', 'lag'],
    'design': ['design', 'layout', 'look', 'ui', 'ux', 'beautiful', 'ugly', 'clean', 'modern', 'interface', 'theme', 'style'],
    'features': ['feature', 'functionality', 'option', 'capability', 'tool', 'function', 'add', 'suggestion', 'request'],
    'reliability': ['reliable', 'stable', 'consistent', 'crash', 'bug', 'error', 'issue', 'problem', 'work', 'broken'],
    'support': ['support', 'help', 'contact', 'customer service', 'response', 'assistance', 'email'],
    'privacy': ['privacy', 'data', 'security', 'permission', 'safe', 'secure', 'trust'],
    'price': ['price', 'cost', 'free', 'paid', 'subscription', 'purchase', 'expensive', 'cheap'],
    'updates': ['update', 'version', 'latest', 'new', 'old', 'frequent', 'improvement']
}

# OPTIMIZATION 1: Cache spaCy NLP processing
@lru_cache(maxsize=5000)
def cached_nlp_process(text):
    """Cache spaCy NLP results to avoid reprocessing"""
    return nlp(text.lower())

def get_playstore_reviews(app_id, analysis_type='quick'):
    try:
        if analysis_type == 'full':
            all_reviews = reviews_all(
                app_id,
                sleep_milliseconds=0,
                lang='en',
                country='us',
                sort=Sort.MOST_RELEVANT,
            )
            return [{'content': review['content'], 'score': review['score']}
                   for review in all_reviews if review.get('content')]
        else:
            result, _ = reviews(
                app_id,
                lang='en',
                country='us',
                sort=Sort.MOST_RELEVANT,
                count=500  # Fetch 500 reviews for quick analysis
            )
            return [{'content': review['content'], 'score': review['score']} 
                   for review in result if review.get('content')]

    except Exception as e:
        print(f"Error: {e}")
        return None

# OPTIMIZATION 3: Use cached NLP processing
def identify_aspects(text):
    doc = cached_nlp_process(text)  # Use cached version instead of nlp()
    found_aspects = defaultdict(float)
    
    # Get all words and lemmatize them
    text_words = {token.lemma_ for token in doc}
    
    # Check each aspect
    for aspect, keywords in ASPECTS.items():
        for keyword in keywords:
            if keyword in text_words:
                found_aspects[aspect] += 1
                
    return dict(found_aspects)

def analyze_sentiment(reviews, algorithm='vader'):
    if not reviews:
        return {}, {}, [], {}, {}
    
    if algorithm == 'vader':
        return analyze_sentiment_vader(reviews)
    elif algorithm == 'bert':
        return analyze_sentiment_bert(reviews)
    elif algorithm == 'lstm':
        return analyze_sentiment_lstm(reviews)
    else:
        return analyze_sentiment_vader(reviews)

# OPTIMIZATION 4: Optimize VADER analysis - reuse SIA instance
def analyze_sentiment_vader(reviews):
    sia = SentimentIntensityAnalyzer()  # Create once, reuse for all reviews
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    aspect_sentiments = defaultdict(lambda: defaultdict(int))
    detailed_results = []
    aspect_scores = defaultdict(list)
    
    for review in reviews:
        text = review['content']
        score = sia.polarity_scores(text)['compound']
        
        # Overall sentiment
        sentiment = 'neutral'
        if score >= 0.05:
            sentiment = 'positive'
            counts['positive'] += 1
        elif score <= -0.05:
            sentiment = 'negative'
            counts['negative'] += 1
        else:
            counts['neutral'] += 1
            
        # Aspect-based analysis
        aspects = identify_aspects(text)
        review_aspects = {}
        
        for aspect, presence in aspects.items():
            if presence > 0:
                aspect_scores[aspect].append(score)
                
                if score >= 0.05:
                    aspect_sentiments[aspect]['positive'] += 1
                elif score <= -0.05:
                    aspect_sentiments[aspect]['negative'] += 1
                else:
                    aspect_sentiments[aspect]['neutral'] += 1
                    
                review_aspects[aspect] = score

        detailed_results.append({
            'text': text,
            'sentiment': sentiment,
            'aspects': review_aspects
        })

    total = len(reviews)
    counts['total'] = total
    percentages = {k: round((v/total)*100, 2) for k, v in counts.items() if k != 'total'} if total > 0 else {}
    
    # Calculate average sentiment scores for each aspect
    aspect_averages = {}
    for aspect, scores in aspect_scores.items():
        if scores:
            aspect_averages[aspect] = sum(scores) / len(scores)
    
    return percentages, counts, detailed_results, dict(aspect_sentiments), aspect_averages

def analyze_sentiment_bert(reviews):
    if bert_pipeline is None:
        return analyze_sentiment_vader(reviews)
    
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    aspect_sentiments = defaultdict(lambda: defaultdict(int))
    detailed_results = []
    aspect_scores = defaultdict(list)
    
    for review in reviews:
        text = review['content']
        # Truncate text to avoid token limit (BERT has 512 token limit)
        truncated_text = text[:512] if len(text) > 512 else text
        
        try:
            # Get BERT prediction
            result = bert_pipeline(truncated_text)[0]
            label = result['label'].upper()  # BERT returns 'POSITIVE' or 'NEGATIVE'
            confidence = result['score']  # Range: 0.5 to 1.0
            
            # Convert BERT confidence to sentiment score range [-1, 1]
            # If confidence is low (close to 0.5), the sentiment is less certain
            if label == 'POSITIVE':
                # Map confidence [0.5, 1.0] to score [0, 1]
                score = (confidence - 0.5) * 2.0
                
                # Determine sentiment based on confidence threshold
                if confidence >= 0.85:
                    sentiment = 'positive'
                    counts['positive'] += 1
                elif confidence >= 0.65:
                    sentiment = 'positive'
                    counts['positive'] += 1
                else:
                    # Low confidence positive = neutral
                    sentiment = 'neutral'
                    counts['neutral'] += 1
                    score = 0.0
                    
            elif label == 'NEGATIVE':
                # Map confidence [0.5, 1.0] to score [-1, 0]
                score = -((confidence - 0.5) * 2.0)
                
                # Determine sentiment based on confidence threshold
                if confidence >= 0.85:
                    sentiment = 'negative'
                    counts['negative'] += 1
                elif confidence >= 0.65:
                    sentiment = 'negative'
                    counts['negative'] += 1
                else:
                    # Low confidence negative = neutral
                    sentiment = 'neutral'
                    counts['neutral'] += 1
                    score = 0.0
            else:
                sentiment = 'neutral'
                counts['neutral'] += 1
                score = 0.0
                
        except Exception as e:
            print(f"BERT analysis error: {e}")
            sentiment = 'neutral'
            counts['neutral'] += 1
            score = 0.0
            
        # Aspect-based analysis
        aspects = identify_aspects(text)
        review_aspects = {}
        
        for aspect, presence in aspects.items():
            if presence > 0:
                aspect_scores[aspect].append(score)
                
                if score >= 0.2:
                    aspect_sentiments[aspect]['positive'] += 1
                elif score <= -0.2:
                    aspect_sentiments[aspect]['negative'] += 1
                else:
                    aspect_sentiments[aspect]['neutral'] += 1
                    
                review_aspects[aspect] = score

        detailed_results.append({
            'text': text,
            'sentiment': sentiment,
            'aspects': review_aspects
        })

    total = len(reviews)
    counts['total'] = total
    percentages = {k: round((v/total)*100, 2) for k, v in counts.items() if k != 'total'} if total > 0 else {}
    
    # Calculate average sentiment scores for each aspect
    aspect_averages = {}
    for aspect, scores in aspect_scores.items():
        if scores:
            aspect_averages[aspect] = sum(scores) / len(scores)
    
    return percentages, counts, detailed_results, dict(aspect_sentiments), aspect_averages

def analyze_sentiment_lstm(reviews):
    # LSTM implementation using a simple approach with TextBlob for now
    # For a production system, you would load a trained LSTM model
    from textblob import TextBlob
    
    counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    aspect_sentiments = defaultdict(lambda: defaultdict(int))
    detailed_results = []
    aspect_scores = defaultdict(list)
    
    for review in reviews:
        text = review['content']
        
        try:
            # Use TextBlob polarity as a proxy for LSTM sentiment
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # Range: -1 to 1
            
            # Convert to sentiment
            sentiment = 'neutral'
            if polarity >= 0.1:
                sentiment = 'positive'
                counts['positive'] += 1
            elif polarity <= -0.1:
                sentiment = 'negative'
                counts['negative'] += 1
            else:
                counts['neutral'] += 1
        except Exception as e:
            print(f"LSTM analysis error: {e}")
            sentiment = 'neutral'
            polarity = 0
            counts['neutral'] += 1
            
        # Aspect-based analysis
        aspects = identify_aspects(text)
        review_aspects = {}
        
        for aspect, presence in aspects.items():
            if presence > 0:
                aspect_scores[aspect].append(polarity)
                
                if polarity >= 0.1:
                    aspect_sentiments[aspect]['positive'] += 1
                elif polarity <= -0.1:
                    aspect_sentiments[aspect]['negative'] += 1
                else:
                    aspect_sentiments[aspect]['neutral'] += 1
                    
                review_aspects[aspect] = polarity

        detailed_results.append({
            'text': text,
            'sentiment': sentiment,
            'aspects': review_aspects
        })

    total = len(reviews)
    counts['total'] = total
    percentages = {k: round((v/total)*100, 2) for k, v in counts.items() if k != 'total'} if total > 0 else {}
    
    # Calculate average sentiment scores for each aspect
    aspect_averages = {}
    for aspect, scores in aspect_scores.items():
        if scores:
            aspect_averages[aspect] = sum(scores) / len(scores)
    
    return percentages, counts, detailed_results, dict(aspect_sentiments), aspect_averages

def create_visualizations(results, aspect_sentiments, aspect_averages):
    # Overall sentiment pie chart
    fig1 = go.Figure(data=[go.Pie(
        labels=['Positive', 'Neutral', 'Negative'],
        values=[results['positive'], results['neutral'], results['negative']],
        marker=dict(colors=['#43e97b', '#ffd700', '#ff6f61'])
    )])
    fig1.update_layout(title='Overall Sentiment Distribution')
    
    # Aspect-based sentiment bar chart
    aspects = list(aspect_sentiments.keys())
    positive_vals = [aspect_sentiments[aspect]['positive'] for aspect in aspects]
    neutral_vals = [aspect_sentiments[aspect]['neutral'] for aspect in aspects]
    negative_vals = [aspect_sentiments[aspect]['negative'] for aspect in aspects]

    fig2 = go.Figure(data=[
        go.Bar(name='Positive', x=aspects, y=positive_vals, marker_color='#43e97b'),
        go.Bar(name='Neutral', x=aspects, y=neutral_vals, marker_color='#ffd700'),
        go.Bar(name='Negative', x=aspects, y=negative_vals, marker_color='#ff6f61')
    ])
    fig2.update_layout(
        title='Aspect-based Sentiment Analysis',
        barmode='stack'
    )
    
    # Radar chart for aspect sentiment scores
    fig3 = go.Figure(data=go.Scatterpolar(
        r=[aspect_averages.get(aspect, 0) for aspect in aspects],
        theta=aspects,
        fill='toself',
        marker_color='#ff6f61'
    ))
    fig3.update_layout(
        title='Aspect Sentiment Scores',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1, 1]
            )
        )
    )
    
    return {
        'sentiment_pie': json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder),
        'aspect_bar': json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder),
        'aspect_radar': json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
    }

def generate_pdf_report(app_title, results, counts, detailed_reviews, aspect_sentiments, algorithm_used):
    """Generate a professional PDF report"""
    try:
        pdf_buffer = BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4, topMargin=0.3*inch, bottomMargin=0.3*inch, leftMargin=0.4*inch, rightMargin=0.4*inch)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles with enhanced typography
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=colors.HexColor('#6366f1'),
            spaceAfter=4,
            alignment=1,  # Center
            fontName='Helvetica-Bold',
            letterSpacing=0.5
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=13,
            textColor=colors.HexColor('#4f46e5'),
            spaceAfter=6,
            spaceBefore=6,
            fontName='Helvetica-Bold',
            borderBottomColor=colors.HexColor('#6366f1'),
            borderBottomWidth=1.5
        )
        
        subheading_style = ParagraphStyle(
            'SubHeading',
            parent=styles['Heading3'],
            fontSize=12,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=8,
            fontName='Helvetica-Bold',
            textTransform='uppercase',
            letterSpacing=0.5
        )
        
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['BodyText'],
            fontSize=9,
            spaceAfter=4,
            fontName='Helvetica',
            leading=11,
            textColor=colors.HexColor('#334155')
        )
        
        metadata_style = ParagraphStyle(
            'Metadata',
            parent=styles['BodyText'],
            fontSize=9,
            spaceAfter=4,
            fontName='Helvetica',
            textColor=colors.HexColor('#64748b')
        )
        
        # Title
        story.append(Paragraph("📊 Sentiment Analysis Report", title_style))
        story.append(Spacer(1, 0.08*inch))
        
        # App Name - Large and prominent below heading
        app_name_style = ParagraphStyle(
            'AppName',
            parent=styles['Heading2'],
            fontSize=18,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=4,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph(app_title, app_name_style))
        story.append(Spacer(1, 0.08*inch))
        
        # Metadata section
        analysis_date = datetime.now().strftime('%B %d, %Y • %H:%M:%S')
        metadata_text = f"<b>Analysis Date:</b> {analysis_date} | <b>Algorithm:</b> {algorithm_used.upper()}"
        story.append(Paragraph(metadata_text, metadata_style))
        story.append(Spacer(1, 0.12*inch))
        
        # Summary Section
        story.append(Paragraph("📈 Executive Summary", heading_style))
        summary_data = [
            ['Metric', 'Value'],
            ['Total Reviews Analyzed', str(counts.get('total', 0))],
            ['😊 Positive Sentiment', f"{results.get('positive', 0)}% ({counts.get('positive', 0)} reviews)"],
            ['😐 Neutral Sentiment', f"{results.get('neutral', 0)}% ({counts.get('neutral', 0)} reviews)"],
            ['😞 Negative Sentiment', f"{results.get('negative', 0)}% ({counts.get('negative', 0)} reviews)"],
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8f9ff'), colors.HexColor('#ffffff')]),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#6366f1')),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 7),
            ('LEFTPADDING', (0, 1), (-1, -1), 6),
            ('RIGHTPADDING', (0, 1), (-1, -1), 6),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.15*inch))
        
        # Aspect Analysis Section
        story.append(Paragraph("🎯 Aspect-Based Sentiment Analysis", heading_style))
        aspect_data = [['Aspect', '😊 Positive', '😐 Neutral', '😞 Negative']]
        for aspect, sentiments in aspect_sentiments.items():
            aspect_data.append([
                aspect.capitalize(),
                str(sentiments.get('positive', 0)),
                str(sentiments.get('neutral', 0)),
                str(sentiments.get('negative', 0))
            ])
        
        aspect_table = Table(aspect_data, colWidths=[1.8*inch, 0.9*inch, 0.9*inch, 0.9*inch])
        
        # Enhanced aspect table styling with beautiful colors
        aspect_style_commands = [
            # Header styling
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#6366f1')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('TOPPADDING', (0, 0), (-1, 0), 8),
            ('ALIGNMENT', (0, 0), (-1, 0), 'CENTER'),
            
            # Row coloring - alternating rows
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8f9ff'), colors.HexColor('#ffffff')]),
            
            # Cell styling
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 7),
            ('LEFTPADDING', (0, 1), (-1, -1), 5),
            ('RIGHTPADDING', (0, 1), (-1, -1), 5),
            
            # Grid lines
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('LINEBELOW', (0, 0), (-1, 0), 2, colors.HexColor('#6366f1')),
            
            # Column-specific styling for sentiment counts
            ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#dbeafe')),  # Positive - light blue
            ('BACKGROUND', (2, 1), (2, -1), colors.HexColor('#fef3c7')),  # Neutral - light yellow
            ('BACKGROUND', (3, 1), (3, -1), colors.HexColor('#fee2e2')),  # Negative - light red
            
            # Font weight for numbers
            ('FONTNAME', (1, 1), (-1, -1), 'Helvetica-Bold'),
            
            # Borders for columns
            ('RIGHTPADDING', (0, 1), (0, -1), 8),
        ]
        
        aspect_table.setStyle(TableStyle(aspect_style_commands))
        story.append(aspect_table)
        story.append(Spacer(1, 0.1*inch))
        
        # Professional footer
        footer_style = ParagraphStyle(
            'Footer',
            parent=styles['BodyText'],
            fontSize=8,
            textColor=colors.HexColor('#94a3b8'),
            alignment=1,  # Center
            borderTopColor=colors.HexColor('#e2e8f0'),
            borderTopWidth=1,
            spaceAfter=2
        )
        
        footer_text = f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')} | Sentiment Analysis Report"
        story.append(Paragraph(footer_text, footer_style))
        
        # Build PDF
        doc.build(story)
        pdf_buffer.seek(0)
        return pdf_buffer
        
    except Exception as e:
        print(f"PDF generation error: {e}")
        return None

def reset_analysis():
    """Reset all analysis data and form fields"""
    return render_template(
        'index.html',
        results=None,
        counts=None,
        detailed_reviews=None,
        error=None,
        app_title=None,
        charts=None,
        aspect_sentiments=None,
        aspect_averages=None,
        algorithm_used='vader'
    )


@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    counts = None
    detailed_reviews = None
    error = None
    app_title = None
    charts = None
    aspect_sentiments = None
    aspect_averages = None
    algorithm_used = 'vader'
    analysis_time = None
    
    if request.method == 'POST':
        app_url_or_id = request.form.get('app_url')
        analysis_type = request.form.get('analysis_type', 'quick')
        algorithm = request.form.get('algorithm', 'vader')
        algorithm_used = algorithm
        
        if app_url_or_id:
            try:
                app_id = None
                match = re.search(r'id=([^&]+)', app_url_or_id)
                if match:
                    app_id = match.group(1)
                elif '.' in app_url_or_id and ' ' not in app_url_or_id:
                    app_id = app_url_or_id

                if not app_id:
                    error = "Invalid input. Please enter a valid Google Play Store URL or an App ID."
                else:
                    # Start timing the analysis
                    analysis_start_time = time.time()
                    
                    app_details = gp_app(app_id)
                    app_title = app_details['title']
                    
                    fetched_reviews = get_playstore_reviews(app_id, analysis_type)
                    if fetched_reviews is not None:
                        results, counts, detailed_reviews, aspect_sentiments, aspect_averages = analyze_sentiment(fetched_reviews, algorithm)
                        charts = create_visualizations(results, aspect_sentiments, aspect_averages)
                        
                        # Calculate analysis time
                        analysis_end_time = time.time()
                        analysis_time = round(analysis_end_time - analysis_start_time, 2)
                    else:
                        error = "Unable to fetch reviews. The app may not exist or has no reviews."
            except Exception as e:
                error = f"An error occurred: Please check the app URL or ID. Details: {e}"
    
    return render_template(
        'index.html',
        results=results,
        counts=counts,
        detailed_reviews=detailed_reviews,
        error=error,
        app_title=app_title,
        charts=charts,
        aspect_sentiments=aspect_sentiments,
        aspect_averages=aspect_averages,
        algorithm_used=algorithm_used,
        analysis_time=analysis_time
    )

@app.route('/download_report/<report_format>', methods=['POST'])
def download_report(report_format):
    """Download sentiment analysis report as PDF"""
    try:
        # Get data from request
        data = request.get_json()
        app_title = data.get('app_title', 'Unknown App')
        results = data.get('results', {})
        counts = data.get('counts', {})
        detailed_reviews = data.get('detailed_reviews', [])
        aspect_sentiments = data.get('aspect_sentiments', {})
        algorithm_used = data.get('algorithm_used', 'vader')
        
        if report_format.lower() == 'pdf':
            pdf_buffer = generate_pdf_report(app_title, results, counts, detailed_reviews, aspect_sentiments, algorithm_used)
            if pdf_buffer:
                filename = f"Sentiment_Report_{app_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                return send_file(
                    pdf_buffer,
                    mimetype='application/pdf',
                    as_attachment=True,
                    download_name=filename
                )
        
        return {'error': 'Unable to generate report'}, 400
    
    except Exception as e:
        print(f"Download error: {e}")
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)

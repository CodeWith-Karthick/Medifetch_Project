import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import matplotlib.pyplot as plt
from collections import Counter
import openai
from functools import lru_cache
import concurrent.futures
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from io import BytesIO

colors = ["red", "blue", "green","white"] 
# Initialize NLP tools (globally for reuse)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
sia = SentimentIntensityAnalyzer()

# Define preprocess_text function before it's used
def preprocess_text(text):
    """Clean and preprocess text for analysis"""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Dark theme matplotlib style
plt.style.use('dark_background')

# Cache expensive operations
@lru_cache(maxsize=32)
def format_drug_name(name):
    return name.lower().replace(" ", "-")

@lru_cache(maxsize=32)
def extract_drug2(drug1):
    url = f"https://www.drugs.com/{format_drug_name(drug1)}.html"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        more_about_header = soup.find(lambda tag: tag.name == "h2" and "More about" in tag.get_text())
        if more_about_header:
            match = re.search(r'\((.*?)\)', more_about_header.get_text().strip())
            if match:
                alternative_name = match.group(1).replace("/", "-")
                return None if "-" in alternative_name else alternative_name
    except:
        return None
    return None

def clean_review(review_text):
    quoted_texts = re.findall(r'"(.*?)"', review_text)
    return quoted_texts[0] if quoted_texts else review_text

def process_review_batch(reviews):
    """Process multiple reviews in one batch for efficiency"""
    processed = []
    for review in reviews:
        cleaned = clean_review(review)
        processed_text = preprocess_text(cleaned)
        sentiment_score = sia.polarity_scores(processed_text)["compound"]
        sentiment_label = ("Positive" if sentiment_score > 0.05 
                         else "Negative" if sentiment_score < -0.05 
                         else "Neutral")
        processed.append([processed_text, sentiment_score, sentiment_label])
    return processed

def extract_reviews_parallel(drug1, drug2=None, max_pages=5):
    """Fetch reviews with parallel processing and page limit"""
    search_drug = drug1 if not drug2 or "-" in drug2 else drug2
    base_url = f"https://www.drugs.com/comments/{format_drug_name(search_drug)}/{format_drug_name(drug1)}.html"
    
    def fetch_page(page):
        url = f"{base_url}?page={page}"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                return [clean_review(r.get_text(strip=True)) 
                       for r in soup.find_all("div", class_="ddc-comment")]
            return []
        except:
            return []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Fetch first page to check content
        first_page = fetch_page(1)
        if not first_page:
            return []
        
        # Fetch remaining pages in parallel
        future_to_page = {
            executor.submit(fetch_page, page): page 
            for page in range(2, max_pages + 1)
        }
        
        reviews = first_page
        for future in concurrent.futures.as_completed(future_to_page):
            reviews.extend(future.result())
        
        # Remove duplicates
        unique_reviews = list({r: None for r in reviews}.keys())
        return unique_reviews[:200]  # Limit to 200 reviews for performance

@lru_cache(maxsize=32)
def extract_side_effects(drug1):
    """Optimized side effect extraction with caching"""
    base_url = "https://www.drugs.com/sfx/"
    urls_to_try = [f"{base_url}{format_drug_name(drug1)}-side-effects.html"]
    
    if alt_name := extract_drug2(drug1):
        urls_to_try.append(f"{base_url}{format_drug_name(alt_name)}-side-effects.html")

    for url in urls_to_try:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Try primary extraction method
            side_effects = []
            headers = soup.find_all("h3")
            for header in headers:
                if any(keyword in header.text.lower() 
                      for keyword in ["more common", "less common", "rare side", "symptoms of"]):
                    if ul := header.find_next("ul"):
                        side_effects.extend([li.text.strip() for li in ul.find_all("li")])
            
            if side_effects:
                return side_effects[:50]  # Limit to top 50 effects
            
            # Fallback method if primary fails
            if call_doctor_tag := soup.find(lambda tag: tag.name == "p" and "Call your doctor" in tag.text):
                if ul := call_doctor_tag.find_next("ul"):
                    return [li.text.strip() for li in ul.find_all("li")][:50]
                    
        except:
            continue
    
    return ["No major side effects reported"]

def classify_side_effects_parallel(side_effects):
    """Classify side effects in batches for better performance"""
    classifier = pipeline("zero-shot-classification", 
                        model="facebook/bart-large-mnli",
                        device=0 if pipeline('feature-extraction').device.type == 'cuda' else -1)
    
    categories = ["Mild", "Moderate", "Severe"]
    batch_size = 8  # Optimal for most GPUs
    
    # Preprocess all effects first
    preprocessed = [preprocess_text(effect) for effect in side_effects]
    
    # Classify in batches
    classified = []
    for i in range(0, len(preprocessed), batch_size):
        batch = preprocessed[i:i + batch_size]
        results = classifier(batch, categories, multi_label=False)
        for effect, result in zip(side_effects[i:i + batch_size], results):
            classified.append([effect, result["labels"][0], result["scores"][0]])
    
    return classified

def generate_report(reviews, side_effects, drug_name):
    """Generate report with dark theme formatting"""
    client = openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key="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"
    )

    sentiment_counts = pd.Series([r[2] for r in reviews]).value_counts().to_dict()
    side_effect_texts = "\n".join([f"- {eff[0]} ({eff[1]}, {eff[2]*100:.1f}%)" for eff in side_effects])

    prompt = f"""Generate a comprehensive medical report for {drug_name} with dark theme styling:
    - Total reviews analyzed: {len(reviews)}
    - Positive: {sentiment_counts.get('Positive', 0)}, Negative: {sentiment_counts.get('Negative', 0)}, Neutral: {sentiment_counts.get('Neutral', 0)}
    - Key side effects: {side_effect_texts[:500]}...
    Provide a detailed analysis with: effectiveness summary, risk assessment, and clinical recommendations.
    Format for dark theme display with appropriate markdown.  """
    
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a medical analyst creating reports for dark-themed applications."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.4
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Report generation error: {str(e)}")
        return None

# Initialize Streamlit with dark theme
st.set_page_config(
    page_title="MediFetch - Optimized Drug Analysis",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
    <style>
    .main {
        background-color: #000000;
        color: #ffffff;
    }
    .stTextInput input {
        background-color: #333333;
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #333333;
        color: #ffffff;
    }
    .stSelectbox select {
        background-color: #333333;
        color: #ffffff;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .report-title {
        color: #ffffff;
        font-size: 2.5em;
        text-align: center;
        margin-bottom: 20px;
    }
    .section-header {
        color: #3498db;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
        margin-top: 20px;
    }
    .positive {
        color: #27ae60;
    }
    .negative {
        color: #e74c3c;
    }
    .neutral {
        color: #f39c12;
    }
    /* Dataframe styling */
    .dataframe {
        background-color: #121212 !important;
        color: white !important;
    }
    .dataframe th {
        background-color: #333333 !important;
        color: white !important;
    }
    .dataframe tr:nth-child(even) {
        background-color: #222222 !important;
    }
    .dataframe tr:nth-child(odd) {
        background-color: #333333 !important;
    }
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# App header
st.title("🚀 MediFetch Drug Analysis")
st.markdown("**Comprehensive drug review analysis with dark theme**")

# Initialize session state
if 'processed_reviews' not in st.session_state:
    st.session_state.processed_reviews = []
if 'side_effects' not in st.session_state:
    st.session_state.side_effects = []
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

# Main app
drug_name = st.text_input("Enter the drug name :", key="drug_input")

if st.button("Analyze Drug"):
    if not drug_name:
        st.warning("Please enter a drug name")
    else:
        with st.spinner(f"Processing {drug_name}..."):
            progress_bar = st.progress(0)
            
            # Step 1: Get alternative name
            alt_name = extract_drug2(drug_name)
            progress_bar.progress(20)
            
            # Step 2: Fetch reviews in parallel
            reviews_list = extract_reviews_parallel(drug_name, alt_name)
            progress_bar.progress(40)
            
            if not reviews_list:
                st.error(f"No reviews found for '{drug_name}'.")
                st.session_state.processed_reviews = []
            else:
                # Step 3: Process reviews in batches
                st.session_state.processed_reviews = process_review_batch(reviews_list)
                progress_bar.progress(60)
                
                # Step 4: Extract and classify side effects
                side_effects = extract_side_effects(drug_name)
                progress_bar.progress(80)
                
                st.session_state.side_effects = classify_side_effects_parallel(side_effects)
                progress_bar.progress(100)
                
                st.success("Analysis complete!")
                st.session_state.report_generated = True

# Display results
if st.session_state.processed_reviews:
    st.markdown("## 📊 Analysis Results")
    
    tab1, tab2, tab3 = st.tabs(["Sentiment Analysis", "Side Effects", "Full Report"])
    
    with tab1:
         st.markdown("### Review Sentiment Distribution")
    
    # Display total reviews count prominently
    total_reviews = len(st.session_state.processed_reviews)
    st.markdown(f"**Total Reviews Analyzed:** <span style='color:#3498db; font-size:1.2em'>{total_reviews}</span>", 
               unsafe_allow_html=True)
    
    sentiments = [r[2] for r in st.session_state.processed_reviews]
    sentiment_counts = pd.Series(sentiments).value_counts()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#27ae60', '#e74c3c', '#f39c12']  # Green, Red, Yellow
    ax.pie(sentiment_counts, labels=sentiment_counts.index, 
          autopct='%1.1f%%', colors=colors, startangle=90,
          textprops={'color': 'white'})
    ax.set_title("Sentiment Distribution", color='white')
    st.pyplot(fig)
    
    st.markdown("### Detailed Reviews by Sentiment")
    
    # Create expandable sections for each sentiment category
    for sentiment in ["Positive", "Negative", "Neutral"]:
        sentiment_reviews = [r for r in st.session_state.processed_reviews if r[2] == sentiment]
        count = len(sentiment_reviews)
        if sentiment_reviews:
            with st.expander(f"{sentiment} Reviews ({count})", expanded=False):
                for i, review in enumerate(sentiment_reviews, 1):
                    bg_color = "#1e3d1e" if sentiment == "Positive" else "#3d1e1e" if sentiment == "Negative" else "#3d3d1e"
                    text_color = "#27ae60" if sentiment == "Positive" else "#e74c3c" if sentiment == "Negative" else "#f39c12"
                    
                    st.markdown(
                        f'<div style="background-color:{bg_color}; padding:10px; border-radius:5px; margin:5px;">'
                        f'<span style="color:{text_color}; font-weight:bold">Review #{i} ({sentiment} - Score: {review[1]:.2f})</span><br>'
                        f'{review[0]}</div>',
                        unsafe_allow_html=True
                    )

    with tab2:
        st.markdown("### Classified Side Effects")
        effects_df = pd.DataFrame(st.session_state.side_effects,
                                columns=["Effect", "Severity", "Confidence"])
        
        # Apply dark theme styling
        def color_severity(val):
            color = '#27ae60' if val == "Mild" else '#f39c12' if val == "Moderate" else '#e74c3c'
            return f'color: {color}; font-weight: bold'
        
        st.dataframe(
            effects_df.style.applymap(color_severity, subset=['Severity'])
            .format({'Confidence': '{:.1%}'})
            .set_properties(**{'background-color': '#121212', 'color': 'white'})
            .set_table_styles([{
                'selector': 'th',
                'props': [('background-color', '#333333'), ('color', 'white')]
            }])
        )
    with tab3:
        if st.button("Generate Full Medical Report"):
            with st.spinner("Generating professional report..."):
                report_text = generate_report(
                st.session_state.processed_reviews,
                st.session_state.side_effects,
                drug_name
            )
            
            if report_text:
                st.markdown("### 📝 Comprehensive Medical Report")
                st.markdown(f'<div style="background-color:#121212; padding:15px; border-radius:5px;">{report_text}</div>', 
                          unsafe_allow_html=True)
                
                # Create PDF
                buffer = BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter)
                styles = getSampleStyleSheet()
                story = []
                
                # Add title
                title_style = styles["Title"]
                story.append(Paragraph(f"{drug_name} Medical Report", title_style))
                story.append(Spacer(1, 12))
                
                # Add content
                normal_style = styles["Normal"]
                for line in report_text.split('\n'):
                    if line.strip():  # Skip empty lines
                        story.append(Paragraph(line, normal_style))
                        story.append(Spacer(1, 6))
                
                # Build PDF
                doc.build(story)
                pdf_bytes = buffer.getvalue()
                buffer.close()
                
                # Download button
                st.download_button(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"{drug_name}_report.pdf",
                    mime="application/pdf"
                )
    
    
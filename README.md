# 💊 MediFetch – AI-Driven Drug Analysis & Medical Report Generator

**MediFetch** is an AI-powered web application that scrapes real-time drug data from [Drugs.com](https://www.drugs.com), performs sentiment analysis on user reviews, classifies side effects using zero-shot learning, and generates professional medical reports. Designed with a sleek dark-mode UI, it's built to assist healthcare professionals, researchers, and students with advanced medical insights.

---

## 🧠 Key Features

✅ **Real-time Drug Scraping**  
Extracts user reviews and side effects from Drugs.com.

🧾 **Sentiment Analysis on Reviews**  
Uses VADER (NLTK) to classify reviews into Positive, Negative, or Neutral with pie charts and detailed feedback view.

💥 **Side Effect Classification**  
Classifies extracted side effects into **Mild**, **Moderate**, or **Severe** using BART (Zero-shot learning) from HuggingFace.

📄 **Medical Report Generation**  
Uses **LLaMA3-8B** via **Groq API** (OpenAI-compatible) to generate a full professional summary in Markdown and PDF format.

⚡ **Optimized Performance**  
Includes `@lru_cache` and `concurrent.futures` for fast scraping and batch processing.

🌙 **Dark Mode UI**  
Fully styled Streamlit UI for professional appearance and readability.

---

## 📷 Screenshots

> *(Add screenshots below after running the app)*

- 📊 Sentiment distribution chart  
- 🧪 Side effects table with severity  
- 📝 Downloadable PDF report

---

## 📁 Project Structure


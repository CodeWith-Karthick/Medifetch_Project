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
medifetch/
├── app.py # Main Streamlit application
├── requirements.txt # All dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

## 🚀 Installation

## Clone the repository
git clone https://github.com/CodeWith-Karthick/Medifetch
cd medifetch

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
🔐 API Key Setup (Groq for Report Generation)
To enable medical report generation using LLaMA3:

Get your API key from https://groq.com

Open app.py and locate the following section:

python
Copy
Edit
openai.OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key="your-api-key-here"
)
Replace "your-api-key-here" with your actual Groq API key.

📦 Requirements
Install all dependencies using:

bash
Copy
Edit
pip install -r requirements.txt
Required libraries:
streamlit

requests

beautifulsoup4

nltk

transformers

matplotlib

pandas

reportlab

openai

📌 Use Cases
Clinical review summaries

Drug safety and side-effect profiling

Medical research and analysis

Patient education tool

⚠️ Disclaimer
This application is for educational and research purposes only.
It does not replace professional medical consultation. Always consult a certified physician for medical advice.

🙌 Acknowledgements
Drugs.com – Data Source

NLTK + VADER – Sentiment Analysis

Hugging Face Transformers – NLP Models

Groq – LLaMA3 Inference API

Streamlit – Frontend Framework

📫 Contact
Created by Karthick G
📧 Email: karthick14mm@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/11karthick/

⭐ Star the Repo
If you found this project useful, please give it a ⭐ on GitHub and share it with others in the community!

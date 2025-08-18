# 🔎 Resume → Job Listings Recommender

This repository contains a **Streamlit application** and supporting **analysis notebooks** for matching resumes against job listings.  
It includes:

- 📑 **Resume Parsing & Enrichment** (PDF/DOCX parsing → LLM-powered information extraction)  
- 🤖 **Job Listing Preprocessing** (scraping, enrichment, embeddings)  
- 🎯 **Scoring & Ranking Engine** to match candidates with top job opportunities
- 📊 **Exploratory & Cluster Analysis** on job listings  

### 🎥 Demo Video

- [Watch the video](https://www.loom.com/share/f6a0846323594364b5e3bab4da3c2953?sid=90f62483-bac2-4985-b76f-f2a21f88c4c8)

## 📑 Reports

- [📑 Resume ↔ Job Matching System Report](https://www.notion.so/Resume-Job-Matching-System-Report-252f137f6f9e802ea51feb2905ca9f0a?source=copy_link)
- [🧠 Job Listings Analysis – Insights & Recommendations](https://www.notion.so/Job-Listings-Analysis-Insights-Recommendations-253f137f6f9e80fa8b28dbec19c3c8da?source=copy_link)
- [🚀 Scoring  Methodology](https://www.notion.so/Scoring-Methodology-253f137f6f9e807e841ef7fae9bd8318?source=copy_link)
  
---

## 📂 Project Structure

```
.
├── app.py                # Streamlit web app entrypoint
├── requirements.txt      # Python dependencies
├── src/                  # Source modules
│   ├── resume.py         # Resume parsing, enrichment, salary inference
│   ├── comparison.py     # Ranking & scoring of listings vs resumes
│   ├── listings.py       # Job listings preprocessing, enrichment
│   ├── scraper.py        # Web scraping utilities
│   └── config.py         # API clients & configuration
├── data/                 # Local storage for precomputed results
│   ├── result.pkl        # Job listings dictionary
│   ├── pre.pkl           # Precomputed embeddings/features
├── notebooks/            # Jupyter notebooks for analysis
│   └── job_listings_analysis.ipynb
└── README.md             # This file
```

---

## ⚡️ Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/mercor-job-matcher.git
cd mercor-job-matcher
```

### 2. Set up environment
Using `pip`:
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

Or with Conda:
```bash
conda create -n mercor python=3.11 -y
conda activate mercor
pip install -r requirements.txt
```

### 3. Add your API keys
Update `src/config.py`) with your credentials:
```
OPENAI_API_KEY=your-key-here
GOOGLE_API_KEY=your-key-here
```

### 4. Run the app
```bash
streamlit run app.py
```

### 5. Explore the analysis
```bash
jupyter notebook notebooks/job_listings_analysis.ipynb
```

---

## 🛠 Features

- **Resume Parser**  
  - Extracts text from PDF/DOCX resumes  
  - Detects contact info, locations, work experience  

- **Resume Enrichment (LLM-powered)**  
  - Uses OpenAI models for structured information extraction  
  - Normalizes technologies, skills, responsibilities, industries  

- **Salary Inference**  
  - Uses Google Gemini + search tool to infer hourly USD salary  

- **Job Listings Pipeline**  
  - Scrapes and normalizes listings (compensation → hourly, embeddings, tags)  

- **Recommendation Engine**  
  - Ranks job listings by title similarity, responsibilities, skills overlap, industry alignment  
  - Transparent scoring & “Why this match?” explanations  

- **Exploratory Data Analysis**  
  - Compensation distribution  
  - Job family clustering (KMeans)  
  - Technology & skills co-occurrence networks  

---



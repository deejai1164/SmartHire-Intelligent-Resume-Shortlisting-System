# SmartHire-Intelligent-Resume-Shortlisting-System

SmartHire is an AI-powered resume screening and candidate ranking system developed using Python and Flask. The project automates the recruitment screening process by combining semantic similarity and keyword-based skill matching to evaluate candidate resumes against job descriptions.

The system supports both individual resume analysis and HR batch screening workflows, helping recruiters identify the most relevant candidates quickly and efficiently.

🚀 Features
📄 Resume Upload (PDF & DOCX)
🧠 Semantic Similarity Matching
🔍 Keyword & Skill Extraction
📊 Hybrid Scoring System
🏆 Candidate Ranking
📉 Skill Gap Analysis
📧 Email & Phone Extraction
👥 HR Batch Resume Processing
🤖 Optional Gemini AI Integration
📈 Evaluation Metrics (Accuracy, Precision, Recall, F1, ROC-AUC)

🛠️ Technologies Used
Technology      	                     Purpose
Python	                               Core programming language
Flask	                                 Web framework
spaCy	                                 NLP preprocessing
NLTK	                                 Keyword processing
Sentence Transformers (MiniLM)	       Semantic similarity
PyMuPDF	                               PDF text extraction
python-docx	                           DOCX parsing
scikit-learn	                         Evaluation utilities
Google Gemini API	                     Optional AI enhancement

⚙️ System Workflow
Resume Upload
      ↓
Text Extraction
      ↓
NLP Preprocessing
      ↓
Semantic Embedding Generation
      ↓
Keyword & Skill Extraction
      ↓
Hybrid Scoring
      ↓
Candidate Ranking & Analysis

🖥️ System Modules
1. Individual Mode
Allows users to upload a single resume and compare it against a job description.
Output:
Match Score
Matched Skills
Missing Skills
Candidate Contact Information

2. HR Mode
Allows recruiters to upload multiple resumes simultaneously.
Output:
Ranked Candidate List
Comparative Scores
Skill Analysis
Shortlisting Support

📊 Performance Metrics
Metric	Value
Accuracy	94.5%
Precision	0.93
Recall	0.92
F1 Score	0.925
ROC-AUC	0.96

📂 Project Structure
SmartHire/
│
├── app.py
├── requirements.txt
├── templates/
├── static/
├── uploads/
├── resume_parser/
├── matcher/
├── evaluation/
└── README.md

🔧 Installation & Setup
Clone Repository
git clone https://github.com/your-username/SmartHire.git
cd SmartHire
Install Dependencies
pip install -r requirements.txt
Run Application
python app.py

🌐 Usage
Open the web application
Upload Resume(s)
Enter Job Description
Click Analyze
View Candidate Scores & Ranking

🎯 Future Scope
OCR Support for Scanned Resumes
Multilingual Resume Processing
ATS Integration
Explainable AI (XAI)
Adaptive Hybrid Weight Learning
Cloud Deployment

🌍 Sustainability Goals (SDGs)

This project contributes to:

SDG 8 – Decent Work & Economic Growth
SDG 9 – Industry, Innovation & Infrastructure
SDG 10 – Reduced Inequalities
SDG 4 – Quality Education
SDG 16 – Peace, Justice & Strong Institutions

👨‍💻 Authors
Deepak Jaiswal
Diwakar Mishra
Ankur Sanjay Yadav
Abhishek Shukla

📜 License
This project is developed for academic and educational purposes.

⭐ SmartHire
An intelligent, scalable, and transparent AI-based recruitment screening solution.

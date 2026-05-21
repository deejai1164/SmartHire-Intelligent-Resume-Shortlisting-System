# SmartHire - AI Resume Shortlisting System

## Project Overview

This is a Flask-based web application that screens resumes against job descriptions using a **dual-matching strategy**:
1. **Semantic Matching** - Uses sentence embeddings (sentence-transformers) to understand meaning and context
2. **Keyword Matching** - Uses NLP (spaCy, WordNet) and domain knowledge to match specific skills

The system integrates with **Google Gemini** for intelligent job description parsing and feedback generation.

---

## Project Structure

```
Automaton-resume-screenning-main/
├── app.py                          # Main Flask application
├── skill_domains.py               # Domain-to-skills mapping for keyword expansion
├── requirements.txt                # Python dependencies
├── services/                       # Service modules (NEW)
│   ├── __init__.py
│   ├── jd_parser.py               # Job description JSON parser
│   ├── resume_parser.py           # Resume text extraction (PDF/DOCX)
│   └── matcher.py                 # Semantic matching engine
├── templates/
│   ├── index.html                 # Upload form
│   └── result.html                # Results display
├── static/
│   └── style.css                  # UI styling
└── venv/                          # Virtual environment

```

---

## Workflow & Data Flow

### 1. **User Upload** (`/` route)
- User uploads a resume (PDF or DOCX)
- User enters job description text
- Form submits to `/analyze` endpoint

### 2. **Resume Processing** (`services/resume_parser.py`)
```
Resume File → extract_text_from_resume() → Raw Text
Raw Text → clean_resume_text() → Cleaned Text
```
- **PDF**: Uses PyMuPDF (fitz) for extraction
- **DOCX**: Uses python-docx for extraction
- **Cleaning**: Removes emails, phone numbers, addresses, extra whitespace

### 3. **Job Description Processing** (`services/jd_parser.py`)

**Option A: Structured JD from Gemini** (if available)
```
Job Text → get_structured_jd_from_gemini() → JSON
JSON → build_structured_jd_text() → Structured Text for Semantic Matching
JSON → extract_skills_from_jd_json() → Skills List
```

**Option B: Fallback** (if Gemini unavailable)
```
Job Text → extract_job_keywords() → Keywords List (spaCy + regex)
```

### 4. **Semantic Matching** (`services/matcher.py`)
```
Cleaned Resume Text + Structured JD Text → calculate_similarity()
→ Sentence Embeddings (all-MiniLM-L6-v2)
→ Cosine Similarity
→ Semantic Score (0-100%)
```

**Model Loading**: SentenceTransformer model is loaded **once globally** on first use (not per request).

### 5. **Keyword Matching** (`app.py` → `extract_skills()`)
```
Resume Text + Job Keywords → extract_resume_skills() → Resume Skills Set
Resume Skills + Job Keywords → Domain Expansion + WordNet Synonyms → Matching
→ Keyword Match Score (0-100%)
```

### 6. **Skills Analysis** (`services/matcher.py` → `skills_analysis()`)
```
Resume Text + Required Skills → Keyword Presence Check
→ Matched Skills List
→ Missing Skills List
```

### 7. **Score Combination**
```
Final Score = (Semantic Score × 0.6) + (Keyword Score × 0.4)
```

### 8. **AI Feedback Generation** (`app.py` → `generate_feedback_with_gemini()`)
- Sends full context to Gemini
- Returns human-readable feedback paragraphs

### 9. **Result Display** (`templates/result.html`)
- Overall combined score
- Score breakdown (semantic + keyword)
- Matched skills list
- Missing skills list
- AI-generated feedback

---

## Key Components

### `services/jd_parser.py`
- **`build_structured_jd_text(jd_json)`**: Converts structured JD JSON to text for semantic matching
- **`extract_skills_from_jd_json(jd_json)`**: Extracts all skills (required + optional) from JD JSON

### `services/resume_parser.py`
- **`extract_text_from_resume(file_path, file_obj)`**: Extracts text from PDF/DOCX files
- **`clean_resume_text(text)`**: Removes PII and normalizes text

### `services/matcher.py`
- **`calculate_similarity(resume_text, jd_text)`**: Computes semantic similarity using sentence embeddings
- **`skills_analysis(resume_text, required_skills)`**: Keyword-based presence check for UI display
- **Global model**: `SentenceTransformer('all-MiniLM-L6-v2')` loaded once

### `app.py` - Main Application
- **`get_structured_jd_from_gemini()`**: Requests structured JD JSON from Gemini
- **`get_job_keywords_from_gemini()`**: Requests keyword list from Gemini
- **`extract_skills()`**: Keyword-based matching with domain expansion
- **`generate_feedback_with_gemini()`**: AI feedback generation
- **`analyze()` route**: Main endpoint that orchestrates all services

---

## Dependencies

### Core
- `flask>=2.0.0` - Web framework
- `PyMuPDF>=1.21.0` - PDF extraction
- `spacy>=3.0.0` - NLP processing
- `nltk>=3.8.0` - WordNet synonyms

### Semantic Matching (NEW)
- `sentence-transformers>=2.2.0` - Sentence embeddings
- `scikit-learn>=1.3.0` - Cosine similarity
- `PyPDF2>=3.0.0` - PDF backup parser
- `python-docx>=1.1.0` - DOCX support

### AI Integration
- `google-generativeai>=0.8.0` - Gemini API

---

## Environment Setup

1. **Set Gemini API Key** (optional but recommended):
   ```powershell
   $env:GEMINI_API_KEY = "your-api-key-here"
   ```

2. **Install dependencies**:
   ```powershell
   .\venv\Scripts\pip install -r requirements.txt
   ```

3. **Download spaCy model** (if not already done):
   ```powershell
   .\venv\Scripts\python -m spacy download en_core_web_sm
   ```

4. **Download NLTK data** (if not already done):
   ```powershell
   .\venv\Scripts\python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
   ```

5. **Run the application**:
   ```powershell
   .\venv\Scripts\python app.py
   ```

---

## Features

### ✅ Dual Matching Strategy
- **Semantic**: Understands meaning and context (60% weight)
- **Keyword**: Precise skill matching (40% weight)

### ✅ Multi-Format Support
- PDF resumes (PyMuPDF)
- DOCX resumes (python-docx)

### ✅ Intelligent JD Processing
- Structured JD extraction via Gemini
- Fallback to local NLP if Gemini unavailable

### ✅ Comprehensive Analysis
- Match score (combined)
- Score breakdown (semantic + keyword)
- Matched skills list
- Missing skills list
- AI-generated feedback

### ✅ Performance Optimized
- SentenceTransformer model loaded once globally
- Efficient text processing
- Error handling and graceful fallbacks

---

## Error Handling

- **Unsupported file types**: Returns 400 with clear error message
- **Empty resume**: Validates minimum text length
- **Gemini failures**: Falls back to local NLP processing
- **Semantic matching failures**: Continues with keyword matching only
- **Missing dependencies**: Clear error messages guide installation

---

## Future Enhancements (Optional)

- Add support for more file formats (TXT, RTF)
- Cache sentence embeddings for faster processing
- Add batch processing for multiple resumes
- Export results to PDF/CSV
- Add user authentication and history
- Support for multiple languages

---

## Notes

- The SentenceTransformer model (`all-MiniLM-L6-v2`) is ~90MB and downloads automatically on first use
- Semantic matching may take 1-3 seconds per resume (depending on text length)
- Gemini API calls require internet connection and API key
- The system works offline for keyword matching if Gemini is unavailable

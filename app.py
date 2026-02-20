import os
import re
import json
from flask import Flask, request, render_template
import fitz  # PyMuPDF
import spacy
from nltk.corpus import wordnet

from skill_domains import (
    get_domain_related_skills,
    get_all_expanded_terms,
    ALL_KNOWN_SINGLE_WORDS,
    KNOWN_SKILL_PHRASES,
)

# Import new services
from services.resume_parser import extract_text_from_resume, clean_resume_text, extract_email, extract_phone
from services.jd_parser import build_structured_jd_text, extract_skills_from_jd_json
from services.matcher import calculate_similarity, skills_analysis

try:
    import google.generativeai as genai  # type: ignore
except Exception:
    genai = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if genai is not None and GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        genai = None

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Get synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower().replace("_", " "))
    except Exception:
        pass
    return synonyms


def _normalize(s):
    return " ".join(s.lower().strip().split())


# Extract keywords from job description: comma-separated input + NLP tokens
def extract_job_keywords(text):
    """
    Fallback keyword extraction using regex + spaCy (used when Gemini is unavailable).
    """
    text = text.strip()
    keywords = set()

    # 1) Comma-separated phrases (user intent: "Data Science, Python, Communication")
    for part in re.split(r"[,;|\n]+", text):
        phrase = _normalize(part)
        if phrase and len(phrase) > 1:
            keywords.add(phrase)

    # 2) NLP extraction (nouns, proper nouns, entities)
    doc = nlp(text.lower())
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] or token.ent_type_ in ["SKILL", "ORG", "PRODUCT"]:
            if not token.is_stop and token.is_alpha:
                keywords.add(token.text.strip())
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "SKILL"]:
            keywords.add(ent.text.lower().strip())

    # 3) Deduplicate and return as list (order preserved for stable scoring)
    seen = set()
    result = []
    for k in keywords:
        n = _normalize(k)
        if n not in seen:
            seen.add(n)
            result.append(n)
    print("📌 Extracted Job Keywords (fallback):", result)
    return result


def get_structured_jd_from_gemini(job_description_text):
    """
    Use Gemini to extract structured JD JSON with role, skills, domain, etc.
    Returns (jd_json_dict, info_message) or (None, error_message).
    """
    if genai is None or not GEMINI_API_KEY:
        return None, "Gemini not configured."

    system_prompt = (
        "You are an ATS-style recruiter assistant. "
        "Given a job description, extract structured information. "
        "Return ONLY valid JSON with this exact structure:\n"
        "{\n"
        '  "job_role": "Job Title",\n'
        '  "required_skills": ["skill1", "skill2", ...],\n'
        '  "optional_skills": ["skill1", "skill2", ...],\n'
        '  "minimum_experience_years": 2,\n'
        '  "domain": "Domain name",\n'
        '  "tools_and_technologies": ["tool1", "tool2", ...]\n'
        "}\n"
        "If a field is not available, use empty array [] or null. "
        "Keep skill names concise (1-3 words)."
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            [
                system_prompt,
                "\nJob description:\n",
                job_description_text,
            ]
        )
        text = (response.text or "").strip()
        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(text)
        print("🤖 Gemini Structured JD:", json.dumps(data, indent=2))
        return data, "Using Gemini-structured JD."
    except Exception as e:
        print("🔥 Gemini structured JD extraction error:", str(e))
        return None, f"Gemini error: {str(e)}"


def get_job_keywords_from_gemini(job_description_text):
    """
    Use Gemini to generate a clean, de-duplicated list of job-related skill keywords.
    Returns (keywords_list, info_message).
    """
    if genai is None or not GEMINI_API_KEY:
        return None, "Gemini not configured – using local NLP for keywords."

    system_prompt = (
        "You are an ATS-style recruiter assistant. "
        "Given a job description, extract a focused list of 10–40 concise skill keywords "
        "(technical and soft skills) that are important for evaluating a resume. "
        "Return ONLY valid JSON with this structure:\n"
        "{\n"
        '  "job_keywords": ["keyword1", "keyword2", "..."]\n'
        "}\n"
        "Use short phrases (1–3 words), all lowercase, no duplicates."
    )

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            [
                system_prompt,
                "\nJob description:\n",
                job_description_text,
            ]
        )
        text = (response.text or "").strip()
        # Remove markdown code blocks if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        data = json.loads(text)
        raw_keywords = data.get("job_keywords", [])
        cleaned = []
        seen = set()
        for k in raw_keywords:
            n = _normalize(str(k))
            if n and n not in seen:
                seen.add(n)
                cleaned.append(n)
        if not cleaned:
            return None, "Gemini returned no keywords – using local NLP instead."
        print("🤖 Gemini Job Keywords:", cleaned)
        return cleaned, "Using Gemini-derived job keywords."
    except Exception as e:
        print("🔥 Gemini keyword extraction error:", str(e))
        return None, "Gemini error – falling back to local NLP keywords."


# Extract all skill-like terms from resume (tokens, entities, known phrases)
def extract_resume_skills(resume_text):
    resume_lower = resume_text.lower()
    doc = nlp(resume_lower)
    extracted = set()

    # Entities (ORG, PRODUCT, SKILL)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "SKILL"]:
            extracted.add(ent.text.lower().strip())

    # Single tokens: nouns, proper nouns, or known skill words
    for token in doc:
        if not token.is_alpha or token.is_stop:
            continue
        t = token.text.lower()
        if token.pos_ in ["NOUN", "PROPN"] or token.ent_type_ in ["SKILL", "ORG", "PRODUCT"]:
            extracted.add(t)
        if t in ALL_KNOWN_SINGLE_WORDS:
            extracted.add(t)

    # Multi-word phrases from our domain map (e.g. "machine learning", "data science")
    for phrase in KNOWN_SKILL_PHRASES:
        if phrase in resume_lower:
            extracted.add(phrase)

    # Also detect common bigrams in text (consecutive noun-like words)
    for i in range(len(doc) - 1):
        bigram = f"{doc[i].text.lower()} {doc[i+1].text.lower()}"
        if bigram in KNOWN_SKILL_PHRASES:
            extracted.add(bigram)

    print("🧠 Extracted Resume Skills (sample):", list(extracted)[:40])
    return extracted


# Match resume skills to job requirements using synonyms + domain expansion
def extract_skills(resume_text, job_description_keywords):
    """
    Compute matching skills and score between a resume and a list of job keywords.
    Returns (matching_skills, percentage, resume_skills_set).
    """
    resume_skills = extract_resume_skills(resume_text)
    matching_skills = []

    for jd_keyword in job_description_keywords:
        # Expanded set: the keyword itself + WordNet synonyms + domain-related skills
        expanded = get_all_expanded_terms(jd_keyword)
        synonyms = get_synonyms(jd_keyword)
        expanded.update(synonyms)

        if any(rs in expanded for rs in resume_skills):
            matching_skills.append(jd_keyword)

    total = len(job_description_keywords)
    percentage = (len(matching_skills) / total * 100) if total else 0
    print("✅ Matching Skills:", matching_skills)
    print("📊 Matching Percentage:", percentage)
    return matching_skills, percentage, resume_skills

# Extract resume text (keeping for backward compatibility)
def extract_resume_text(file):
    """Legacy function - now uses resume_parser service."""
    try:
        return extract_text_from_resume(file_obj=file)
    except Exception as e:
        # Fallback to original PyMuPDF method
        text = ""
        file.seek(0)
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        for page in pdf:
            text += page.get_text()
        pdf.close()
        return text


def generate_feedback_with_gemini(job_text, resume_skills, job_keywords, matching_skills, percentage, missing_skills):
    """
    Ask Gemini for human-readable feedback about the resume vs job fit.
    Returns a plain-text paragraph string.
    """
    if genai is None or not GEMINI_API_KEY:
        return None

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "You are a concise resume reviewer for recruiters.\n\n"
            "Given the job description, the skills the candidate appears to have, "
            "the core job keywords, which ones matched, which ones are missing, "
            "and the numeric match score, write a short feedback summary.\n\n"
            "Guidelines:\n"
            "- 2–4 short paragraphs, plain text (no markdown, no bullet points).\n"
            "- Start with an overall impression of fit.\n"
            "- Call out the strongest aligned skills.\n"
            "- Then clearly describe the main missing or weak areas.\n"
            "- Use neutral, professional language as if writing notes for a recruiter.\n"
            "- Do NOT restate the numeric score.\n\n"
            f"Job description:\n{job_text}\n\n"
            f"All extracted job keywords (after processing):\n{job_keywords}\n\n"
            f"Skills detected in the resume (normalized):\n{sorted(list(resume_skills))}\n\n"
            f"Job keywords that matched the resume:\n{matching_skills}\n\n"
            f"Job keywords that appear to be missing or weak:\n{missing_skills}\n\n"
            f"Heuristic match score: {percentage:.1f}%.\n"
        )
        response = model.generate_content(prompt)
        feedback = (response.text or "").strip()
        print("📝 Gemini feedback generated.")
        return feedback
    except Exception as e:
        print("🔥 Gemini feedback error:", str(e))
        return None

# Flask app
app = Flask(__name__)

def process_single_resume(file, job_text, filename=None):
    """
    Process a single resume file and return analysis results.
    Used by both individual and batch processing.
    
    Returns:
        dict with keys: filename, combined_score, semantic_score, keyword_score,
                       matched_skills, missing_skills, email, phone, resume_text_raw
    """
    if filename is None:
        filename = getattr(file, 'filename', 'Unknown')
    
    # Extract and clean resume text
    resume_text_raw = extract_text_from_resume(file_obj=file)
    if not resume_text_raw or len(resume_text_raw.strip()) < 10:
        raise ValueError(f"Resume '{filename}' appears to be empty or could not be extracted.")
    
    resume_text_cleaned = clean_resume_text(resume_text_raw)
    
    # Extract contact information BEFORE cleaning
    email = extract_email(resume_text_raw)
    phone = extract_phone(resume_text_raw)
    
    # Try to get structured JD from Gemini
    jd_json, _ = get_structured_jd_from_gemini(job_text)
    jd_text_for_semantic = job_text
    
    if jd_json:
        jd_text_for_semantic = build_structured_jd_text(jd_json)
        if not jd_text_for_semantic:
            jd_text_for_semantic = job_text
    
    # Calculate semantic similarity score
    semantic_score = 0.0
    try:
        semantic_score = calculate_similarity(resume_text_cleaned, jd_text_for_semantic)
    except Exception as e:
        print(f"⚠️ Semantic matching failed for {filename}: {str(e)}")
    
    # Get job keywords
    gemini_keywords, _ = get_job_keywords_from_gemini(job_text)
    if gemini_keywords:
        job_keywords = gemini_keywords
    else:
        job_keywords = extract_job_keywords(job_text)
    
    # Compute keyword-based matching
    skills, keyword_percentage, resume_skills = extract_skills(resume_text_cleaned, job_keywords)
    
    # Skills analysis
    skills_for_analysis = job_keywords
    if jd_json:
        structured_skills = extract_skills_from_jd_json(jd_json)
        if structured_skills:
            skills_for_analysis = structured_skills
    
    skills_analysis_result = skills_analysis(resume_text_cleaned, skills_for_analysis)
    matched_skills_list = skills_analysis_result.get("matched_skills", skills)
    missing_skills_list = skills_analysis_result.get("missing_skills", [])
    
    # Combine scores
    if semantic_score > 0:
        combined_score = (semantic_score * 0.6) + (keyword_percentage * 0.4)
    else:
        combined_score = keyword_percentage
    
    return {
        'filename': filename,
        'combined_score': combined_score,
        'semantic_score': semantic_score,
        'keyword_score': keyword_percentage,
        'matched_skills': matched_skills_list,
        'missing_skills': missing_skills_list,
        'email': email,
        'phone': phone,
        'resume_text_raw': resume_text_raw,  # For feedback generation if needed
        'resume_skills': resume_skills,
        'job_keywords': job_keywords,
    }


@app.route('/')
def dashboard():
    """Main dashboard - choose between HR and Individual."""
    return render_template('dashboard.html')


@app.route('/individual')
def individual_upload():
    """Individual resume upload page."""
    return render_template('index.html')


@app.route('/hr')
def hr_upload():
    """HR batch upload page."""
    return render_template('hr_upload.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Individual resume analysis endpoint."""
    if 'resume' not in request.files:
        print("🚫 No file uploaded.")
        return "No file uploaded", 400

    file = request.files['resume']
    job_text = request.form['job_description']

    print("📥 Job Description Received:", job_text)

    # Validate file type
    if not file or not file.filename:
        return "No file uploaded", 400
    
    file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    if file_ext not in ['pdf', 'docx']:
        return f"Unsupported file type. Only PDF and DOCX are allowed. Got: {file_ext}", 400

    try:
        print("📁 Uploaded Resume:", file.filename)
        
        # Process resume using shared function
        result = process_single_resume(file, job_text, file.filename)
        
        # Generate AI feedback for individual view
        feedback = generate_feedback_with_gemini(
            job_text=job_text,
            resume_skills=result['resume_skills'],
            job_keywords=result['job_keywords'],
            matching_skills=result['matched_skills'],
            percentage=result['combined_score'],
            missing_skills=result['missing_skills'],
        )

        return render_template(
            "result.html",
            skills=result['matched_skills'],
            percentage=result['combined_score'],
            semantic_score=result['semantic_score'],
            keyword_score=result['keyword_score'],
            missing_skills=result['missing_skills'],
            feedback=feedback,
        )
    except ValueError as e:
        print(f"🔥 Validation error: {str(e)}")
        return f"Error: {str(e)}", 400
    except Exception as e:
        print(f"🔥 Error while analyzing: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 400


@app.route('/hr/analyze', methods=['POST'])
def hr_analyze():
    """HR batch resume analysis endpoint."""
    if 'resumes' not in request.files:
        print("🚫 No files uploaded.")
        return "No files uploaded", 400

    files = request.files.getlist('resumes')
    job_text = request.form['job_description']

    if not files or len(files) == 0:
        return "No files uploaded", 400

    print(f"📥 Job Description Received. Processing {len(files)} resume(s)...")

    results = []
    errors = []

    for file in files:
        if not file.filename:
            continue
        
        filename = file.filename
        file_ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if file_ext not in ['pdf', 'docx']:
            errors.append(f"{filename}: Unsupported file type")
            continue

        try:
            print(f"📁 Processing: {filename}")
            result = process_single_resume(file, job_text, filename)
            results.append(result)
            print(f"✅ Processed {filename}: Score = {result['combined_score']:.2f}%")
        except Exception as e:
            error_msg = f"{filename}: {str(e)}"
            errors.append(error_msg)
            print(f"🔥 Error processing {filename}: {str(e)}")

    if not results:
        return f"No resumes could be processed. Errors: {', '.join(errors)}", 400

    # Sort results by combined_score (highest first)
    results.sort(key=lambda x: x['combined_score'], reverse=True)

    print(f"📊 Batch processing complete: {len(results)} successful, {len(errors)} errors")

    return render_template(
        "hr_results.html",
        results=results,
        errors=errors if errors else None,
    )

if __name__ == '__main__':
    print("🚀 Flask server starting on http://127.0.0.1:5000 ...")
    app.run(debug=True)

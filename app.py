import os
import re
import json
from flask import Flask, request, render_template, jsonify
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


GENERIC_ROLE_WORDS = {
    "developer", "engineer", "specialist", "consultant", "lead", "manager",
    "senior", "junior", "stack", "full", "intern", "associate", "expert",
}


def _dedupe_keywords_preserve_phrases(keywords):
    """
    Keep meaningful phrases (e.g. 'full stack developer') and avoid noisy single-word
    fragments (e.g. 'full', 'stack', 'developer') when phrase already exists.
    """
    normalized = []
    seen = set()
    for k in keywords:
        n = _normalize(k)
        if n and n not in seen:
            seen.add(n)
            normalized.append(n)

    phrases = [k for k in normalized if " " in k]
    out = []
    for k in normalized:
        if " " not in k and k in GENERIC_ROLE_WORDS:
            if any(re.search(rf"\b{re.escape(k)}\b", p) for p in phrases):
                continue
        out.append(k)
    return out


def enrich_jd_keywords(base_keywords, max_terms=35):
    """
    Expand JD intent with domain-related terms while preserving main phrases.
    """
    ordered = _dedupe_keywords_preserve_phrases(base_keywords)
    enriched = list(ordered)
    seen = set(enriched)

    for k in ordered:
        related = sorted(get_domain_related_skills(k))
        for r in related:
            nr = _normalize(r)
            if not nr or nr in seen:
                continue
            seen.add(nr)
            enriched.append(nr)
            if len(enriched) >= max_terms:
                return enriched
    return enriched


def get_matching_weights():
    """
    Tunable hybrid weights (keyword-first by default).
    Set via env vars:
      - SEMANTIC_WEIGHT (default 0.30)
      - KEYWORD_WEIGHT  (default 0.70)
    """
    try:
        semantic_w = float(os.getenv("SEMANTIC_WEIGHT", "0.30"))
    except Exception:
        semantic_w = 0.30
    try:
        keyword_w = float(os.getenv("KEYWORD_WEIGHT", "0.70"))
    except Exception:
        keyword_w = 0.70

    # Keep sane bounds and normalize
    semantic_w = max(0.0, min(1.0, semantic_w))
    keyword_w = max(0.0, min(1.0, keyword_w))
    total = semantic_w + keyword_w
    if total <= 0:
        return 0.30, 0.70
    return semantic_w / total, keyword_w / total


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

    # 2) NLP extraction (preserve phrases first)
    doc = nlp(text.lower())

    for chunk in doc.noun_chunks:
        phrase = _normalize(chunk.text)
        if phrase and len(phrase) > 2:
            keywords.add(phrase)

    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT", "SKILL"]:
            keywords.add(ent.text.lower().strip())

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] or token.ent_type_ in ["SKILL", "ORG", "PRODUCT"]:
            if not token.is_stop and token.is_alpha:
                keywords.add(token.text.strip())

    # 3) Dedupe and preserve phrases (avoid adding extra expanded/noisy keywords)
    result = _dedupe_keywords_preserve_phrases(list(keywords))
    print("Extracted Job Keywords (fallback):", result)
    return result


def _keyword_aliases(keyword):
    """
    Build deterministic aliases for a keyword so near-equivalent terms match.
    Example: mongodb -> {"mongodb", "mongo", "mongo database"}.
    """
    base = _normalize(str(keyword))
    if not base:
        return set()

    aliases = {base}
    compact = re.sub(r"[^a-z0-9]+", "", base)
    if compact:
        aliases.add(compact)

    tokens = [t for t in re.split(r"[\s\-/]+", base) if t]
    if tokens:
        aliases.update(tokens)

    # database/db style variants
    if "mongodb" in aliases or "mongo" in aliases or "mongo database" in aliases:
        aliases.update({"mongo", "mongodb", "mongo db", "mongo database"})
    if "postgresql" in aliases or "postgres" in aliases:
        aliases.update({"postgres", "postgresql", "postgre sql"})
    if "nodejs" in aliases or "node.js" in base or "node js" in base:
        aliases.update({"node", "nodejs", "node js", "node.js"})
    if "javascript" in aliases:
        aliases.update({"javascript", "js"})
    if "typescript" in aliases:
        aliases.update({"typescript", "ts"})

    return {_normalize(a) for a in aliases if _normalize(a)}


def _resume_has_any_alias(resume_text, aliases):
    resume_lower = f" {_normalize(resume_text)} "
    resume_compact = re.sub(r"[^a-z0-9]+", "", resume_lower)

    for a in aliases:
        a_norm = _normalize(a)
        if not a_norm:
            continue
        if f" {a_norm} " in resume_lower:
            return True
        a_compact = re.sub(r"[^a-z0-9]+", "", a_norm)
        if a_compact and a_compact in resume_compact:
            return True
    return False


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
        print("Gemini Structured JD:", json.dumps(data, indent=2))
        return data, "Using Gemini-structured JD."
    except Exception as e:
        print("Gemini structured JD extraction error:", str(e))
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
        "Given a job description, extract a focused list of 8–25 concise skill keywords "
        "(technical and soft skills) that are important for evaluating a resume. "
        "Return ONLY valid JSON with this structure:\n"
        "{\n"
        '  "job_keywords": ["keyword1", "keyword2", "..."]\n'
        "}\n"
        "Use short phrases (1–3 words), all lowercase, no duplicates. "
        "Do NOT add broad related skills that are not explicitly required in the job description. "
        "Prefer canonical names that allow alias matching (example: mongodb, postgresql, javascript)."
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
        cleaned = _dedupe_keywords_preserve_phrases(cleaned)
        print("Gemini Job Keywords:", cleaned)
        return cleaned, "Using Gemini-derived focused keywords."
    except Exception as e:
        print("Gemini keyword extraction error:", str(e))
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

    print("Extracted Resume Skills (sample):", list(extracted)[:40])
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
        # Alias-aware strict matching: any alias hit gives full credit for that keyword.
        aliases = _keyword_aliases(jd_keyword)
        alias_hit = _resume_has_any_alias(resume_text, aliases)

        # Keep legacy extracted-skill set as a secondary check for stability.
        if not alias_hit:
            expanded = get_all_expanded_terms(jd_keyword)
            synonyms = get_synonyms(jd_keyword)
            expanded.update(synonyms)
            alias_hit = any(rs in expanded for rs in resume_skills)

        if alias_hit:
            matching_skills.append(jd_keyword)

    total = len(job_description_keywords)
    percentage = (len(matching_skills) / total * 100) if total else 0
    print("Matching Skills:", matching_skills)
    print("Matching Percentage:", percentage)
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
        print("Gemini feedback generated.")
        return feedback
    except Exception as e:
        print("Gemini feedback error:", str(e))
        return None

# Flask app
app = Flask(__name__)


def score_resume_vs_job(resume_text_raw, job_text, filename="Unknown", offline_eval=False):
    """
    Core scoring pipeline from raw resume text + job description.
    offline_eval=True skips Gemini (deterministic benchmark runs).
    """
    if not resume_text_raw or len(resume_text_raw.strip()) < 10:
        raise ValueError(f"Resume '{filename}' appears to be empty or could not be extracted.")

    resume_text_cleaned = clean_resume_text(resume_text_raw)

    email = extract_email(resume_text_raw)
    phone = extract_phone(resume_text_raw)

    jd_json = None
    if not offline_eval:
        jd_json, _ = get_structured_jd_from_gemini(job_text)

    jd_text_for_semantic = job_text
    if jd_json:
        jd_text_for_semantic = build_structured_jd_text(jd_json)
        if not jd_text_for_semantic:
            jd_text_for_semantic = job_text

    semantic_score = 0.0
    try:
        semantic_score = calculate_similarity(resume_text_cleaned, jd_text_for_semantic)
    except Exception as e:
        print(f"Semantic matching failed for {filename}: {str(e)}")

    if offline_eval:
        job_keywords = extract_job_keywords(job_text)
    else:
        gemini_keywords, _ = get_job_keywords_from_gemini(job_text)
        if gemini_keywords:
            job_keywords = gemini_keywords
        else:
            job_keywords = extract_job_keywords(job_text)

    scoring_keywords = []
    if jd_json and isinstance(jd_json, dict):
        required = jd_json.get("required_skills") or []
        if isinstance(required, list) and required:
            scoring_keywords = _dedupe_keywords_preserve_phrases(required)

    if not scoring_keywords:
        scoring_keywords = _dedupe_keywords_preserve_phrases(job_keywords)

    scoring_keywords = scoring_keywords[:20]

    enriched_keywords = enrich_jd_keywords(list(scoring_keywords))

    skills, keyword_percentage, resume_skills = extract_skills(resume_text_cleaned, scoring_keywords)

    skills_for_analysis = enriched_keywords
    if jd_json:
        structured_skills = extract_skills_from_jd_json(jd_json)
        if structured_skills:
            skills_for_analysis = _dedupe_keywords_preserve_phrases(structured_skills)

    skills_analysis_result = skills_analysis(resume_text_cleaned, skills_for_analysis)
    matched_skills_list = skills_analysis_result.get("matched_skills", skills)
    missing_skills_list = skills_analysis_result.get("missing_skills", [])

    semantic_w, keyword_w = get_matching_weights()
    if semantic_score > 0:
        combined_score = (semantic_score * semantic_w) + (keyword_percentage * keyword_w)
    else:
        combined_score = keyword_percentage

    return {
        "filename": filename,
        "combined_score": combined_score,
        "semantic_score": semantic_score,
        "keyword_score": keyword_percentage,
        "matched_skills": matched_skills_list,
        "missing_skills": missing_skills_list,
        "email": email,
        "phone": phone,
        "resume_text_raw": resume_text_raw,
        "resume_skills": resume_skills,
        "job_keywords": scoring_keywords,
    }


def process_single_resume(file, job_text, filename=None):
    """
    Process a single resume file and return analysis results.
    Used by both individual and batch processing.
    """
    if filename is None:
        filename = getattr(file, "filename", "Unknown")

    resume_text_raw = extract_text_from_resume(file_obj=file)
    return score_resume_vs_job(resume_text_raw, job_text, filename=filename, offline_eval=False)


def _benchmark_json_path():
    return os.path.join(os.path.dirname(__file__), "evaluation", "benchmark_cases.json")


def run_evaluation_benchmark(offline=True):
    """
    Run labeled benchmark cases and return classification metrics.
    offline=True: no Gemini (reproducible for report / lab runs).
    """
    path = _benchmark_json_path()
    if not os.path.isfile(path):
        return {"error": f"Benchmark file not found: {path}", "cases": []}

    try:
        threshold = float(os.getenv("EVAL_THRESHOLD", "55"))
    except Exception:
        threshold = 55.0

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    rows = []
    tp = fp = tn = fn = 0
    n_errors = 0

    for i, c in enumerate(cases):
        rid = c.get("id", i + 1)
        resume_text = (c.get("resume_text") or "").strip()
        jd = (c.get("job_description") or "").strip()
        gt = bool(c.get("ground_truth_relevant", c.get("relevant", True)))

        if not resume_text or not jd:
            n_errors += 1
            rows.append(
                {
                    "id": rid,
                    "error": "missing resume_text or job_description",
                    "ground_truth_relevant": gt,
                }
            )
            continue

        try:
            out = score_resume_vs_job(
                resume_text, jd, filename=f"benchmark_{rid}", offline_eval=offline
            )
            score = float(out["combined_score"])
        except Exception as e:
            n_errors += 1
            rows.append({"id": rid, "error": str(e), "ground_truth_relevant": gt})
            continue

        pred = score >= threshold
        if gt and pred:
            tp += 1
        elif gt and not pred:
            fn += 1
        elif not gt and pred:
            fp += 1
        else:
            tn += 1

        rows.append(
            {
                "id": rid,
                "combined_score": round(score, 2),
                "semantic_score": round(float(out["semantic_score"]), 2),
                "keyword_score": round(float(out["keyword_score"]), 2),
                "predicted_relevant": pred,
                "ground_truth_relevant": gt,
                "correct": pred == gt,
            }
        )

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0

    return {
        "threshold": threshold,
        "offline_eval": offline,
        "cases_in_file": len(cases),
        "cases_scored": total,
        "cases_with_errors": n_errors,
        "accuracy": round(accuracy * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        "cases": rows,
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


@app.route("/evaluation")
def evaluation_page():
    """
    Offline benchmark: no Gemini calls. Tune threshold with env EVAL_THRESHOLD (default 55).
    Append ?format=json for machine-readable output (reports, scripts).
    """
    payload = run_evaluation_benchmark(offline=True)
    if request.args.get("format") == "json":
        return jsonify(payload)
    return render_template("evaluation.html", data=payload)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Individual resume analysis endpoint."""
    if 'resume' not in request.files:
        print("No file uploaded.")
        return "No file uploaded", 400

    file = request.files['resume']
    job_text = request.form['job_description']

    print("Job Description Received:", job_text)

    # Validate file type
    if not file or not file.filename:
        return "No file uploaded", 400
    
    file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
    if file_ext not in ['pdf', 'docx']:
        return f"Unsupported file type. Only PDF and DOCX are allowed. Got: {file_ext}", 400

    try:
        print("Uploaded Resume:", file.filename)
        
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
        print(f"Validation error: {str(e)}")
        return f"Error: {str(e)}", 400
    except Exception as e:
        print(f"Error while analyzing: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", 400


@app.route('/hr/analyze', methods=['POST'])
def hr_analyze():
    """HR batch resume analysis endpoint."""
    if 'resumes' not in request.files:
        print("No files uploaded.")
        return "No files uploaded", 400

    files = request.files.getlist('resumes')
    job_text = request.form['job_description']

    if not files or len(files) == 0:
        return "No files uploaded", 400

    print(f"Job Description Received. Processing {len(files)} resume(s)...")

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
            print(f"Processing: {filename}")
            result = process_single_resume(file, job_text, filename)
            results.append(result)
            print(f"Processed {filename}: Score = {result['combined_score']:.2f}%")
        except Exception as e:
            error_msg = f"{filename}: {str(e)}"
            errors.append(error_msg)
            print(f"Error processing {filename}: {str(e)}")

    if not results:
        return f"No resumes could be processed. Errors: {', '.join(errors)}", 400

    # Sort results by combined_score (highest first)
    results.sort(key=lambda x: x['combined_score'], reverse=True)

    print(f"Batch processing complete: {len(results)} successful, {len(errors)} errors")

    return render_template(
        "hr_results.html",
        results=results,
        errors=errors if errors else None,
    )

if __name__ == '__main__':
    print("Flask server starting on http://127.0.0.1:5000 ...")
    app.run(debug=True)

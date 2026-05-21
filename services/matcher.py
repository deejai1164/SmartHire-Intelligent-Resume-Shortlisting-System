"""
Semantic Matching Engine Service
Uses sentence-transformers for semantic similarity calculation.
Semantic matching is applied only to soft-skills content (not technical skills).
"""

import re
from typing import Dict, List

# Soft-skill keywords used to filter text for semantic matching (technical skills excluded)
SOFT_SKILL_KEYWORDS = {
    "communication", "leadership", "problem solving", "teamwork", "collaboration",
    "agile", "scrum", "project management", "time management", "adaptability",
    "critical thinking", "analytical", "mentoring", "stakeholder", "presentation",
    "written", "verbal", "documentation", "planning", "decision", "ownership",
    "iterative", "kanban", "sprint", "jira", "troubleshooting", "debugging",
    "management", "team lead", "delivery", "timeline", "logic",
}

# Global model instance (loaded once)
_model = None


def _get_model():
    """Load SentenceTransformer model once globally."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence transformer model loaded.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence transformer model: {str(e)}")
    return _model


def get_soft_skills_text(text: str) -> str:
    """
    Extract only soft-skill-related segments from text for semantic matching.
    Technical skills are excluded; only phrases containing soft-skill keywords are kept.
    
    Args:
        text: Full text (resume or job description)
        
    Returns:
        Concatenated soft-skill-related segments, or empty string if none found
    """
    if not text or not text.strip():
        return ""
    text_lower = text.lower()
    # Split into segments (sentences or clauses)
    segments = re.split(r"[.\n?!;]+", text)
    kept = []
    for seg in segments:
        s = seg.strip()
        if len(s) < 3:
            continue
        seg_lower = s.lower()
        if any(kw in seg_lower for kw in SOFT_SKILL_KEYWORDS):
            kept.append(s)
    return " ".join(kept).strip() if kept else ""


def calculate_similarity(resume_text: str, jd_text: str) -> float:
    """
    Calculate semantic similarity between resume and job description using sentence embeddings.
    
    Args:
        resume_text: Cleaned resume text
        jd_text: Job description text (structured or plain)
        
    Returns:
        Similarity score as percentage (0-100)
    """
    if not resume_text or not jd_text:
        return 0.0
    
    try:
        model = _get_model()
        from sentence_transformers import util

        # Encode both texts as tensors for modern vector similarity
        resume_embedding = model.encode(resume_text, convert_to_tensor=True)
        jd_embedding = model.encode(jd_text, convert_to_tensor=True)

        # Cosine similarity (vector-first ranking)
        similarity = util.pytorch_cos_sim(resume_embedding, jd_embedding).item()
        
        # Convert to percentage (0-100)
        percentage = float(similarity * 100)
        
        return round(percentage, 2)
    except Exception as e:
        print(f"Error calculating semantic similarity: {str(e)}")
        return 0.0


def calculate_soft_skills_similarity(resume_text: str, jd_text: str) -> float:
    """
    Semantic similarity applied only to soft-skills content (not technical skills).
    Extracts soft-skill-related segments from both texts, then computes embedding similarity.
    Returns 0 if either side has no soft-skill content.
    
    Args:
        resume_text: Cleaned resume text
        jd_text: Job description text (structured or plain)
        
    Returns:
        Similarity score as percentage (0-100), or 0 if no soft-skill content
    """
    resume_soft = get_soft_skills_text(resume_text)
    jd_soft = get_soft_skills_text(jd_text)
    if not resume_soft or not jd_soft:
        return 0.0
    return calculate_similarity(resume_soft, jd_soft)


def skills_analysis(resume_text: str, required_skills: List[str]) -> Dict[str, List[str]]:
    """
    Analyze which required skills are present in the resume (keyword-based matching).
    This is NOT embedding-based - it's simple keyword presence detection for UI display.
    
    Args:
        resume_text: Cleaned resume text (will be lowercased for matching)
        required_skills: List of skill keywords to check for
        
    Returns:
        Dictionary with 'matched_skills' and 'missing_skills' lists
    """
    if not resume_text:
        return {
            "matched_skills": [],
            "missing_skills": required_skills.copy() if required_skills else []
        }
    
    if not required_skills:
        return {
            "matched_skills": [],
            "missing_skills": []
        }
    
    resume_lower = resume_text.lower()
    matched_skills = []
    missing_skills = []
    
    for skill in required_skills:
        if not skill:
            continue
        
        # Normalize skill for matching (lowercase, strip)
        skill_normalized = skill.lower().strip()
        
        # Check if skill keyword appears in resume
        # Simple substring match (can be enhanced with word boundaries)
        if skill_normalized in resume_lower:
            matched_skills.append(skill)
        else:
            # Also check for partial matches (e.g., "machine learning" matches "machine learning engineer")
            # Split skill into words and check if all words appear
            skill_words = skill_normalized.split()
            if len(skill_words) > 1:
                # For multi-word skills, check if all words appear (not necessarily together)
                if all(word in resume_lower for word in skill_words if len(word) > 2):
                    matched_skills.append(skill)
                else:
                    missing_skills.append(skill)
            else:
                missing_skills.append(skill)
    
    return {
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }

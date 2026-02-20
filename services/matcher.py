"""
Semantic Matching Engine Service
Uses sentence-transformers for semantic similarity calculation.
"""

import os
from typing import Dict, List
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Global model instance (loaded once)
_model = None


def _get_model():
    """Load SentenceTransformer model once globally."""
    global _model
    if _model is None:
        try:
            from sentence_transformers import SentenceTransformer
            print("🔄 Loading sentence transformer model (all-MiniLM-L6-v2)...")
            _model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Model loaded successfully.")
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load sentence transformer model: {str(e)}")
    return _model


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
        
        # Encode both texts
        resume_embedding = model.encode([resume_text], convert_to_numpy=True)
        jd_embedding = model.encode([jd_text], convert_to_numpy=True)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(resume_embedding, jd_embedding)[0][0]
        
        # Convert to percentage (0-100)
        percentage = float(similarity * 100)
        
        return round(percentage, 2)
    except Exception as e:
        print(f"🔥 Error calculating semantic similarity: {str(e)}")
        return 0.0


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

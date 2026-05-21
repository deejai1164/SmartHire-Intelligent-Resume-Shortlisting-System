"""
Job Description Parser Service
Handles structured JD JSON from Gemini and converts it to text for semantic matching.
"""


def build_structured_jd_text(jd_json: dict) -> str:
    """
    Convert structured JD JSON into a clean text paragraph for embedding similarity.
    
    Expected JSON format:
    {
        "job_role": "Machine Learning Engineer",
        "required_skills": ["Python", "Machine Learning", "NLP", "scikit-learn"],
        "optional_skills": ["TensorFlow", "Deep Learning"],
        "minimum_experience_years": 2,
        "domain": "Natural Language Processing",
        "tools_and_technologies": ["Git", "Docker"]
    }
    
    Args:
        jd_json: Dictionary containing structured job description data
        
    Returns:
        Clean text string formatted for semantic matching
    """
    parts = []
    
    # Job role
    if jd_json.get("job_role"):
        parts.append(f"Role: {jd_json['job_role']}")
    
    # Required skills
    if jd_json.get("required_skills"):
        skills_str = ", ".join(jd_json["required_skills"])
        parts.append(f"Required Skills: {skills_str}")
    
    # Optional skills
    if jd_json.get("optional_skills"):
        optional_str = ", ".join(jd_json["optional_skills"])
        parts.append(f"Optional Skills: {optional_str}")
    
    # Domain
    if jd_json.get("domain"):
        parts.append(f"Domain: {jd_json['domain']}")
    
    # Tools and technologies
    if jd_json.get("tools_and_technologies"):
        tools_str = ", ".join(jd_json["tools_and_technologies"])
        parts.append(f"Tools: {tools_str}")
    
    # Experience
    if jd_json.get("minimum_experience_years"):
        parts.append(f"Minimum Experience: {jd_json['minimum_experience_years']} years")
    
    # If no structured data, return empty string
    if not parts:
        return ""
    
    return "\n".join(parts)


def extract_skills_from_jd_json(jd_json: dict) -> list:
    """
    Extract all skills (required + optional) from JD JSON.
    
    Args:
        jd_json: Dictionary containing structured job description data
        
    Returns:
        List of all skill strings (required + optional combined)
    """
    skills = []
    
    if jd_json.get("required_skills"):
        skills.extend(jd_json["required_skills"])
    
    if jd_json.get("optional_skills"):
        skills.extend(jd_json["optional_skills"])
    
    return skills

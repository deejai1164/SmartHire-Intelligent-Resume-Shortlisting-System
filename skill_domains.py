# Domain-to-related-skills mapping for smarter resume matching.
# When a job asks for "Data Science", we treat resume mentions of ML, Pandas, etc. as a match.
# All keys and values are normalized to lowercase for matching.

DOMAIN_SKILL_MAP = {
    # Data Science (broad)
    "data science": {
        "machine learning", "ml", "artificial intelligence", "ai", "python", "numpy", "pandas",
        "scipy", "scikit-learn", "sklearn", "statistics", "statistical", "sql", "data analysis",
        "data analytics", "visualization", "matplotlib", "seaborn", "jupyter", "tensorflow",
        "pytorch", "keras", "deep learning", "neural network", "regression", "classification",
        "clustering", "nlp", "natural language processing", "data mining", "big data", "spark",
        "hadoop", "excel", "r", "r programming", "tableau", "power bi", "powerbi", "etl",
        "data pipeline", "data modeling", "data visualization", "exploratory analysis", "eda",
        "hypothesis testing", "a/b testing", "ab testing", "feature engineering", "modeling",
    },
    "data scientist": {
        "machine learning", "ml", "python", "numpy", "pandas", "sql", "statistics", "r",
        "scikit-learn", "tensorflow", "pytorch", "data analysis", "visualization", "tableau",
    },
    "data analytics": {
        "sql", "python", "excel", "tableau", "power bi", "statistics", "reporting", "dashboard",
        "etl", "data visualization", "pandas", "numpy",
    },
    "data analyst": {
        "sql", "excel", "tableau", "power bi", "python", "pandas", "statistics", "reporting",
    },

    # Machine Learning / AI
    "machine learning": {
        "ml", "python", "scikit-learn", "sklearn", "tensorflow", "pytorch", "keras", "xgboost",
        "regression", "classification", "clustering", "nlp", "deep learning", "neural network",
        "feature engineering", "model training", "cross-validation", "pandas", "numpy", "data science",
        "artificial intelligence", "ai", "supervised", "unsupervised", "reinforcement learning",
    },
    "artificial intelligence": {
        "ai", "machine learning", "ml", "deep learning", "neural network", "nlp", "computer vision",
        "tensorflow", "pytorch", "keras", "python", "reinforcement learning", "natural language",
    },
    "deep learning": {
        "neural network", "tensorflow", "pytorch", "keras", "cnn", "rnn", "lstm", "transformer",
        "computer vision", "nlp", "python", "gpu", "cuda",
    },
    "nlp": {
        "natural language processing", "text mining", "spacy", "nltk", "transformers", "bert",
        "tokenization", "sentiment analysis", "ner", "named entity", "machine learning", "python",
    },
    "natural language processing": {
        "nlp", "spacy", "nltk", "text mining", "bert", "transformers", "tokenization", "python",
    },
    "computer vision": {
        "opencv", "image processing", "cnn", "tensorflow", "pytorch", "keras", "deep learning",
    },

    # Programming languages (expand to common ecosystems)
    "python": {
        "django", "flask", "fastapi", "numpy", "pandas", "scikit-learn", "tensorflow", "pytorch",
        "jupyter", "pip", "conda", "python3", "scripting", "automation",
    },
    "java": {
        "spring", "spring boot", "maven", "gradle", "jvm", "junit", "android", "kotlin",
    },
    "javascript": {
        "js", "node", "nodejs", "react", "angular", "vue", "typescript", "frontend", "jquery",
    },
    "r": {
        "r programming", "ggplot2", "dplyr", "tidyr", "shiny", "statistics", "data analysis",
    },
    "sql": {
        "mysql", "postgresql", "postgres", "sqlite", "query", "database", "etl", "joins",
        "stored procedures", "query optimization",
    },

    # Web development
    "web development": {
        "html", "css", "javascript", "react", "angular", "vue", "node", "django", "flask",
        "rest", "api", "frontend", "backend", "responsive", "bootstrap", "jquery", "typescript",
    },
    "frontend": {
        "html", "css", "javascript", "react", "angular", "vue", "bootstrap", "responsive", "ui",
    },
    "backend": {
        "node", "django", "flask", "fastapi", "rest", "api", "database", "sql", "python", "java",
    },
    "full stack": {
        "frontend", "backend", "react", "node", "javascript", "python", "sql", "api",
    },
    "react": {
        "redux", "hooks", "javascript", "frontend", "node", "npm",
    },
    "django": {
        "python", "rest", "api", "orm", "backend", "mysql", "postgresql",
    },
    "flask": {
        "python", "rest", "api", "backend", "microframework",
    },

    # DevOps / Cloud / Tools
    "devops": {
        "docker", "kubernetes", "k8s", "jenkins", "ci/cd", "cicd", "aws", "azure", "gcp",
        "linux", "bash", "terraform", "ansible", "git", "github", "gitlab",
    },
    "aws": {
        "amazon web services", "ec2", "s3", "lambda", "cloud", "devops", "docker",
    },
    "docker": {
        "containers", "kubernetes", "devops", "ci/cd", "deployment",
    },
    "kubernetes": {
        "k8s", "containers", "docker", "orchestration", "devops",
    },
    "git": {
        "github", "gitlab", "version control", "branching", "merge", "ci/cd",
    },

    # Soft skills / domains (expand to common related terms)
    "communication": {
        "written", "verbal", "presentation", "stakeholder", "collaboration", "team", "documentation",
    },
    "leadership": {
        "team lead", "mentoring", "management", "project", "ownership", "decision",
    },
    "problem solving": {
        "analytical", "debugging", "troubleshooting", "critical thinking", "logic",
    },
    "agile": {
        "scrum", "sprint", "jira", "kanban", "iterative", "project management",
    },
    "project management": {
        "agile", "scrum", "jira", "planning", "stakeholder", "timeline", "delivery",
    },

    # Standalone terms that map to broader domains
    "statistics": {
        "statistical", "regression", "hypothesis", "probability", "r", "python", "pandas", "analysis",
    },
    "analytics": {
        "data analysis", "sql", "excel", "tableau", "reporting", "dashboard", "metrics", "kpi",
    },
    "excel": {
        "spreadsheet", "vlookup", "pivot", "formulas", "data analysis", "reporting",
    },
    "tableau": {
        "visualization", "dashboard", "bi", "reporting", "data visualization",
    },
    "visualization": {
        "matplotlib", "seaborn", "tableau", "power bi", "plotting", "charts", "dashboard",
    },
}

# Normalize key: strip, lower, collapse spaces
def _normalize(s):
    return " ".join(s.lower().strip().split())


# Build flat set of all single-word skills and multi-word phrases for resume extraction
ALL_KNOWN_SKILL_TOKENS = set()
KNOWN_SKILL_PHRASES = set()  # bigrams/trigrams to detect in resume text

for domain, skills in DOMAIN_SKILL_MAP.items():
    ALL_KNOWN_SKILL_TOKENS.add(_normalize(domain))
    for s in skills:
        normalized = _normalize(s)
        ALL_KNOWN_SKILL_TOKENS.add(normalized)
        if " " in normalized:
            KNOWN_SKILL_PHRASES.add(normalized)
        else:
            ALL_KNOWN_SKILL_TOKENS.add(normalized)

# Single-word tokens only (for quick token lookup from resume)
ALL_KNOWN_SINGLE_WORDS = {w for w in ALL_KNOWN_SKILL_TOKENS if " " not in w}


def get_domain_related_skills(keyword):
    """Return set of related skills for a job keyword (e.g. 'data science' -> pandas, ml, ...)."""
    key = _normalize(keyword)
    return set(DOMAIN_SKILL_MAP.get(key, set()))


def get_all_expanded_terms(keyword):
    """Return the keyword itself plus all domain-related terms (for matching)."""
    key = _normalize(keyword)
    related = get_domain_related_skills(keyword)
    out = {key}
    out.update(related)
    return out

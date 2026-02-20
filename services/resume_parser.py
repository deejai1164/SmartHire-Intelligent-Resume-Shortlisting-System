"""
Resume Parser Service
Handles text extraction from PDF and DOCX files, and text cleaning.
"""

import re
import fitz  # PyMuPDF (already in use)
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None


def extract_text_from_resume(file_path: str = None, file_obj=None) -> str:
    """
    Extract text from resume file (PDF or DOCX).
    
    Args:
        file_path: Path to the resume file (optional if file_obj provided)
        file_obj: File-like object (optional if file_path provided)
        
    Returns:
        Extracted plain text from the resume
        
    Raises:
        ValueError: If file type is unsupported or extraction fails
    """
    # If file_obj is provided, use it; otherwise use file_path
    if file_obj is not None:
        filename = getattr(file_obj, 'filename', '')
        if filename.endswith('.pdf'):
            return _extract_pdf_text_from_file_obj(file_obj)
        elif filename.endswith('.docx'):
            if file_path:
                return _extract_docx_text(file_path)
            else:
                raise ValueError("DOCX extraction requires file_path, not file_obj")
        else:
            raise ValueError(f"Unsupported file type: {filename}")
    
    if not file_path:
        raise ValueError("Either file_path or file_obj must be provided")
    
    if file_path.endswith('.pdf'):
        return _extract_pdf_text(file_path)
    elif file_path.endswith('.docx'):
        return _extract_docx_text(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def _extract_pdf_text(file_path: str) -> str:
    """Extract text from PDF using PyMuPDF (fitz)."""
    try:
        pdf = fitz.open(file_path)
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract PDF text: {str(e)}")


def _extract_pdf_text_from_file_obj(file_obj) -> str:
    """Extract text from PDF file object using PyMuPDF (fitz)."""
    try:
        # Reset file pointer
        file_obj.seek(0)
        pdf = fitz.open(stream=file_obj.read(), filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        pdf.close()
        return text
    except Exception as e:
        raise ValueError(f"Failed to extract PDF text from file object: {str(e)}")


def _extract_docx_text(file_path: str) -> str:
    """Extract text from DOCX file."""
    if Document is None:
        raise ValueError("python-docx is not installed. Install it with: pip install python-docx")
    
    try:
        doc = Document(file_path)
        text = []
        for paragraph in doc.paragraphs:
            text.append(paragraph.text)
        return "\n".join(text)
    except Exception as e:
        raise ValueError(f"Failed to extract DOCX text: {str(e)}")


def extract_email(text: str) -> str:
    """
    Extract email address from resume text.
    
    Args:
        text: Raw resume text
        
    Returns:
        Email address string or None if not found
    """
    if not text:
        return None
    
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    matches = re.findall(email_pattern, text)
    if matches:
        return matches[0]  # Return first email found
    return None


def extract_phone(text: str) -> str:
    """
    Extract phone number from resume text.
    
    Args:
        text: Raw resume text
        
    Returns:
        Phone number string or None if not found
    """
    if not text:
        return None
    
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890 or 123.456.7890
        r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',  # (123) 456-7890
        r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',  # International
    ]
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[0]  # Return first phone found
    
    return None


def clean_resume_text(text: str) -> str:
    """
    Clean resume text by removing:
    - Email addresses
    - Phone numbers
    - Addresses (basic regex)
    - Extra whitespace
    
    Args:
        text: Raw resume text
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""
    
    cleaned = text
    
    # Remove email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    cleaned = re.sub(email_pattern, '', cleaned)
    
    # Remove phone numbers (various formats)
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # 123-456-7890 or 123.456.7890
        r'\b\(\d{3}\)\s?\d{3}[-.]?\d{4}\b',  # (123) 456-7890
        r'\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',  # International
    ]
    for pattern in phone_patterns:
        cleaned = re.sub(pattern, '', cleaned)
    
    # Remove addresses (basic - looks for common address patterns)
    # This is a simple heuristic - may not catch all addresses
    address_patterns = [
        r'\b\d+\s+[A-Za-z0-9\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Way|Circle|Cir)\b',
        r'\b(?:P\.?O\.?\s*Box|PO Box)\s+\d+\b',
    ]
    for pattern in address_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single space
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Multiple newlines to single
    cleaned = cleaned.strip()
    
    return cleaned

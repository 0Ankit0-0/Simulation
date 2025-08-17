import os
import logging
from datetime import datetime
from .document_parser import extract_text_from_file

logger = logging.getLogger(__name__)

def parse_evidence(filepath: str) -> dict:
    """
    Enhanced evidence parsing with OCR capabilities
    
    Args:
        filepath: Path to the evidence file
        
    Returns:
        Dictionary containing parsed evidence data
    """
    try:
        # Extract text using the enhanced parser
        extraction_result = extract_text_from_file(filepath)
        
        # Get file metadata
        filename = os.path.basename(filepath)
        file_size = os.path.getsize(filepath)
        file_ext = os.path.splitext(filepath)[1].lower()
        
        # Create evidence object
        evidence = {
            "filename": filename,
            "filepath": filepath,
            "file_type": file_ext,
            "file_size": file_size,
            "extracted_text": extraction_result.get('text', ''),
            "extraction_confidence": extraction_result.get('confidence', 0.0),
            "extraction_method": extraction_result.get('method', 'unknown'),
            "processed_at": datetime.now().isoformat(),
            "text_length": len(extraction_result.get('text', '')),
            "extraction_metadata": {
                key: value for key, value in extraction_result.items() 
                if key not in ['text', 'confidence', 'method']
            }
        }
        
        # Add quality assessment
        evidence["quality_assessment"] = _assess_extraction_quality(extraction_result)
        
        logger.info(f"Successfully parsed evidence: {filename} using {evidence['extraction_method']}")
        return evidence
        
    except Exception as e:
        logger.error(f"Error parsing evidence {filepath}: {e}")
        return {
            "filename": os.path.basename(filepath) if filepath else "unknown",
            "filepath": filepath,
            "error": str(e),
            "extracted_text": "",
            "extraction_confidence": 0.0,
            "extraction_method": "failed",
            "processed_at": datetime.now().isoformat(),
            "quality_assessment": "failed"
        }

def _assess_extraction_quality(extraction_result: dict) -> str:
    """
    Assess the quality of text extraction
    
    Args:
        extraction_result: Result from document parser
        
    Returns:
        Quality assessment string
    """
    confidence = extraction_result.get('confidence', 0.0)
    text = extraction_result.get('text', '')
    method = extraction_result.get('method', '')
    
    if 'failed' in method or 'error' in extraction_result:
        return 'failed'
    elif confidence >= 0.9 and len(text) > 100:
        return 'excellent'
    elif confidence >= 0.7 and len(text) > 50:
        return 'good'
    elif confidence >= 0.5 and len(text) > 20:
        return 'fair'
    else:
        return 'poor'
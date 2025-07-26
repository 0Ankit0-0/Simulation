import os
import logging
import magic
import fitz
import docx
import easyocr
import torch
from PIL import Image
from transformers import pipeline
from contextlib import contextmanager
from datetime import datetime
import tempfile
import hashlib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvidenceParser:
    """Optimized evidence parser with lazy loading and error handling"""
    
    def __init__(self):
        self._ocr_reader = None
        self._summarizer = None
        self.max_text_length = 50000  # Limit text processing
        self.max_summary_length = 200
    
    @property
    def ocr_reader(self):
        """Lazy load OCR reader"""
        if self._ocr_reader is None:
            try:
                self._ocr_reader = easyocr.Reader(
                    ["en"], 
                    gpu=torch.cuda.is_available(),
                    verbose=False
                )
                logger.info("OCR reader initialized")
            except Exception as e:
                logger.error(f"Failed to initialize OCR reader: {e}")
                raise
        return self._ocr_reader
    
    @property
    def summarizer(self):
        """Lazy load summarizer"""
        if self._summarizer is None:
            try:
                self._summarizer = pipeline(
                    "summarization",
                    model="t5-small",
                    tokenizer="t5-small",
                    framework="pt",
                    device=0 if torch.cuda.is_available() else -1
                )
                logger.info("Summarizer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize summarizer: {e}")
                # Fallback to CPU
                try:
                    self._summarizer = pipeline(
                        "summarization",
                        model="t5-small",
                        tokenizer="t5-small",
                        framework="pt",
                        device=-1
                    )
                    logger.info("Summarizer initialized on CPU")
                except Exception as fallback_error:
                    logger.error(f"Failed to initialize summarizer on CPU: {fallback_error}")
                    self._summarizer = None
        return self._summarizer
    
    def get_file_type(self, filepath):
        """Detect actual file type using python-magic"""
        try:
            # Use python-magic for accurate file type detection
            mime_type = magic.from_file(filepath, mime=True)
            return mime_type
        except Exception as e:
            logger.warning(f"Could not detect MIME type for {filepath}: {e}")
            # Fallback to extension-based detection
            return filepath.lower().split('.')[-1]
    
    def validate_file_integrity(self, filepath):
        """Basic file integrity check"""
        try:
            if not os.path.exists(filepath):
                return False, "File does not exist"
            
            if os.path.getsize(filepath) == 0:
                return False, "File is empty"
            
            # Check if file is readable
            with open(filepath, 'rb') as f:
                f.read(1024)  # Try to read first 1KB
            
            return True, "File is valid"
        
        except Exception as e:
            return False, f"File validation error: {str(e)}"
    
    def calculate_file_hash(self, filepath):
        """Calculate file hash for integrity and deduplication"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating file hash: {e}")
            return None
    
    def parse_pdf(self, filepath):
        """Parse PDF with enhanced error handling"""
        try:
            # Validate file first
            is_valid, validation_error = self.validate_file_integrity(filepath)
            if not is_valid:
                raise ValueError(validation_error)
            
            doc = fitz.open(filepath)
            text = ""
            page_count = len(doc)
            
            # Limit processing for very large PDFs
            max_pages = 100
            if page_count > max_pages:
                logger.warning(f"PDF has {page_count} pages, limiting to {max_pages}")
                pages_to_process = max_pages
            else:
                pages_to_process = page_count
            
            for page_num in range(pages_to_process):
                try:
                    page = doc[page_num]
                    page_text = page.get_text()
                    text += page_text + "\n"
                    
                    # Stop if text is getting too long
                    if len(text) > self.max_text_length:
                        text = text[:self.max_text_length]
                        logger.warning(f"PDF text truncated at {self.max_text_length} characters")
                        break
                        
                except Exception as page_error:
                    logger.warning(f"Error processing page {page_num}: {page_error}")
                    continue
            
            doc.close()
            
            return {
                "type": "pdf",
                "filename": os.path.basename(filepath),
                "text": text.strip(),
                "pages_processed": pages_to_process,
                "total_pages": page_count,
                "file_hash": self.calculate_file_hash(filepath)
            }
            
        except Exception as e:
            logger.error(f"Error parsing PDF {filepath}: {e}")
            return {
                "type": "pdf",
                "filename": os.path.basename(filepath),
                "text": "",
                "error": f"PDF parsing failed: {str(e)}",
                "file_hash": self.calculate_file_hash(filepath)
            }
    
    def parse_docx(self, filepath):
        """Parse DOCX with enhanced error handling"""
        try:
            # Validate file first
            is_valid, validation_error = self.validate_file_integrity(filepath)
            if not is_valid:
                raise ValueError(validation_error)
            
            doc = docx.Document(filepath)
            paragraphs = []
            
            for para in doc.paragraphs:
                if para.text.strip():  # Only add non-empty paragraphs
                    paragraphs.append(para.text.strip())
                    
                    # Stop if text is getting too long
                    current_text = "\n".join(paragraphs)
                    if len(current_text) > self.max_text_length:
                        current_text = current_text[:self.max_text_length]
                        logger.warning(f"DOCX text truncated at {self.max_text_length} characters")
                        break
            
            text = "\n".join(paragraphs)
            
            # Extract additional metadata
            core_props = doc.core_properties
            metadata = {
                "author": core_props.author,
                "created": str(core_props.created) if core_props.created else None,
                "modified": str(core_props.modified) if core_props.modified else None,
                "title": core_props.title
            }
            
            return {
                "type": "docx",
                "filename": os.path.basename(filepath),
                "text": text,
                "paragraph_count": len(paragraphs),
                "metadata": metadata,
                "file_hash": self.calculate_file_hash(filepath)
            }
            
        except Exception as e:
            logger.error(f"Error parsing DOCX {filepath}: {e}")
            return {
                "type": "docx",
                "filename": os.path.basename(filepath),
                "text": "",
                "error": f"DOCX parsing failed: {str(e)}",
                "file_hash": self.calculate_file_hash(filepath)
            }
    
    def parse_image(self, filepath):
        """Parse image with enhanced OCR and error handling"""
        try:
            # Validate file first
            is_valid, validation_error = self.validate_file_integrity(filepath)
            if not is_valid:
                raise ValueError(validation_error)
            
            # Validate image file
            try:
                with Image.open(filepath) as img:
                    img.verify()  # Verify it's a valid image
            except Exception as img_error:
                raise ValueError(f"Invalid image file: {img_error}")
            
            # Re-open for processing (verify() closes the image)
            with Image.open(filepath) as img:
                # Get image metadata
                width, height = img.size
                img_format = img.format
                img_mode = img.mode
                
                # Convert to RGB if necessary
                if img_mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize if image is too large (for memory efficiency)
                max_dimension = 2000
                if max(width, height) > max_dimension:
                    img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                    logger.info(f"Image resized from {width}x{height} to {img.size}")
            
            # Perform OCR
            try:
                result = self.ocr_reader.readtext(filepath, detail=0, paragraph=True)
                text = "\n".join(result) if result else ""
                
                # Additional OCR with different settings if first attempt yields little text
                if len(text.strip()) < 10:
                    result_detailed = self.ocr_reader.readtext(filepath, detail=0, paragraph=False)
                    text_detailed = " ".join(result_detailed) if result_detailed else ""
                    if len(text_detailed) > len(text):
                        text = text_detailed
                
                confidence_scores = []
                if result:
                    detailed_result = self.ocr_reader.readtext(filepath, detail=1)
                    confidence_scores = [item[2] for item in detailed_result if len(item) > 2]
                
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                
            except Exception as ocr_error:
                logger.error(f"OCR failed for {filepath}: {ocr_error}")
                text = ""
                avg_confidence = 0
            
            return {
                "type": "image",
                "filename": os.path.basename(filepath),
                "text": text.strip(),
                "ocr_confidence": avg_confidence,
                "image_info": {
                    "width": width,
                    "height": height,
                    "format": img_format,
                    "mode": img_mode
                },
                "file_hash": self.calculate_file_hash(filepath)
            }
            
        except Exception as e:
            logger.error(f"Error parsing image {filepath}: {e}")
            return {
                "type": "image",
                "filename": os.path.basename(filepath),
                "text": "",
                "error": f"Image parsing failed: {str(e)}",
                "file_hash": self.calculate_file_hash(filepath)
            }
    
    def generate_summary(self, text, max_length=None):
        """Generate summary with fallback options"""
        if not text or not text.strip():
            return ""
        
        # Use instance max_length if not provided
        if max_length is None:
            max_length = self.max_summary_length
        
        # Clean and prepare text
        text = text.strip()
        
        # If text is shorter than desired summary, return as is
        if len(text) <= max_length:
            return text
        
        try:
            # Try AI summarization first
            summarizer = self.summarizer
            if summarizer is not None:
                # Limit input text for summarization (models have token limits)
                max_input_length = 1500
                input_text = text[:max_input_length] if len(text) > max_input_length else text
                
                result = summarizer(
                    input_text,
                    max_length=min(max_length, 150),  # T5 model limits
                    min_length=min(50, max_length // 2),
                    do_sample=False,
                    truncation=True
                )
                
                if result and len(result) > 0 and 'summary_text' in result[0]:
                    summary = result[0]['summary_text'].strip()
                    if summary:
                        return summary
            
            # Fallback to extractive summary
            return self.extractive_summary(text, max_length)
            
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            # Final fallback: simple truncation
            return self.extractive_summary(text, max_length)
    
    def extractive_summary(self, text, max_length):
        """Simple extractive summary as fallback"""
        sentences = text.replace('\n', ' ').split('. ')
        
        # Clean sentences
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        
        if not sentences:
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # Build summary by adding sentences until we hit length limit
        summary = ""
        for sentence in sentences:
            if len(summary + sentence) <= max_length:
                summary += sentence + " "
            else:
                break
        
        return summary.strip() or (text[:max_length] + "..." if len(text) > max_length else text)
    
    def parse_evidence(self, filepath):
        """Main evidence parsing function with comprehensive error handling"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File not found: {filepath}")
            
            # Get file extension and MIME type
            ext = filepath.lower().split(".")[-1]
            mime_type = self.get_file_type(filepath)
            
            logger.info(f"Processing file: {os.path.basename(filepath)} (type: {ext}, mime: {mime_type})")
            
            # Parse based on file type
            if ext == "pdf" or "pdf" in mime_type:
                parsed = self.parse_pdf(filepath)
            elif ext == "docx" or "officedocument" in mime_type:
                parsed = self.parse_docx(filepath)
            elif ext in ["jpg", "jpeg", "png"] or "image" in mime_type:
                parsed = self.parse_image(filepath)
            else:
                logger.warning(f"Unsupported file type: {ext} (mime: {mime_type})")
                parsed = {
                    "type": "unknown",
                    "filename": os.path.basename(filepath),
                    "text": "",
                    "error": f"Unsupported file type: {ext}",
                    "file_hash": self.calculate_file_hash(filepath)
                }
            
            # Add common metadata
            parsed["processed_at"] = datetime.now().isoformat()
            parsed["file_size"] = os.path.getsize(filepath)
            parsed["mime_type"] = mime_type
            
            # Generate summary if text was extracted successfully
            if parsed.get("text") and not parsed.get("error"):
                try:
                    parsed["summary"] = self.generate_summary(parsed["text"])
                    parsed["character_count"] = len(parsed["text"])
                    parsed["word_count"] = len(parsed["text"].split())
                except Exception as summary_error:
                    logger.error(f"Summary generation failed: {summary_error}")
                    parsed["summary"] = ""
            else:
                parsed["summary"] = ""
                parsed["character_count"] = 0
                parsed["word_count"] = 0
            
            # Add processing status
            parsed["processing_status"] = "completed" if not parsed.get("error") else "failed"
            
            return parsed
            
        except Exception as e:
            logger.error(f"Critical error processing {filepath}: {e}")
            return {
                "type": "error",
                "filename": os.path.basename(filepath) if filepath else "unknown",
                "text": "",
                "error": f"Processing failed: {str(e)}",
                "processed_at": datetime.now().isoformat(),
                "processing_status": "failed",
                "file_hash": self.calculate_file_hash(filepath) if filepath and os.path.exists(filepath) else None
            }

# Global parser instance
parser_instance = EvidenceParser()

def parse_evidence(filepath):
    """Public function to parse evidence files"""
    return parser_instance.parse_evidence(filepath)

# Cleanup function for testing/maintenance
def cleanup_resources():
    """Clean up loaded models and resources"""
    global parser_instance
    if parser_instance._ocr_reader is not None:
        del parser_instance._ocr_reader
        parser_instance._ocr_reader = None
    
    if parser_instance._summarizer is not None:
        del parser_instance._summarizer
        parser_instance._summarizer = None
    
    # Force garbage collection
    import gc
    gc.collect()
    
    logger.info("Parser resources cleaned up")

# Context manager for temporary resource management
@contextmanager
def temporary_parser():
    """Context manager for temporary parser instance"""
    temp_parser = EvidenceParser()
    try:
        yield temp_parser
    finally:
        # Cleanup temporary resources
        if temp_parser._ocr_reader is not None:
            del temp_parser._ocr_reader
        if temp_parser._summarizer is not None:
            del temp_parser._summarizer
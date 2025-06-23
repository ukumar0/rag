import hashlib
from datetime import datetime
import langdetect
import re
import spacy
from typing import Dict, Any

# Load spaCy model for entity extraction
nlp = spacy.load("en_core_web_sm")

class MetadataEnhancer:
    def enhance(self, text: str, base_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enhance document metadata with rich features.
        
        Args:
            text: The text content to analyze
            base_metadata: Optional base metadata to extend
            
        Returns:
            Dict containing enhanced metadata
        """
        if base_metadata is None:
            base_metadata = {}
            
        enhanced_metadata = {
            "doc_type": self._detect_document_type(text),
            "language": self._detect_language(text),
            "key_entities": self._extract_entities(text),
            "section_headers": self._extract_headers(text),
            "content_hash": hashlib.md5(text.encode()).hexdigest(),
            "word_count": len(text.split()),
            "created_at": datetime.now().isoformat(),
        }
        
        # Merge base metadata with enhanced metadata
        return {**base_metadata, **enhanced_metadata}

    def _detect_language(self, text):
        try:
            return langdetect.detect(text)
        except:
            return "unknown"

    def _detect_document_type(self, text):
        if "invoice" in text.lower():
            return "invoice"
        elif "contract" in text.lower():
            return "contract"
        else:
            return "generic"

    def _extract_entities(self, text):
        doc = nlp(text)
        return list(set([ent.text for ent in doc.ents]))

    def _extract_headers(self, text):
        return re.findall(r"\n[0-9]?\s*([A-Z][A-Za-z\s]+):", text)

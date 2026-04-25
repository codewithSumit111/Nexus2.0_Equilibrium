"""
nexus_engine/privacy_guard.py — Nexus Privacy Guard
====================================================
Strict PII masking guardrail for SAR Narrative Generator.

Uses regex to identify and tokenize:
  - Names (e.g., "John Doe", "Jane Smith")
  - Account Numbers (12-digit numbers)
  - SSNs (XXX-XX-XXXX format)

Replaces with reversible tokens like [ENTITY_1], [ACCT_A], [SSN_1].
"""

import re
from typing import Tuple, Dict


class NexusPrivacyGuard:
    """
    Privacy guard for PII masking/unmasking in SAR narratives.
    
    Uses regex patterns to identify sensitive data and replace with
    reversible tokens before LLM processing.
    """
    
    def __init__(self):
        # Counter state for generating unique tokens
        self._entity_counter = 0
        self._account_counter = 0
        self._ssn_counter = 0
    
    def _get_entity_token(self) -> str:
        """Generate next entity token like [ENTITY_1]."""
        self._entity_counter += 1
        return f"[ENTITY_{self._entity_counter}]"
    
    def _get_account_token(self) -> str:
        """Generate next account token like [ACCT_A]."""
        self._account_counter += 1
        # Use A, B, C... then AA, AB, etc.
        token = ""
        n = self._account_counter - 1
        while n >= 0:
            token = chr(ord('A') + (n % 26)) + token
            n = n // 26 - 1
        return f"[ACCT_{token or 'A'}]"
    
    def _get_ssn_token(self) -> str:
        """Generate next SSN token like [SSN_1]."""
        self._ssn_counter += 1
        return f"[SSN_{self._ssn_counter}]"
    
    def mask_pii(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Mask PII in text using regex patterns.
        
        Args:
            text: Raw input text potentially containing PII
            
        Returns:
            Tuple of (masked_text, mapping_dict)
            mapping_dict: {token: original_value} for reversal
        """
        if not text:
            return text, {}
        
        mapping: Dict[str, str] = {}
        masked_text = text
        
        # Pattern 1: SSNs (XXX-XX-XXXX or XXX XX XXXX)
        ssn_pattern = r'\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b'
        for match in re.finditer(ssn_pattern, masked_text):
            original = match.group(0)
            token = self._get_ssn_token()
            mapping[token] = original
            masked_text = masked_text.replace(original, token, 1)
        
        # Pattern 2: 12-digit Account Numbers
        # Matches standalone 12-digit numbers (with or without spaces/hyphens)
        acct_pattern = r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b|\b\d{12}\b'
        for match in re.finditer(acct_pattern, masked_text):
            original = match.group(0)
            # Skip if already replaced as SSN (12 digits could match both)
            if not any(token in original for token in mapping.keys()):
                token = self._get_account_token()
                mapping[token] = original
                masked_text = masked_text.replace(original, token, 1)
        
        # Pattern 3: Names (Title Case, 2-3 words)
        # Matches patterns like "John Doe", "Jane Smith", "Robert Downey Jr"
        # Avoids common false positives by requiring title case
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b'
        for match in re.finditer(name_pattern, masked_text):
            original = match.group(0)
            # Skip if looks like a token already
            if original.startswith('[') and original.endswith(']'):
                continue
            token = self._get_entity_token()
            mapping[token] = original
            masked_text = masked_text.replace(original, token, 1)
        
        return masked_text, mapping
    
    def unmask_pii(self, text: str, mapping: Dict[str, str]) -> str:
        """
        Reverse PII masking using the mapping dictionary.
        
        Args:
            text: Masked text with tokens like [ENTITY_1]
            mapping: {token: original_value} from mask_pii()
            
        Returns:
            Unmasked text with original PII restored
        """
        if not text or not mapping:
            return text
        
        unmasked_text = text
        
        # Replace tokens in reverse order (highest number first)
        # to avoid partial replacements
        sorted_tokens = sorted(
            mapping.keys(),
            key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0,
            reverse=True
        )
        
        for token in sorted_tokens:
            original = mapping[token]
            unmasked_text = unmasked_text.replace(token, original)
        
        return unmasked_text


# Convenience function for direct import
def create_guard() -> NexusPrivacyGuard:
    """Factory function to create a fresh NexusPrivacyGuard instance."""
    return NexusPrivacyGuard()

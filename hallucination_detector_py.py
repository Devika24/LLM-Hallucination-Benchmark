"""
Hallucination Detection Module

This module implements multiple approaches to detect hallucinations in LLM outputs:
1. Semantic Similarity: Using Sentence-BERT embeddings
2. Factual Consistency: Entity and claim verification
3. Contradiction Detection: Logical inconsistency identification

Author: Your Name
Date: 2024
License: MIT
"""

import re
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sentence_transformers import SentenceTransformer, util
import torch

logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Multi-method hallucination detector for LLM outputs.
    
    This class provides three primary detection methods:
    1. Semantic similarity comparison with reference text
    2. Factual consistency checking using entity extraction
    3. Internal contradiction detection
    
    Attributes:
        model: Sentence-BERT model for semantic embeddings
        threshold: Similarity threshold for hallucination detection
        device: Computing device (cuda or cpu)
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.75,
        device: Optional[str] = None
    ):
        """
        Initialize the hallucination detector.
        
        Args:
            model_name: Sentence-BERT model identifier
            threshold: Minimum cosine similarity score for non-hallucination
            device: Computing device ('cuda', 'cpu', or None for auto-detect)
        """
        self.threshold = threshold
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load sentence transformer model
        logger.info(f"Loading model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        logger.info("Hallucination detector initialized successfully")
    
    def detect(
        self,
        response: str,
        reference: str,
        method: str = "semantic"
    ) -> Dict:
        """
        Detect hallucinations in LLM response.
        
        This is the main detection method that supports multiple approaches:
        - semantic: Cosine similarity between response and reference embeddings
        - factual: Entity-based factual consistency checking
        - contradiction: Internal logical contradiction detection
        
        Args:
            response: LLM generated text to evaluate
            reference: Ground truth or source text
            method: Detection method ('semantic', 'factual', 'contradiction', 'all')
            
        Returns:
            Dictionary containing:
                - is_hallucination: Boolean detection result
                - similarity: Cosine similarity score (0-1)
                - confidence: Detection confidence (0-1)
                - errors: List of detected factual errors
                - method_used: Detection method applied
        """
        if method == "semantic" or method == "all":
            return self._detect_semantic(response, reference)
        elif method == "factual":
            return self._detect_factual(response, reference)
        elif method == "contradiction":
            return self._detect_contradiction(response)
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def _detect_semantic(self, response: str, reference: str) -> Dict:
        """
        Detect hallucinations using semantic similarity.
        
        Process:
        1. Generate embeddings for both texts
        2. Calculate cosine similarity
        3. Compare against threshold
        4. Analyze sentence-level similarities
        
        Args:
            response: LLM generated text
            reference: Ground truth text
            
        Returns:
            Detection results with similarity scores
        """
        # Calculate overall similarity
        similarity = self.calculate_similarity(response, reference)
        
        # Detect based on threshold
        is_hallucination = similarity < self.threshold
        
        # Calculate confidence based on distance from threshold
        confidence = abs(similarity - self.threshold) / self.threshold
        confidence = min(confidence, 1.0)
        
        # Sentence-level analysis for detailed error detection
        errors = []
        if is_hallucination:
            errors = self._find_hallucinated_sentences(response, reference)
        
        return {
            'is_hallucination': is_hallucination,
            'similarity': float(similarity),
            'confidence': float(confidence),
            'errors': errors,
            'method_used': 'semantic',
            'threshold': self.threshold
        }
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Uses caching to avoid recomputing embeddings for identical texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (0-1)
        """
        # Get embeddings (with caching)
        emb1 = self._get_embedding(text1)
        emb2 = self._get_embedding(text2)
        
        # Calculate cosine similarity
        similarity = util.cos_sim(emb1, emb2)
        
        return float(similarity.item())
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """
        Get embedding for text with caching.
        
        Args:
            text: Input text
            
        Returns:
            Embedding tensor
        """
        # Check cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Generate embedding
        embedding = self.model.encode(text, convert_to_tensor=True)
        
        # Cache for future use (limit cache size)
        if len(self.embedding_cache) < 1000:
            self.embedding_cache[text] = embedding
        
        return embedding
    
    def _find_hallucinated_sentences(
        self,
        response: str,
        reference: str
    ) -> List[Dict]:
        """
        Identify specific sentences that contain hallucinations.
        
        Splits response into sentences and checks each against reference.
        
        Args:
            response: LLM generated text
            reference: Ground truth text
            
        Returns:
            List of dictionaries with hallucinated sentence details
        """
        errors = []
        
        # Split into sentences
        response_sentences = self._split_sentences(response)
        
        # Get reference embedding once
        ref_embedding = self._get_embedding(reference)
        
        # Check each sentence
        for idx, sentence in enumerate(response_sentences):
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            # Calculate similarity
            sent_embedding = self._get_embedding(sentence)
            similarity = util.cos_sim(sent_embedding, ref_embedding).item()
            
            # If below threshold, mark as potential hallucination
            if similarity < self.threshold:
                errors.append({
                    'sentence_index': idx,
                    'sentence': sentence,
                    'similarity': float(similarity),
                    'severity': 'high' if similarity < 0.5 else 'medium'
                })
        
        return errors
    
    def _detect_factual(self, response: str, reference: str) -> Dict:
        """
        Detect factual inconsistencies using entity extraction.
        
        Process:
        1. Extract named entities from both texts
        2. Extract numerical facts
        3. Compare entity consistency
        4. Identify contradictions
        
        Args:
            response: LLM generated text
            reference: Ground truth text
            
        Returns:
            Detection results with factual error details
        """
        # Extract entities and facts
        response_entities = self._extract_entities(response)
        reference_entities = self._extract_entities(reference)
        
        # Check for inconsistencies
        errors = []
        hallucination_score = 0
        
        # Check numerical facts
        response_numbers = self._extract_numbers(response)
        reference_numbers = self._extract_numbers(reference)
        
        for num_fact in response_numbers:
            if not any(self._numbers_match(num_fact, ref_num) 
                      for ref_num in reference_numbers):
                errors.append({
                    'type': 'numerical',
                    'value': num_fact,
                    'context': 'Number not found in reference'
                })
                hallucination_score += 1
        
        # Check named entities
        for entity in response_entities:
            if entity not in reference_entities:
                # Check if semantically similar
                is_similar = any(
                    self.calculate_similarity(entity, ref_entity) > 0.8
                    for ref_entity in reference_entities
                )
                if not is_similar:
                    errors.append({
                        'type': 'entity',
                        'value': entity,
                        'context': 'Entity not verified in reference'
                    })
                    hallucination_score += 0.5
        
        # Calculate final score
        total_facts = len(response_numbers) + len(response_entities)
        if total_facts > 0:
            hallucination_rate = hallucination_score / total_facts
        else:
            hallucination_rate = 0
        
        is_hallucination = hallucination_rate > 0.3  # 30% threshold
        
        return {
            'is_hallucination': is_hallucination,
            'similarity': 1 - hallucination_rate,
            'confidence': 0.7,  # Factual method has moderate confidence
            'errors': errors,
            'method_used': 'factual',
            'factual_errors_count': len(errors)
        }
    
    def _detect_contradiction(self, response: str) -> Dict:
        """
        Detect internal contradictions within the response.
        
        Checks for logical inconsistencies by comparing different
        parts of the response against each other.
        
        Args:
            response: LLM generated text
            
        Returns:
            Detection results with contradiction details
        """
        sentences = self._split_sentences(response)
        contradictions = []
        
        # Compare each pair of sentences
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                # Check for explicit negation patterns
                if self._are_contradictory(sentences[i], sentences[j]):
                    contradictions.append({
                        'sentence1': sentences[i],
                        'sentence2': sentences[j],
                        'type': 'logical_contradiction'
                    })
        
        is_hallucination = len(contradictions) > 0
        
        return {
            'is_hallucination': is_hallucination,
            'similarity': 0.5 if is_hallucination else 1.0,
            'confidence': 0.8,
            'errors': contradictions,
            'method_used': 'contradiction',
            'contradiction_count': len(contradictions)
        }
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Simple sentence splitter
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text.
        
        Simple regex-based approach. For production, use spaCy or similar.
        """
        # Extract capitalized words (simple NER)
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(entities))
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numerical facts with context."""
        # Find numbers with surrounding context
        pattern = r'\b\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|percent|%))?\b'
        numbers = re.findall(pattern, text, re.IGNORECASE)
        return numbers
    
    def _numbers_match(self, num1: str, num2: str, tolerance: float = 0.05) -> bool:
        """Check if two numerical values match within tolerance."""
        try:
            # Extract pure numbers
            val1 = float(re.search(r'\d+(?:\.\d+)?', num1).group())
            val2 = float(re.search(r'\d+(?:\.\d+)?', num2).group())
            
            # Check within tolerance
            return abs(val1 - val2) / max(val1, val2) <= tolerance
        except:
            return num1.lower() == num2.lower()
    
    def _are_contradictory(self, sent1: str, sent2: str) -> bool:
        """
        Check if two sentences contradict each other.
        
        Simple heuristic based on negation patterns and antonyms.
        """
        # Check for explicit negation
        negation_words = ['not', 'never', 'no', "n't", 'neither', 'nor']
        
        sent1_lower = sent1.lower()
        sent2_lower = sent2.lower()
        
        # Check if one has negation and other doesn't (oversimplified)
        sent1_negated = any(word in sent1_lower for word in negation_words)
        sent2_negated = any(word in sent2_lower for word in negation_words)
        
        # If one is negated and they're semantically similar, might be contradiction
        if sent1_negated != sent2_negated:
            similarity = self.calculate_similarity(sent1, sent2)
            if similarity > 0.7:  # High similarity but opposite polarity
                return True
        
        return False
    
    def batch_detect(
        self,
        responses: List[str],
        references: List[str],
        method: str = "semantic"
    ) -> List[Dict]:
        """
        Batch processing for multiple response-reference pairs.
        
        Args:
            responses: List of LLM responses
            references: List of reference texts
            method: Detection method
            
        Returns:
            List of detection results
        """
        results = []
        for response, reference in zip(responses, references):
            result = self.detect(response, reference, method)
            results.append(result)
        return results
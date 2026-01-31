"""
Task classification for consensus-appropriate routing.

Based on evaluation findings (400 questions, 5 benchmarks, 4 models):

CONSENSUS HELPS:
- FACTUAL: HIGH consensus = 97-100% accuracy
- VERIFICATION: +6% hallucination detection improvement
- COMMONSENSE: HIGH consensus = 95% accuracy

CONSENSUS HURTS:
- REASONING/MATH: -35% accuracy (different reasoning chains should not be averaged)
- CREATIVE: Flattens diversity

This module provides recommendations for when to use multi-model consensus
based on task type, integrating with the router.TaskType classification.
"""

from enum import Enum
from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass
import re

# Import router TaskType if available (avoid circular imports)
if TYPE_CHECKING:
    from .router import TaskType as RouterTaskType


class ConsensusTaskCategory(str, Enum):
    """High-level task categories for consensus appropriateness."""
    FACTUAL = "factual"           # Facts, trivia, knowledge retrieval
    VERIFICATION = "verification"  # Claim checking, hallucination detection
    REASONING = "reasoning"        # Math, logic, multi-step problems
    CREATIVE = "creative"          # Writing, brainstorming, generation
    COMMONSENSE = "commonsense"    # Common knowledge, yes/no reasoning
    UNKNOWN = "unknown"            # Unable to classify


@dataclass
class ConsensusRecommendation:
    """Recommendation for whether to use multi-model consensus."""
    should_use_consensus: bool
    task_category: ConsensusTaskCategory
    confidence: float  # 0-1, how confident we are in the classification
    reason: str
    suggested_approach: str  # What to do instead if not consensus
    calibrated_confidence: Optional[float] = None  # If consensus used, expected accuracy


# Mapping from router.TaskType to ConsensusTaskCategory
# This allows integration with the existing routing system
ROUTER_TASK_TO_CONSENSUS_CATEGORY = {
    # These task types SHOULD use consensus
    "factual_qa": ConsensusTaskCategory.FACTUAL,
    "research": ConsensusTaskCategory.FACTUAL,
    "summarization": ConsensusTaskCategory.FACTUAL,
    "translation": ConsensusTaskCategory.FACTUAL,
    "chat": ConsensusTaskCategory.COMMONSENSE,
    "instruction_following": ConsensusTaskCategory.COMMONSENSE,
    
    # These task types should NOT use consensus
    "mathematical": ConsensusTaskCategory.REASONING,
    "logical_reasoning": ConsensusTaskCategory.REASONING,
    "scientific": ConsensusTaskCategory.REASONING,
    "analytical": ConsensusTaskCategory.REASONING,
    "code_generation": ConsensusTaskCategory.REASONING,  # Code has specific correct answers
    "code_review": ConsensusTaskCategory.REASONING,
    "code_debugging": ConsensusTaskCategory.REASONING,
    
    # Creative - don't use consensus
    "creative_writing": ConsensusTaskCategory.CREATIVE,
    "copywriting": ConsensusTaskCategory.CREATIVE,
    "brainstorming": ConsensusTaskCategory.CREATIVE,
    "roleplay": ConsensusTaskCategory.CREATIVE,
    
    # Specialized - depends, default to factual
    "legal": ConsensusTaskCategory.FACTUAL,
    "medical": ConsensusTaskCategory.FACTUAL,
    "financial": ConsensusTaskCategory.FACTUAL,
    "technical_docs": ConsensusTaskCategory.FACTUAL,
    
    # Unknown
    "unknown": ConsensusTaskCategory.UNKNOWN,
}


def router_task_to_consensus_category(router_task_type: str) -> ConsensusTaskCategory:
    """Convert a router.TaskType value to a ConsensusTaskCategory."""
    return ROUTER_TASK_TO_CONSENSUS_CATEGORY.get(
        router_task_type.lower(), 
        ConsensusTaskCategory.UNKNOWN
    )


# Patterns for heuristic classification
MATH_PATTERNS = [
    r'\b\d+\s*[\+\-\*\/\^]\s*\d+',  # arithmetic operations
    r'\bcalculate\b', r'\bcompute\b', r'\bsolve\b',
    r'\bequation\b', r'\bformula\b', r'\balgebra\b',
    r'\bderivative\b', r'\bintegral\b', r'\bprobability\b',
    r'\bhow\s+many\b.*\b(total|left|remaining)\b',
    r'\bif\s+.*\bthen\s+how\b',  # word problems
    r'\bstep\s*by\s*step\b',
    r'\bprove\b', r'\bproof\b',
    r'\bGSM8K\b',  # benchmark indicator
    r'\bdo\s+the\s+math\b', r'\bmath\s+problem\b',  # explicit math context
]

FACTUAL_PATTERNS = [
    r'\bwhat\s+is\s+(the|a)\b', r'\bwho\s+(is|was|are|were)\b',
    r'\bwhen\s+(did|was|were|is)\b', r'\bwhere\s+(is|was|are|were)\b',
    r'\bcapital\s+of\b', r'\bpopulation\s+of\b',
    r'\bfounded\b', r'\binvented\b', r'\bdiscovered\b',
    r'\bborn\b', r'\bdied\b', r'\belected\b',
    r'\bname\s+(the|a)\b', r'\blist\s+(the|all)\b',
    r'\btrivia\b', r'\bfact\b',
]

VERIFICATION_PATTERNS = [
    r'\bis\s+(it|this|that)\s+(true|false|correct|accurate)\b',
    r'\bverify\b', r'\bfact[\s-]?check\b',
    r'\bclaim\b', r'\ballegation\b',
    r'\bdid\s+.*\breally\b', r'\bactually\b',
    r'\btrue\s+or\s+false\b',
]

CREATIVE_PATTERNS = [
    r'\bwrite\s+(a|an|me)\b', r'\bcompose\b', r'\bcreate\b',
    r'\bstory\b', r'\bpoem\b', r'\bessay\b', r'\barticle\b',
    r'\bimagine\b', r'\bbrainstorm\b', r'\bgenerate\b',
    r'\bcome\s+up\s+with\b', r'\bsuggest\s+(some|a|an)\b',
    r'\bdesign\b', r'\binvent\b',
]

REASONING_INDICATORS = [
    r'\bstep\b.*\bstep\b',  # step-by-step
    r'\bexplain\s+(your|the)\s+(reasoning|logic|thought)\b',
    r'\bwhy\b.*\bbecause\b',
    r'\bif\b.*\bthen\b.*\bwhat\b',
    r'\blogic(al)?\b', r'\bdeduc\b', r'\binduc\b',
    r'\bchain\s+of\s+thought\b', r'\bCoT\b',
]


def _match_patterns(text: str, patterns: List[str]) -> int:
    """Count how many patterns match in the text."""
    text_lower = text.lower()
    matches = 0
    for pattern in patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            matches += 1
    return matches


def classify_task_heuristic(question: str) -> Tuple[ConsensusTaskCategory, float]:
    """
    Classify task type using heuristic pattern matching.
    
    Returns:
        Tuple of (ConsensusTaskCategory, confidence 0-1)
    """
    question_lower = question.lower()
    
    # Count matches for each category
    math_score = _match_patterns(question, MATH_PATTERNS)
    math_score += _match_patterns(question, REASONING_INDICATORS)
    factual_score = _match_patterns(question, FACTUAL_PATTERNS)
    verification_score = _match_patterns(question, VERIFICATION_PATTERNS)
    creative_score = _match_patterns(question, CREATIVE_PATTERNS)
    
    # Boost math score if numbers are heavily present
    number_count = len(re.findall(r'\b\d+\b', question))
    if number_count >= 3:
        math_score += 2
    
    # Boost verification score for strong indicators (these are unambiguous)
    if re.search(r'\b(true|false)\s+or\s+(false|true)\b', question_lower):
        verification_score += 2
    if re.search(r'\bis\s+(it|this|that)\s+(true|false)\b', question_lower):
        verification_score += 2  # Strong verification indicator
    
    scores = {
        ConsensusTaskCategory.REASONING: math_score,
        ConsensusTaskCategory.FACTUAL: factual_score,
        ConsensusTaskCategory.VERIFICATION: verification_score,
        ConsensusTaskCategory.CREATIVE: creative_score,
    }
    
    max_score = max(scores.values())
    
    if max_score == 0:
        # No clear indicators - check for yes/no question patterns
        if re.search(r'^(is|are|was|were|do|does|did|can|could|should|would|will)\b', question_lower):
            return ConsensusTaskCategory.COMMONSENSE, 0.5
        return ConsensusTaskCategory.UNKNOWN, 0.3
    
    # Find winner
    winner = max(scores, key=scores.get)
    
    # Calculate confidence based on margin
    total_score = sum(scores.values())
    confidence = min(0.9, 0.5 + (max_score / (total_score + 1)) * 0.4)
    
    return winner, confidence


def recommend_consensus(
    question: str,
    task_category: Optional[ConsensusTaskCategory] = None,
    router_task_type: Optional[str] = None,
    force: Optional[bool] = None
) -> ConsensusRecommendation:
    """
    Recommend whether to use multi-model consensus for a question.
    
    Based on evaluation findings (400 questions, 4 models):
    - FACTUAL: Use consensus (HIGH consensus = 97-100% accuracy)
    - VERIFICATION: Use consensus (+6% hallucination detection)
    - REASONING: DON'T use consensus (-35% on math!)
    - CREATIVE: DON'T use consensus (flattens diversity)
    - COMMONSENSE: Use consensus (HIGH consensus = 95% accuracy)
    
    Args:
        question: The question to analyze
        task_category: Override automatic classification with ConsensusTaskCategory
        router_task_type: router.TaskType value (will be converted)
        force: Force consensus on (True) or off (False)
        
    Returns:
        ConsensusRecommendation with guidance
    """
    if force is not None:
        return ConsensusRecommendation(
            should_use_consensus=force,
            task_category=task_category or ConsensusTaskCategory.UNKNOWN,
            confidence=1.0,
            reason="Forced by user",
            suggested_approach="consensus" if force else "single_model",
            calibrated_confidence=0.95 if force else None
        )
    
    # Determine task category
    if task_category is not None:
        confidence = 1.0  # User provided classification
    elif router_task_type is not None:
        task_category = router_task_to_consensus_category(router_task_type)
        confidence = 0.9  # From router classification
    else:
        task_category, confidence = classify_task_heuristic(question)
    
    # Recommendations based on evaluation findings
    recommendations = {
        ConsensusTaskCategory.FACTUAL: ConsensusRecommendation(
            should_use_consensus=True,
            task_category=ConsensusTaskCategory.FACTUAL,
            confidence=confidence,
            reason="Factual questions benefit from consensus. HIGH consensus = 97-100% accuracy.",
            suggested_approach="Use 3-4 diverse models, trust HIGH consensus answers.",
            calibrated_confidence=0.97
        ),
        ConsensusTaskCategory.VERIFICATION: ConsensusRecommendation(
            should_use_consensus=True,
            task_category=ConsensusTaskCategory.VERIFICATION,
            confidence=confidence,
            reason="Consensus improves hallucination detection by 6%. LOW/NONE consensus flags confabulation.",
            suggested_approach="Use consensus; LOW consensus indicates potential hallucination.",
            calibrated_confidence=0.95
        ),
        ConsensusTaskCategory.REASONING: ConsensusRecommendation(
            should_use_consensus=False,
            task_category=ConsensusTaskCategory.REASONING,
            confidence=confidence,
            reason="WARNING: Consensus DEGRADES math/reasoning by 35%. Different reasoning chains should not be averaged.",
            suggested_approach="Use single best model with chain-of-thought, or self-consistency within ONE model.",
            calibrated_confidence=None
        ),
        ConsensusTaskCategory.CREATIVE: ConsensusRecommendation(
            should_use_consensus=False,
            task_category=ConsensusTaskCategory.CREATIVE,
            confidence=confidence,
            reason="Consensus flattens creative diversity. Use single model for unique outputs.",
            suggested_approach="Use single model; consider sampling multiple outputs from ONE model if variety needed.",
            calibrated_confidence=None
        ),
        ConsensusTaskCategory.COMMONSENSE: ConsensusRecommendation(
            should_use_consensus=True,
            task_category=ConsensusTaskCategory.COMMONSENSE,
            confidence=confidence,
            reason="Commonsense questions benefit from consensus. HIGH consensus = 95% accuracy.",
            suggested_approach="Use consensus for yes/no and common knowledge questions.",
            calibrated_confidence=0.95
        ),
        ConsensusTaskCategory.UNKNOWN: ConsensusRecommendation(
            should_use_consensus=True,  # Default to consensus for unknown
            task_category=ConsensusTaskCategory.UNKNOWN,
            confidence=confidence,
            reason="Task type unclear. Defaulting to consensus with caution.",
            suggested_approach="Use consensus but check if answer involves multi-step reasoning.",
            calibrated_confidence=0.85
        ),
    }
    
    return recommendations.get(task_category, recommendations[ConsensusTaskCategory.UNKNOWN])


def get_confidence_calibration(consensus_level: str) -> float:
    """
    Convert consensus level to calibrated confidence score.
    
    Based on evaluation data:
    - HIGH consensus: 95-100% accuracy → 0.95 confidence
    - MEDIUM consensus: ~75% accuracy → 0.75 confidence  
    - LOW consensus: ~50-70% accuracy → 0.60 confidence
    - NONE consensus: ~50% accuracy → 0.50 confidence
    
    Args:
        consensus_level: One of "high", "medium", "low", "none", "contradictory"
        
    Returns:
        Calibrated confidence score 0-1
    """
    calibration = {
        "high": 0.95,
        "medium": 0.75,
        "low": 0.60,
        "none": 0.50,
        "contradictory": 0.40,
    }
    return calibration.get(consensus_level.lower(), 0.50)


# Convenience functions

def is_consensus_appropriate(question: str) -> bool:
    """Quick check if consensus is recommended for this question."""
    return recommend_consensus(question).should_use_consensus


def get_task_category(question: str) -> ConsensusTaskCategory:
    """Get the classified task category for a question."""
    task_category, _ = classify_task_heuristic(question)
    return task_category


def should_warn_about_consensus(question: str) -> Optional[str]:
    """
    Return a warning message if consensus is inappropriate for this question.
    Returns None if consensus is appropriate.
    """
    rec = recommend_consensus(question)
    if not rec.should_use_consensus:
        return rec.reason
    return None

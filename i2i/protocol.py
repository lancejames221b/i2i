"""
Main AICP Protocol implementation.

This is the primary interface for the AI-to-AI Communication Protocol,
providing a unified API for consensus queries, cross-verification,
epistemic classification, and multi-model orchestration.
"""

import asyncio
from typing import List, Optional, Dict, Any

from .schema import (
    Message,
    MessageType,
    Response,
    ConsensusResult,
    VerificationResult,
    EpistemicClassification,
    EpistemicType,
    ConsensusLevel,
)
from .providers import ProviderRegistry
from .consensus import ConsensusEngine
from .verification import VerificationEngine
from .epistemic import EpistemicClassifier
from .router import ModelRouter, TaskType, RoutingStrategy, RoutingResult
from .config import get_consensus_models


class AICP:
    """
    AI-to-AI Communication Protocol.

    This class provides the main interface for multi-model AI interactions,
    including consensus queries, cross-verification, and epistemic classification.

    Example:
        protocol = AICP()

        # Consensus query
        result = await protocol.consensus_query(
            "What causes inflation?",
            models=["gpt-4", "claude-3", "gemini-pro"]
        )

        # Cross-verification
        verified = await protocol.verify_claim(
            "The Earth is approximately 4.5 billion years old",
            verifiers=["gpt-4", "claude-3"]
        )

        # Epistemic classification
        classification = await protocol.classify_question(
            "Is consciousness substrate-independent?"
        )
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AICP protocol.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.registry = ProviderRegistry()
        self.consensus_engine = ConsensusEngine(self.registry)
        self.verification_engine = VerificationEngine(self.registry)
        self.epistemic_classifier = EpistemicClassifier(self.registry)
        self.router = ModelRouter(self.registry)

    # ==================== Provider Management ====================

    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models by provider."""
        return self.registry.list_available_models()

    def list_configured_providers(self) -> List[str]:
        """List providers that are properly configured."""
        return self.registry.list_configured_providers()

    # ==================== Direct Queries ====================

    async def query(
        self,
        prompt: str,
        model: str,
        context: Optional[List[Message]] = None,
    ) -> Response:
        """
        Query a single model.

        Args:
            prompt: The prompt/question
            model: Model identifier (e.g., "gpt-4", "claude-3")
            context: Optional conversation context

        Returns:
            Response from the model
        """
        message = Message(
            type=MessageType.QUERY,
            content=prompt,
            context=context,
        )
        return await self.registry.query(message, model)

    async def query_multiple(
        self,
        prompt: str,
        models: List[str],
        context: Optional[List[Message]] = None,
    ) -> List[Response]:
        """
        Query multiple models in parallel.

        Args:
            prompt: The prompt/question
            models: List of model identifiers
            context: Optional conversation context

        Returns:
            List of responses (may include errors)
        """
        message = Message(
            type=MessageType.QUERY,
            content=prompt,
            context=context,
        )
        return await self.registry.query_multiple(message, models)

    # ==================== Consensus Queries ====================

    async def consensus_query(
        self,
        query: str,
        models: Optional[List[str]] = None,
        context: Optional[List[Message]] = None,
    ) -> ConsensusResult:
        """
        Query multiple models and analyze their consensus.

        This is the primary method for getting reliable answers by
        cross-referencing multiple AI architectures.

        Args:
            query: The question/prompt
            models: Models to query (uses defaults if None)
            context: Optional conversation context

        Returns:
            ConsensusResult with analysis of agreement/disagreement
        """
        if models is None:
            # Default to a diverse set of models
            models = self._get_default_models()

        return await self.consensus_engine.query_for_consensus(
            query=query,
            models=models,
            context=context,
        )

    def _get_default_models(self) -> List[str]:
        """Get default models for consensus queries."""
        # Get from config (can be overridden via env vars or programmatically)
        preferred = get_consensus_models() + [
            "llama-3.3-70b-versatile",
            "mistral-large-latest",
        ]

        available = []
        for model in preferred:
            adapter = self.registry.get_adapter(model)
            if adapter and adapter.is_configured():
                available.append(model)

        # Need at least 2 for consensus
        if len(available) < 2:
            raise ValueError(
                "Need at least 2 configured providers for consensus. "
                f"Available: {self.list_configured_providers()}"
            )

        return available[:4]  # Limit to 4 for cost

    # ==================== Cross-Verification ====================

    async def verify_claim(
        self,
        claim: str,
        verifiers: Optional[List[str]] = None,
        original_source: Optional[str] = None,
        context: Optional[str] = None,
    ) -> VerificationResult:
        """
        Have multiple AI models verify a claim.

        Args:
            claim: The claim/statement to verify
            verifiers: Models to use for verification
            original_source: Model that originally made the claim
            context: Additional context

        Returns:
            VerificationResult with analysis
        """
        if verifiers is None:
            verifiers = self._get_default_models()[:2]

        return await self.verification_engine.verify_claim(
            claim=claim,
            verifier_models=verifiers,
            original_source=original_source,
            context=context,
        )

    async def verify_claim_grounded(
        self,
        claim: str,
        verifiers: Optional[List[str]] = None,
        search_backend: Optional[str] = None,
        num_sources: int = 5,
        original_source: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a claim with search-grounded evidence (RAG verification).

        This method retrieves relevant sources from the web before
        asking verification models to evaluate the claim. The result
        includes source citations for transparency.

        Args:
            claim: The claim/statement to verify
            verifiers: Models to use for verification
            search_backend: Search backend to use (brave, serpapi, tavily)
            num_sources: Number of sources to retrieve (default: 5)
            original_source: Model that originally made the claim

        Returns:
            VerificationResult with source_citations and retrieved_sources
        """
        if verifiers is None:
            verifiers = self._get_default_models()[:2]

        return await self.verification_engine.verify_claim_with_search(
            claim=claim,
            verifier_models=verifiers,
            search_backend=search_backend,
            num_sources=num_sources,
            original_source=original_source,
        )

    async def challenge_response(
        self,
        response: Response,
        challengers: Optional[List[str]] = None,
        challenge_type: str = "general",
    ) -> Dict[str, Any]:
        """
        Have AI models challenge another AI's response.

        Args:
            response: The response to challenge
            challengers: Models to use as challengers
            challenge_type: Type of challenge (general, factual, logical, ethical)

        Returns:
            Dictionary with challenges and analysis
        """
        if challengers is None:
            challengers = self._get_default_models()[:2]

        return await self.verification_engine.challenge_response(
            original_response=response,
            challenger_models=challengers,
            challenge_type=challenge_type,
        )

    # ==================== Epistemic Classification ====================

    async def classify_question(
        self,
        question: str,
        classifiers: Optional[List[str]] = None,
    ) -> EpistemicClassification:
        """
        Classify a question's epistemic status.

        Determines if a question is:
        - ANSWERABLE: Can be resolved with available information
        - UNCERTAIN: Answerable but with significant uncertainty
        - UNDERDETERMINED: Multiple hypotheses fit equally
        - IDLE: Well-formed but non-action-guiding
        - MALFORMED: Incoherent or self-contradictory

        Args:
            question: The question to classify
            classifiers: Models to use for classification

        Returns:
            EpistemicClassification with analysis
        """
        return await self.epistemic_classifier.classify_question(
            question=question,
            classifier_models=classifiers,
        )

    def quick_classify(self, question: str) -> EpistemicType:
        """
        Quick heuristic classification without API calls.

        Useful for pre-filtering before expensive classification.
        """
        return self.epistemic_classifier.quick_classify(question)

    # ==================== Intelligent Routing ====================

    async def routed_query(
        self,
        query: str,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        force_task: Optional[TaskType] = None,
    ) -> RoutingResult:
        """
        Automatically route a query to the optimal model.

        The router analyzes the query to determine task type (code, math,
        creative writing, etc.) and selects the best model based on the
        chosen strategy.

        Args:
            query: The question/prompt
            strategy: Routing strategy to use
                - BEST_QUALITY: Optimize for output quality
                - BEST_SPEED: Optimize for latency
                - BEST_VALUE: Optimize for cost-effectiveness
                - BALANCED: Balance all factors
                - ENSEMBLE: Use multiple models and synthesize
            force_task: Override automatic task detection

        Returns:
            RoutingResult with the routing decision and response
        """
        return await self.router.route_and_execute(
            query=query,
            strategy=strategy,
            force_task=force_task,
        )

    async def ensemble_query(
        self,
        query: str,
        num_models: int = 3,
    ) -> RoutingResult:
        """
        Query multiple models and synthesize their responses.

        This is a convenience method that uses the ENSEMBLE strategy
        to get diverse perspectives and combine them.

        Args:
            query: The question/prompt
            num_models: Number of models to query (default 3)

        Returns:
            RoutingResult with synthesized response
        """
        return await self.router.route_and_execute(
            query=query,
            strategy=RoutingStrategy.ENSEMBLE,
            num_models=num_models,
        )

    def get_model_recommendation(
        self,
        task_type: TaskType,
    ) -> Dict[str, Any]:
        """
        Get model recommendations for a specific task type.

        Returns recommendations for different strategies (quality, speed, value).

        Args:
            task_type: The type of task (e.g., CODE_GENERATION, CREATIVE_WRITING)

        Returns:
            Dictionary with model recommendations per strategy
        """
        return self.router.get_model_recommendation(task_type)

    def classify_task(self, query: str) -> tuple:
        """
        Classify what type of task a query represents.

        Args:
            query: The input query

        Returns:
            Tuple of (TaskType, confidence)
        """
        return self.router.classifier.classify(query)

    # ==================== High-Level Workflows ====================

    async def smart_query(
        self,
        query: str,
        require_consensus: bool = True,
        verify_result: bool = False,
    ) -> Dict[str, Any]:
        """
        Intelligent query that adapts based on question type.

        This method:
        1. Classifies the question's epistemic status
        2. If answerable, queries appropriate models
        3. If consensus is required, ensures agreement
        4. Optionally verifies the result

        Args:
            query: The question/prompt
            require_consensus: Whether to require multi-model consensus
            verify_result: Whether to verify the final answer

        Returns:
            Dictionary with answer and metadata
        """
        result = {
            "query": query,
            "classification": None,
            "answer": None,
            "consensus": None,
            "verification": None,
            "warnings": [],
        }

        # Step 1: Classify the question
        classification = await self.classify_question(query)
        result["classification"] = classification

        # Step 2: Handle based on classification
        if classification.classification == EpistemicType.IDLE:
            result["warnings"].append(
                "This question is classified as 'idle' - well-formed but non-action-guiding. "
                "The answer may not help you make any decisions."
            )

        if classification.classification == EpistemicType.MALFORMED:
            result["warnings"].append(
                "This question appears to be malformed or self-contradictory."
            )
            result["answer"] = "The question cannot be meaningfully answered as posed."
            return result

        if classification.classification == EpistemicType.UNDERDETERMINED:
            result["warnings"].append(
                "This question is underdetermined - multiple hypotheses fit the evidence equally."
            )

        # Step 3: Query for consensus
        if require_consensus:
            consensus = await self.consensus_query(query)
            result["consensus"] = {
                "level": consensus.consensus_level.value,
                "answer": consensus.consensus_answer,
                "models_queried": consensus.models_queried,
                "divergences": len(consensus.divergences),
            }

            if consensus.consensus_level in [ConsensusLevel.HIGH, ConsensusLevel.MEDIUM]:
                result["answer"] = consensus.consensus_answer
            else:
                result["warnings"].append(
                    f"Models did not reach consensus (level: {consensus.consensus_level.value})"
                )
                # Return all responses
                result["answer"] = "No consensus reached. See individual responses."
                result["individual_responses"] = [
                    {"model": r.model, "content": r.content[:500]}
                    for r in consensus.responses
                ]
        else:
            # Single model query
            response = await self.query(query, self._get_default_models()[0])
            result["answer"] = response.content

        # Step 4: Verify if requested
        if verify_result and result["answer"]:
            verification = await self.verify_claim(result["answer"])
            result["verification"] = {
                "verified": verification.verified,
                "confidence": verification.confidence,
                "issues": verification.issues_found,
            }

            if not verification.verified:
                result["warnings"].append(
                    f"Answer verification failed. Issues: {verification.issues_found}"
                )

        return result

    async def debate(
        self,
        topic: str,
        models: Optional[List[str]] = None,
        rounds: int = 2,
    ) -> Dict[str, Any]:
        """
        Have multiple AI models debate a topic.

        Each model presents its position, then responds to others.

        Args:
            topic: The debate topic/question
            models: Models to participate
            rounds: Number of debate rounds

        Returns:
            Dictionary with debate transcript and analysis
        """
        if models is None:
            models = self._get_default_models()[:3]

        debate = {
            "topic": topic,
            "participants": models,
            "rounds": [],
            "summary": None,
        }

        # Initial positions
        initial_prompt = f"""We are having a structured debate on the following topic:

TOPIC: {topic}

Please present your initial position on this topic. Be clear, substantive, and
acknowledge any uncertainties. Structure your response as:

1. POSITION: Your main stance
2. KEY ARGUMENTS: Your strongest arguments
3. CAVEATS: Any limitations or uncertainties in your position"""

        initial_responses = await self.query_multiple(initial_prompt, models)
        debate["rounds"].append({
            "round": 0,
            "type": "initial_positions",
            "responses": [
                {"model": r.model, "content": r.content}
                for r in initial_responses
                if isinstance(r, Response)
            ]
        })

        # Debate rounds
        for round_num in range(1, rounds + 1):
            round_responses = []

            for i, model in enumerate(models):
                # Build context from other models' previous responses
                other_responses = [
                    r for r in debate["rounds"][-1]["responses"]
                    if r["model"] != f"*/{model}" and model not in r["model"]
                ]

                response_prompt = f"""DEBATE ROUND {round_num}

TOPIC: {topic}

Other participants' positions:
{chr(10).join([f'{r["model"]}: {r["content"][:500]}...' for r in other_responses])}

Please respond to the other participants' arguments:
1. ADDRESS: Which arguments do you agree/disagree with and why?
2. REBUTTALS: Counter any points you find weak
3. UPDATES: Have you updated your position based on their arguments?
4. SYNTHESIS: Any points of agreement emerging?"""

                try:
                    response = await self.query(response_prompt, model)
                    round_responses.append({
                        "model": response.model,
                        "content": response.content
                    })
                except Exception as e:
                    round_responses.append({
                        "model": model,
                        "content": f"Error: {str(e)}"
                    })

            debate["rounds"].append({
                "round": round_num,
                "type": "response",
                "responses": round_responses
            })

        # Generate summary
        summary_prompt = f"""Please summarize this debate:

TOPIC: {topic}

DEBATE TRANSCRIPT:
{self._format_debate_transcript(debate)}

Provide:
1. AREAS OF AGREEMENT: Where did participants converge?
2. PERSISTENT DISAGREEMENTS: Where do fundamental differences remain?
3. STRONGEST ARGUMENTS: Which arguments were most compelling?
4. CONCLUSION: What can we conclude from this debate?"""

        summary_response = await self.query(summary_prompt, models[0])
        debate["summary"] = summary_response.content

        return debate

    def _format_debate_transcript(self, debate: Dict) -> str:
        """Format debate rounds into a readable transcript."""
        lines = []
        for round_data in debate["rounds"]:
            lines.append(f"\n=== Round {round_data['round']} ({round_data['type']}) ===\n")
            for resp in round_data["responses"]:
                lines.append(f"\n[{resp['model']}]:\n{resp['content'][:1000]}\n")
        return "\n".join(lines)

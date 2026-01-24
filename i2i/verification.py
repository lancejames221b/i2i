"""
Cross-verification system for AI-to-AI fact-checking.

This module enables AI models to verify, challenge, and critique
each other's responses.
"""

import asyncio
from typing import List, Optional, Dict, Any

from .schema import (
    Message,
    MessageType,
    Response,
    VerificationResult,
    ConfidenceLevel,
)
from .providers import ProviderRegistry


class VerificationEngine:
    """
    Engine for cross-verification of claims between AI models.
    """

    def __init__(self, registry: ProviderRegistry):
        self.registry = registry

    async def verify_claim(
        self,
        claim: str,
        verifier_models: List[str],
        original_source: Optional[str] = None,
        context: Optional[str] = None,
    ) -> VerificationResult:
        """
        Have multiple AI models verify a claim.

        Args:
            claim: The claim/statement to verify
            verifier_models: Models to use for verification
            original_source: Model that originally made the claim (if any)
            context: Additional context about the claim

        Returns:
            VerificationResult with analysis
        """
        # Build verification prompt
        verification_prompt = self._build_verification_prompt(claim, original_source, context)

        message = Message(
            type=MessageType.VERIFY,
            content=verification_prompt,
        )

        # Query all verifiers in parallel
        responses = await self.registry.query_multiple(message, verifier_models)

        # Filter valid responses
        valid_responses = [r for r in responses if isinstance(r, Response)]

        if not valid_responses:
            raise ValueError("All verification queries failed")

        # Analyze verification results
        verified, confidence, issues, corrections = self._analyze_verification(
            claim, valid_responses
        )

        return VerificationResult(
            original_claim=claim,
            original_source=original_source,
            verifiers=[r.model for r in valid_responses],
            verification_responses=valid_responses,
            verified=verified,
            confidence=confidence,
            issues_found=issues,
            corrections=corrections,
        )

    async def verify_claim_with_search(
        self,
        claim: str,
        verifier_models: List[str],
        search_backend: Optional[str] = None,
        num_sources: int = 5,
        original_source: Optional[str] = None,
    ) -> VerificationResult:
        """
        Verify a claim with search-grounded evidence (RAG verification).

        This method retrieves relevant sources from the web before
        asking verification models to evaluate the claim against those sources.

        Args:
            claim: The claim/statement to verify
            verifier_models: Models to use for verification
            search_backend: Specific search backend to use (optional)
            num_sources: Number of sources to retrieve
            original_source: Model that originally made the claim (if any)

        Returns:
            VerificationResult with source citations
        """
        from .search import SearchRegistry

        # 1. Search for relevant sources
        search_registry = SearchRegistry()
        search_results = await search_registry.search(
            claim, backend=search_backend, num_results=num_sources
        )

        # 2. Build grounded verification prompt
        if search_results:
            context = "\n".join([
                f"- {r.title}: {r.snippet} ({r.url})"
                for r in search_results
            ])
            grounded_prompt = f"""Verify this claim using the provided sources:

CLAIM: "{claim}"

RETRIEVED SOURCES:
{context}

Evaluate if the claim is SUPPORTED, CONTRADICTED, or NOT ADDRESSED by these sources.
Cite specific sources in your reasoning by referencing their URLs.

Please respond with:
1. VERDICT: TRUE (supported by sources), FALSE (contradicted), PARTIALLY TRUE, or UNVERIFIABLE
2. CONFIDENCE: How confident are you (HIGH, MEDIUM, LOW)?
3. ISSUES: List any factual errors or problems with the claim
4. CORRECTIONS: If false or partially true, provide correct information from sources
5. REASONING: Explain your verification citing specific sources"""
        else:
            # Fallback to regular verification if no sources found
            grounded_prompt = self._build_verification_prompt(claim, original_source, None)

        message = Message(
            type=MessageType.VERIFY,
            content=grounded_prompt,
        )

        # 3. Query verifiers
        responses = await self.registry.query_multiple(message, verifier_models)
        valid_responses = [r for r in responses if isinstance(r, Response)]

        if not valid_responses:
            raise ValueError("All verification queries failed")

        # 4. Analyze verification results
        verified, confidence, issues, corrections = self._analyze_verification(
            claim, valid_responses
        )

        # 5. Build result with source tracking
        return VerificationResult(
            original_claim=claim,
            original_source=original_source,
            verifiers=[r.model for r in valid_responses],
            verification_responses=valid_responses,
            verified=verified,
            confidence=confidence,
            issues_found=issues,
            corrections=corrections,
            source_citations=[r.url for r in search_results],
            retrieved_sources=[
                {"title": r.title, "url": r.url, "snippet": r.snippet}
                for r in search_results
            ],
        )

    def _build_verification_prompt(
        self,
        claim: str,
        original_source: Optional[str],
        context: Optional[str],
    ) -> str:
        """Build the verification prompt."""
        prompt = f"""Please verify the following claim for accuracy:

CLAIM: "{claim}"
"""
        if original_source:
            prompt += f"\nSource: {original_source}"
        if context:
            prompt += f"\nContext: {context}"

        prompt += """

Please analyze this claim and respond with:
1. VERDICT: Is this claim TRUE, FALSE, PARTIALLY TRUE, or UNVERIFIABLE?
2. CONFIDENCE: How confident are you (HIGH, MEDIUM, LOW)?
3. ISSUES: List any factual errors, misleading statements, or problematic aspects
4. CORRECTIONS: If the claim is false or partially true, provide the correct information
5. REASONING: Explain your verification process

Format your response clearly with these sections."""

        return prompt

    def _analyze_verification(
        self,
        claim: str,
        responses: List[Response],
    ) -> tuple[bool, float, List[str], Optional[str]]:
        """
        Analyze verification responses to determine overall verdict.

        Returns:
            (verified, confidence, issues_found, corrections)
        """
        verdicts = []
        all_issues = []
        all_corrections = []

        for resp in responses:
            content = resp.content.lower()

            # Extract verdict
            if "verdict: true" in content or "verdict:true" in content:
                verdicts.append(1.0)
            elif "verdict: false" in content or "verdict:false" in content:
                verdicts.append(0.0)
            elif "partially true" in content:
                verdicts.append(0.5)
            elif "unverifiable" in content:
                verdicts.append(0.5)  # Uncertain
            else:
                # Heuristic based on content
                if "correct" in content and "incorrect" not in content:
                    verdicts.append(0.8)
                elif "incorrect" in content or "false" in content or "wrong" in content:
                    verdicts.append(0.2)
                else:
                    verdicts.append(0.5)

            # Extract issues (look for numbered lists or "issues:" section)
            if "issues:" in content:
                issues_section = content.split("issues:")[1].split("corrections:")[0]
                issues = [line.strip() for line in issues_section.split("\n") if line.strip() and line.strip()[0].isdigit()]
                all_issues.extend(issues)

            # Extract corrections
            if "corrections:" in content:
                corrections_section = content.split("corrections:")[1].split("reasoning:")[0]
                if corrections_section.strip():
                    all_corrections.append(corrections_section.strip())

        # Calculate overall verdict
        if not verdicts:
            return False, 0.0, [], None

        avg_verdict = sum(verdicts) / len(verdicts)
        verified = avg_verdict >= 0.6

        # Calculate confidence based on agreement
        verdict_variance = sum((v - avg_verdict) ** 2 for v in verdicts) / len(verdicts)
        confidence = max(0.0, min(1.0, 1.0 - verdict_variance * 2))

        # Deduplicate issues
        unique_issues = list(set(all_issues))

        # Combine corrections
        corrections = "\n".join(all_corrections) if all_corrections else None

        return verified, confidence, unique_issues, corrections

    async def challenge_response(
        self,
        original_response: Response,
        challenger_models: List[str],
        challenge_type: str = "general",
    ) -> Dict[str, Any]:
        """
        Have AI models challenge another AI's response.

        Args:
            original_response: The response to challenge
            challenger_models: Models to use as challengers
            challenge_type: Type of challenge (general, factual, logical, ethical)

        Returns:
            Dictionary with challenges and analysis
        """
        challenge_prompt = self._build_challenge_prompt(
            original_response.content,
            original_response.model,
            challenge_type,
        )

        message = Message(
            type=MessageType.CHALLENGE,
            content=challenge_prompt,
        )

        responses = await self.registry.query_multiple(message, challenger_models)
        valid_responses = [r for r in responses if isinstance(r, Response)]

        # Analyze challenges
        challenges = []
        for resp in valid_responses:
            challenge = self._parse_challenge(resp)
            challenges.append({
                "challenger": resp.model,
                "challenge": challenge,
                "full_response": resp.content,
            })

        # Determine if original response withstands challenges
        withstands = self._evaluate_challenges(challenges)

        return {
            "original_model": original_response.model,
            "original_response": original_response.content[:500],
            "challengers": [r.model for r in valid_responses],
            "challenges": challenges,
            "withstands_challenges": withstands,
            "challenge_summary": self._summarize_challenges(challenges),
        }

    def _build_challenge_prompt(
        self,
        response_content: str,
        source_model: str,
        challenge_type: str,
    ) -> str:
        """Build a challenge prompt."""
        type_instructions = {
            "general": "Look for any weaknesses, errors, or questionable claims.",
            "factual": "Focus on factual accuracy. Identify any incorrect facts or unsupported claims.",
            "logical": "Focus on logical consistency. Identify any logical fallacies, contradictions, or non-sequiturs.",
            "ethical": "Focus on ethical implications. Identify any harmful, biased, or problematic content.",
        }

        instruction = type_instructions.get(challenge_type, type_instructions["general"])

        return f"""Please critically analyze the following AI response:

SOURCE: {source_model}
RESPONSE:
{response_content}

YOUR TASK: {instruction}

Please provide:
1. VALIDITY: Is the response fundamentally sound? (YES, PARTIALLY, NO)
2. WEAKNESSES: List specific weaknesses or errors (be precise)
3. COUNTERARGUMENTS: Provide counterarguments or alternative perspectives
4. IMPROVEMENT: How could this response be improved?
5. OVERALL ASSESSMENT: Brief summary of your challenge

Be rigorous but fair. Only flag genuine issues, not stylistic preferences."""

    def _parse_challenge(self, response: Response) -> Dict[str, Any]:
        """Parse a challenge response into structured format."""
        content = response.content

        # Extract sections (simple parsing)
        challenge = {
            "validity": "unknown",
            "weaknesses": [],
            "counterarguments": [],
            "improvements": [],
            "assessment": "",
        }

        content_lower = content.lower()

        # Extract validity
        if "validity: yes" in content_lower or "validity:yes" in content_lower:
            challenge["validity"] = "yes"
        elif "validity: no" in content_lower or "validity:no" in content_lower:
            challenge["validity"] = "no"
        elif "validity: partially" in content_lower:
            challenge["validity"] = "partially"

        # Extract assessment (last paragraph or after "overall assessment")
        if "overall assessment:" in content_lower:
            assessment_start = content_lower.index("overall assessment:")
            challenge["assessment"] = content[assessment_start + 20:].strip()[:500]
        else:
            # Use last paragraph
            paragraphs = content.strip().split("\n\n")
            if paragraphs:
                challenge["assessment"] = paragraphs[-1][:500]

        return challenge

    def _evaluate_challenges(self, challenges: List[Dict]) -> bool:
        """Determine if the original response withstands challenges."""
        if not challenges:
            return True

        validity_scores = []
        for challenge in challenges:
            parsed = challenge.get("challenge", {})
            validity = parsed.get("validity", "unknown")

            if validity == "yes":
                validity_scores.append(1.0)
            elif validity == "partially":
                validity_scores.append(0.5)
            elif validity == "no":
                validity_scores.append(0.0)
            else:
                validity_scores.append(0.5)

        avg_validity = sum(validity_scores) / len(validity_scores)
        return avg_validity >= 0.5

    def _summarize_challenges(self, challenges: List[Dict]) -> str:
        """Create a summary of all challenges."""
        if not challenges:
            return "No challenges raised."

        assessments = [
            c.get("challenge", {}).get("assessment", "")
            for c in challenges
            if c.get("challenge", {}).get("assessment")
        ]

        if not assessments:
            return "Challenges raised but no clear assessments."

        # Combine assessments
        return " | ".join(assessments[:3])  # Limit to first 3

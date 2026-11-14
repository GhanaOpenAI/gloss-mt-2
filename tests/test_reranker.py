"""Unit tests for Cross-Encoder Reranker data and scoring logic."""

import pytest
from gloss_mt_2.inference import substitute_glossary
from gloss_mt_2.lookup import TwiLookup


class MockReranker:
    def rerank(self, twi_context: str, noun_phrase: str, candidates: list) -> str:
        # Evaluates Twi context: if Twi sentence contains "ɔtemmufoɔ" (judge) or "asɛm" (case), pick "asɛnnibea"
        if "ɔtemmufoɔ" in twi_context or "asɛm" in twi_context:
            if "asɛnnibea" in candidates:
                return "asɛnnibea"
        return candidates[0]


def test_substitute_with_reranker_in_twi_context():
    lookup = TwiLookup(custom_dict={"court": "asɛnnibea"})
    lookup.candidates = {"court": ["kɔɔto", "asɛnnibea", "ahemfie"]}

    # Model generates Twi with the English noun in <...>:
    twi_generated = "Ɔtemmufoɔ no wuraa <court> no mu sɛ ɔrebɛbue asɛm no."

    mock_reranker = MockReranker()

    final, applied = substitute_glossary(
        translation=twi_generated,
        lookup=lookup,
        direction="en-tw",
        reranker=mock_reranker,
        source_context=twi_generated,
    )

    assert "asɛnnibea no" in final
    assert applied[0]["target"] == "asɛnnibea"

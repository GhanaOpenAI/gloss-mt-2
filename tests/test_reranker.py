"""Unit tests for Cross-Encoder Reranker data and scoring logic."""

import pytest
from gloss_mt_2.inference import substitute_glossary
from gloss_mt_2.lookup import TwiLookup


def test_mark_noun_in_sentence():
    import re
    sentence = "The judge entered the court to begin the trial."
    noun = "court"
    pat = re.compile(r"(?<!\w)" + re.escape(noun) + r"(?!\w)", re.IGNORECASE)
    marked = pat.sub(f"<{noun}>", sentence, count=1)

    assert marked == "The judge entered the <court> to begin the trial."


class MockReranker:
    def rerank(self, context: str, noun_phrase: str, candidates: list) -> str:
        # Mock logic: if context contains "judge" or "trial", pick "asɛnnibea"
        if "judge" in context or "trial" in context:
            if "asɛnnibea" in candidates:
                return "asɛnnibea"
        return candidates[0]


def test_substitute_with_reranker():
    lookup = TwiLookup(custom_dict={"court": "asɛnnibea"})
    # Attach multi-candidate options
    lookup.candidates = {"court": ["kɔɔto", "asɛnnibea", "ahemfie"]}

    context = "The judge entered the court to begin the trial."
    raw_translation = "Ɔtemmufoɔ no wuraa <court> no mu sɛ ɔrebɛbue asɛm no."

    mock_reranker = MockReranker()

    final, applied = substitute_glossary(
        translation=raw_translation,
        lookup=lookup,
        direction="en-tw",
        reranker=mock_reranker,
        source_context=context,
    )

    assert "asɛnnibea no" in final
    assert applied[0]["target"] == "asɛnnibea"

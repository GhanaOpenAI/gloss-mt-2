"""Unit tests for Cross-Encoder Reranker data and scoring logic."""

import pytest
from gloss_mt_2.inference import GlossMT2Translator, normalize_glossary_input, substitute_glossary
from gloss_mt_2.lookup import TwiLookup


class MockReranker:
    def rerank(self, twi_context: str, noun_phrase: str, candidates: list) -> str:
        # Evaluates Twi context: if Twi sentence contains "ɔtemmufoɔ" (judge) or "asɛm" (case), pick "asɛnnibea"
        if "ɔtemmufoɔ" in twi_context or "asɛm" in twi_context:
            if "asɛnnibea" in candidates:
                return "asɛnnibea"
        # If context is commerce/market, pick "dwadie"
        if "Kejetia" in twi_context or "dwadibea" in twi_context:
            if "dwadie" in candidates:
                return "dwadie"
        return candidates[0]


class MockSeq2SeqModel:
    def __init__(self, output_template):
        self.output_template = output_template

    def generate(self, text: str) -> str:
        return self.output_template


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


def test_custom_list_with_reranker_no_default_dict():
    """Test passing custom candidate lists without any default dictionary."""
    mock_model = MockSeq2SeqModel("Wɔkɔhwɛɛ sɛnea <trade> no teɛ wɔ Kejetia dwadibea hɔ.")
    mock_reranker = MockReranker()

    # Pass translator with NO dictionary by default (use_default_dictionary=False)
    translator = GlossMT2Translator(
        model=mock_model,
        reranker=mock_reranker,
        use_default_dictionary=False,
    )

    assert translator.lookup is None

    # Pass custom list of pairs with multiple candidate options:
    custom_list = [
        ("trade", ["adwuma", "dwadie", "aguadi"]),
    ]

    res = translator.translate("They checked how trade is doing.", custom_glossary=custom_list)

    assert "dwadie no" in res["translation"]
    assert res["substitutions"][0]["target"] == "dwadie"
    assert res["substitutions"][0]["substituted"] is True


def test_custom_list_formats():
    """Test various custom list formats: list of tuples, list of dicts, and TSV strings."""
    # List of tuples
    res1 = normalize_glossary_input([("trade", ["dwadie", "aguadi"]), ("hospital", "ayaresabea")])
    assert res1["trade"] == ["dwadie", "aguadi"]
    assert res1["hospital"] == "ayaresabea"

    # List of dicts
    res2 = normalize_glossary_input([
        {"source": "trade", "candidates": ["dwadie", "aguadi"]},
        {"source": "court", "target": "asɛnnibea"},
    ])
    assert res2["trade"] == ["dwadie", "aguadi"]
    assert res2["court"] == "asɛnnibea"

    # List of TSV strings
    res3 = normalize_glossary_input(["trade\tdwadie\taguadi", "court\tasɛnnibea"])
    assert res3["trade"] == ["dwadie", "aguadi"]
    assert res3["court"] == "asɛnnibea"


def test_multi_word_phrase_reranking():
    """Verify that multi-word phrases with multiple words are matched and substituted properly."""
    mock_model = MockSeq2SeqModel("Aban no bɛsi <regional hospital> foforɔ wɔ Kumasi.")
    mock_reranker = MockReranker()
    translator = GlossMT2Translator(model=mock_model, reranker=mock_reranker, use_default_dictionary=False)

    custom_list = [
        ("regional hospital", ["mantam ayaresabea", "akwahosan bea"]),
        ("Ministry of Finance", "Sikasɛm Asoeɛ"),
    ]
    res = translator.translate("The government will build a regional hospital.", custom_glossary=custom_list)
    assert "mantam ayaresabea" in res["translation"]
    assert res["substitutions"][0]["source"] == "regional hospital"
    assert "<" not in res["translation"]

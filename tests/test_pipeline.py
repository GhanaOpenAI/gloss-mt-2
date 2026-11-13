"""Tests for GLOSS-MT-2 bidirectional lookup and post-translation glossary substitution."""

import pytest
from gloss_mt_2.lookup import TwiLookup
from gloss_mt_2.inference import substitute_glossary


def test_custom_dictionary():
    custom = {
        "court": "asɛnnibea",
        "decision": "atɛmmuo",
        "hospital": "ayaresabea",
    }
    lookup = TwiLookup(custom_dict=custom)

    assert lookup.lookup("court", direction="en-tw") == "asɛnnibea"
    assert lookup.lookup("Court", direction="en-tw") == "asɛnnibea"
    assert lookup.lookup("asɛnnibea", direction="tw-en") == "court"


def test_substitute_glossary_en_to_tw():
    custom = {
        "hospital": "ayaresabea",
        "ministry of finance": "Sikasɛm Asoeɛ",
        "yesterday": "ɛnora",
    }
    lookup = TwiLookup(custom_dict=custom)

    # Translation produced by model where source nouns are preserved:
    model_output = "Ministry of Finance no kɔsraa hospital no ɛnora."
    source_nouns = ["Ministry of Finance", "hospital"]

    final, applied = substitute_glossary(
        translation=model_output,
        source_terms=source_nouns,
        lookup=lookup,
        direction="en-tw",
    )

    assert "Sikasɛm Asoeɛ" in final
    assert "ayaresabea" in final
    assert len(applied) == 2


def test_substitute_glossary_tw_to_en():
    custom = {
        "doctor": "dɔkota",
        "hospital": "ayaresabea",
    }
    lookup = TwiLookup(custom_dict=custom)

    # Translation produced by tw->en model where Twi nouns are preserved:
    model_output = "The dɔkota visited the ayaresabea yesterday."
    source_nouns = ["dɔkota", "ayaresabea"]

    final, applied = substitute_glossary(
        translation=model_output,
        source_terms=source_nouns,
        lookup=lookup,
        direction="tw-en",
    )

    assert "doctor" in final
    assert "hospital" in final
    assert len(applied) == 2

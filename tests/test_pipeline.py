"""Tests for Nkrane-MT bidirectional lookup and post-translation glossary substitution."""

import pytest
from nkrane_mt.lookup import TwiLookup
from nkrane_mt.inference import substitute_glossary


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
    }
    lookup = TwiLookup(custom_dict=custom)

    # Model generates Twi with the untranslated English nouns marked in <...>:
    model_output = "<Ministry of Finance> no kɔsraa <hospital> no wɔ <TechCrunch> ɛnora."

    final, applied = substitute_glossary(
        translation=model_output,
        lookup=lookup,
        direction="en-tw",
    )

    # "Ministry of Finance" -> "Sikasɛm Asoeɛ"
    assert "Sikasɛm Asoeɛ" in final
    # "hospital" -> "ayaresabea"
    assert "ayaresabea" in final
    # "TechCrunch" is a brand name not in dict -> shed brackets
    assert "TechCrunch" in final
    assert "<" not in final
    assert ">" not in final
    assert len(applied) == 3


def test_substitute_glossary_tw_to_en():
    custom = {
        "doctor": "dɔkota",
        "hospital": "ayaresabea",
    }
    lookup = TwiLookup(custom_dict=custom)

    # Model generates English with the untranslated Twi nouns marked in <...>:
    model_output = "The <dɔkota> visited the <ayaresabea> in <Kumasi> yesterday."

    final, applied = substitute_glossary(
        translation=model_output,
        lookup=lookup,
        direction="tw-en",
    )

    # "dɔkota" -> "doctor"
    assert "doctor" in final
    # "ayaresabea" -> "hospital"
    assert "hospital" in final
    # "Kumasi" stays without brackets
    assert "Kumasi" in final
    assert "<" not in final
    assert ">" not in final
    assert len(applied) == 3

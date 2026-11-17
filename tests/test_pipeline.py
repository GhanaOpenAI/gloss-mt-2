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


def test_translate_direct_and_batch():
    from nkrane_mt.inference import NkraneMTTranslator

    class DummyModel:
        def generate(self, text):
            # Simulates model outputting natural Twi or preserved terms
            return f"Twi: {text}"

    translator = NkraneMTTranslator(
        model=DummyModel(),
        use_default_dictionary=False,
    )

    # 1. Direct translation (no marking, no dictionary substitution)
    res = translator.translate("Hello world", direct=True)
    assert res["translation"] == "Twi: Hello world"
    assert res["substitutions"] == []

    # 2. Batch translation
    texts = ["First sentence", "Second sentence", "Third sentence"]
    batch_res = translator.translate_batch(texts, batch_size=2, skip_substitution=True)
    assert len(batch_res) == 3
    assert batch_res[0] == "Twi: First sentence"
    assert batch_res[1] == "Twi: Second sentence"
    assert batch_res[2] == "Twi: Third sentence"

    # 3. Stream translation
    streamed = list(translator.translate_stream(iter(texts), batch_size=2))
    assert streamed == batch_res


def test_chunking_and_reassembly():
    from nkrane_mt.inference import NkraneMTTranslator, chunk_text

    # Test chunking helper
    long_doc = "This is a sentence. " * 50
    chunks = chunk_text(long_doc, max_tokens=30)
    assert len(chunks) > 1
    assert "".join(chunks).replace(" ", "") == long_doc.strip().replace(" ", "")

    class DummyModel:
        def generate(self, text):
            return f"Twi({text})"

    translator = NkraneMTTranslator(model=DummyModel())

    # Mixed batch: short, very long paragraph, and another short sentence
    mixed_input = [
        "Short sentence one.",
        "Sentence A. " * 30, # Triggers chunking
        "Short sentence two."
    ]

    results = translator.translate_batch(mixed_input, max_length=50, chunk_long_texts=True)
    assert len(results) == 3
    assert results[0] == "Twi(Short sentence one.)"
    assert results[2] == "Twi(Short sentence two.)"
    assert "Twi(" in results[1]

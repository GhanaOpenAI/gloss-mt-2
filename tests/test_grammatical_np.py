import re
import spacy
from nltk.corpus import wordnet as wn

CONCRETE_LEXNAMES = {
    "noun.artifact", "noun.person", "noun.location", "noun.plant",
    "noun.animal", "noun.substance", "noun.food", "noun.body", "noun.object",
    "noun.group"
}

ABSTRACT_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence",
    "ship", "hood", "dom", "ism", "cracy", "logy", "phy",
    "ure"
)

CONCRETE_EXCEPTIONS = {
    "instrument", "monument", "apartment", "document", "garment", "settlement",
    "building", "painting", "drawing", "clothing", "ring", "wing", "spring",
    "gold", "silver", "money", "court", "market", "office", "church", "school",
    "hospital", "bank", "prison", "station", "university", "college",
    "police", "army", "family", "team", "equipment", "device", "vehicle",
    "governor", "president", "king", "minister", "chief", "director",
    "persons", "people", "children", "women", "men", "patients", "workers"
}

ABSTRACT_EXCEPTIONS = {
    "part", "role", "aspect", "matter", "issue", "problem", "concept", "process",
    "way", "manner", "factor", "case", "point", "feature", "system", "relation"
}

def is_abstract_word(word: str) -> bool:
    lower = word.lower()
    if lower in CONCRETE_EXCEPTIONS:
        return False
    if lower in ABSTRACT_EXCEPTIONS:
        return True
    if any(lower.endswith(sfx) for sfx in ABSTRACT_SUFFIXES):
        return True
    syns = wn.synsets(lower, pos=wn.NOUN)
    if syns:
        if any(s.lexname() in CONCRETE_LEXNAMES for s in syns[:3]):
            if syns[0].lexname() in {"noun.attribute", "noun.feeling", "noun.state", "noun.motive", "noun.relation"}:
                return True
            return False
        return True
    return any(lower.endswith(sfx) for sfx in ABSTRACT_SUFFIXES)

nlp = spacy.load("en_core_web_sm")
leading_articles = re.compile(r"^(?:the|a|an|this|that|these|those)\s+", re.IGNORECASE)
NOMINAL_DEPS = {"compound", "amod", "nmod", "nummod", "det", "poss", "case", "appos"}
NOMINAL_PP_PREPS = {"of", "with", "without", "for", "in"}

def is_nominal_pp(prep_token):
    return prep_token.text.lower() in NOMINAL_PP_PREPS and prep_token.head.pos_ in ("NOUN", "PROPN")

def extract_selective_noun_phrases(text: str) -> list:
    doc = nlp(text)
    heads = []
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            if token.dep_ in ("compound", "nmod"):
                continue
            if token.dep_ in ("pobj", "pcomp") and is_nominal_pp(token.head) and token.head.head.pos_ in ("NOUN", "PROPN"):
                continue
            heads.append(token)
            
    selected = []
    seen = set()
    for head in heads:
        nodes = {head}
        def collect(node):
            for child in node.children:
                if child.dep_ in NOMINAL_DEPS:
                    nodes.add(child)
                    collect(child)
                elif child.dep_ == "prep" and is_nominal_pp(child):
                    nodes.add(child)
                    for pobj in child.children:
                        if pobj.dep_ in ("pobj", "pcomp"):
                            nodes.add(pobj)
                            collect(pobj)
        collect(head)
        sorted_nodes = sorted(list(nodes), key=lambda t: t.i)
        span_text = doc[sorted_nodes[0].i : sorted_nodes[-1].i + 1].text.strip()
        base = leading_articles.sub("", re.sub(r"^[\W_]+|[\W_]+$", "", span_text)).strip()
        if not base or len(base) < 2:
            continue
            
        words = base.split()
        m = re.search(r"\b" + re.escape(base) + r"\b", text)
        start_pos = m.start() if m else 0
        prefix = text[:start_pos].strip(" \"'([{<-")
        is_sent_start = (len(prefix) == 0)
        
        has_capitalized = any(w[:1].isupper() and any(c.isalpha() for c in w) for w in words)
        root_word = head.text
        is_abstract = is_abstract_word(root_word)
        
        keep = False
        if has_capitalized:
            if is_sent_start:
                if len(words) > 1:
                    keep = True
                else:
                    keep = not is_abstract
            else:
                keep = True
        else:
            keep = not is_abstract
            
        if keep and base.lower() not in seen:
            seen.add(base.lower())
            selected.append(base)
            
    return selected

test_sentences = [
    "Persons with disabilities are an important part of any economy.",
    "The president of Ghana visited the Ministry of Finance in May 2026.",
    "The Bank of Ghana governor spoke about hospital equipment in Kumasi.",
    "The Minister for Health announced a new policy for the regional hospital.",
    "Information was provided to Kantanka Automotive Company Limited regarding early development.",
    "The King of Asante attended the Kejetia market place opening with news reporters."
]

for s in test_sentences:
    print("SENTENCE:", s)
    print("  EXTRACTED:", extract_selective_noun_phrases(s))
    print()

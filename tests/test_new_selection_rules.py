import re
import spacy
from nltk.corpus import wordnet as wn

CONCRETE_LEXNAMES = {
    'noun.artifact', 'noun.person', 'noun.location', 'noun.plant',
    'noun.animal', 'noun.substance', 'noun.food', 'noun.body', 'noun.object',
    'noun.group'
}

ABSTRACT_SUFFIXES = (
    'tion', 'sion', 'ment', 'ness', 'ity', 'ance', 'ence',
    'ship', 'hood', 'dom', 'ism', 'cracy', 'logy', 'phy',
    'ure'
)

CONCRETE_EXCEPTIONS = {
    'instrument', 'monument', 'apartment', 'document', 'garment', 'settlement',
    'building', 'painting', 'drawing', 'clothing', 'ring', 'wing', 'spring',
    'gold', 'silver', 'money', 'court', 'market', 'office', 'church', 'school',
    'hospital', 'bank', 'prison', 'station', 'university', 'college',
    'police', 'army', 'family', 'team'
}

def is_abstract_word(word: str) -> bool:
    lower = word.lower()
    if lower in CONCRETE_EXCEPTIONS:
        return False
    if any(lower.endswith(sfx) for sfx in ABSTRACT_SUFFIXES):
        return True
    syns = wn.synsets(lower, pos=wn.NOUN)
    if not syns:
        return False
    primary_lex = syns[0].lexname()
    return primary_lex not in CONCRETE_LEXNAMES

nlp = spacy.load('en_core_web_sm', disable=['lemmatizer'])
leading_articles = re.compile(r'^(?:the|a|an|this|that|these|those)\s+', re.IGNORECASE)

def extract_selective_nouns(text: str):
    doc = nlp(text)
    selected = []
    
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == 'PRON':
            continue
            
        parts = re.split(r'\s+\b(?:and|or)\b\s+', chunk.text)
        for part in parts:
            part_clean = re.sub(r'^[\W_]+|[\W_]+$', '', part.strip())
            base = leading_articles.sub('', part_clean).strip()
            if not base or len(base) < 2:
                continue
            
            words = base.split()
            root_word = chunk.root.text if len(words) > 1 and chunk.root.text in words else words[-1]
            is_abstract = is_abstract_word(root_word)
            
            m = re.search(r'\b' + re.escape(base) + r'\b', text)
            start_pos = m.start() if m else 0
            prefix = text[:start_pos].strip(' "\'([{-')
            is_sent_start = (len(prefix) == 0)
            
            is_capitalized = words[0][:1].isupper()
            
            keep = False
            if is_capitalized:
                if is_sent_start:
                    if len(words) > 1:
                        keep = True
                    else:
                        keep = not is_abstract
                else:
                    keep = True
            else:
                keep = not is_abstract
                
            selected.append((base, keep, f'abstract={is_abstract}, cap={is_capitalized}, start={is_sent_start}'))
    return selected

test_sentences = [
    'Information was provided to Kantanka Automotive Company Limited regarding early development and hospital equipment in Ghana.',
    'Telecel Ghana built a regional hospital and provided peace and security for the soldiers.',
    'Hospital is an important facility.',
    'Development requires money, road, and patience.'
]

for s in test_sentences:
    print('SENTENCE:', s)
    for term, keep, info in extract_selective_nouns(s):
        tag = 'KEPT' if keep else 'EXCLUDED'
        print(f'  [{tag:<8}] {term:<40} ({info})')
    print()

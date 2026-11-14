import sys
import os

from gloss_mt_2.inference import substitute_glossary
from gloss_mt_2.reranker import ContextualReranker

text = (
    "Efi <provided information> no mu Nea edidi so yi yɛ abakɔsɛm a ɛfa "
    "<Kantanka Automotive Company Limited> a <establishment history> ne "
    "<early development> ho <complete and polished investigation>. "
    "<its founding> wɔ bere a edi kan no mu ne nsakrae a edi hɔ "
    "<indigenous automobile manufacturing and assembly enterprise>. ### 1."
)

print("=" * 70)
print("1. RAW GENERATED TWI (WITH PRESERVED NOUN PHRASES)")
print("=" * 70)
print(text)

print("\n" + "=" * 70)
print("2. LOADING AFRO-XLMR CROSS-ENCODER RERANKER")
print("=" * 70)
reranker_path = "/mnt/volume_d2wey28/projects/gloss-mt-2/checkpoints/reranker-afro-xlmr/final"
if not os.path.exists(reranker_path):
    reranker_path = "ghanaopenai/gloss-mt-2-reranker-afro-xlmr"

reranker = ContextualReranker(reranker_path)

# Custom domain glossary with multiple candidates for the reranker to score:
custom_glossary = [
    ("provided information", ["nsɛm a wɔde ama", "amaneɛbɔ a ɛwɔ hɔ"]),
    ("establishment history", ["ntimideɛ ho abakɔsɛm", "sɛnea wɔde sii hɔ ho abakɔsɛm"]),
    ("early development", ["nkɔsoɔ a edi kan", "mpontuo mfiaseɛ"]),
    ("complete and polished investigation", ["nhwehwɛmu a edi mu na wɔapene so", "nhwehwɛmu a ɛkɔ akyiri"]),
    ("its founding", ["ne fapem", "bere a wɔhyehyɛeɛ no"]),
    ("indigenous automobile manufacturing and assembly enterprise", [
        "ɔman no ankasa hyɛn yɛ ne nkabom nnwumakuo",
        "Ghana hyɛn yɛ adwumakuo a ɛwɔ ha"
    ]),
    # "Kantanka Automotive Company Limited" is NOT in the glossary
    # so the pipeline preserves it cleanly without brackets!
]

final_text, substitutions = substitute_glossary(
    translation=text,
    custom_glossary=custom_glossary,
    reranker=reranker,
    source_context=text,
)

print("\n" + "=" * 70)
print("3. FINAL TRANSLATION (SUBSTITUTED & PRESERVED)")
print("=" * 70)
print(final_text)

print("\n" + "=" * 70)
print("4. DETAILED SUBSTITUTION BREAKDOWN")
print("=" * 70)
for s in substitutions:
    tag = "Reranked & Substituted" if s["substituted"] else "Preserved Proper Entity"
    print(f"  • Source Phrase: {s['source']}")
    print(f"    Selected Twi:  {s['target']} ({tag})\n")

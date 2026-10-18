"""
build_corpus.py - generate data/fixtures/*.json and data/gold/*.json.

Documents are written with inline markup:   [[CATEGORY|exact text]]
The script strips the markup, computes exact character offsets (Python indices,
end-exclusive) and computes doc_id = first 16 hex chars of SHA-256 of the raw text.

QUASI_COMBO spans have no position, so they are listed separately per document.

All 20 documents are SYNTHETIC, written in the style of ChatDoctor (HealthCareMagic)
and Enron email records. No real person is described. To use real records, replace
`text` with a real record and re-annotate by hand following docs/annotation_guide.md.

Usage:  python tools/build_corpus.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIX_DIR = ROOT / "data" / "fixtures"
GOLD_DIR = ROOT / "data" / "gold"

MARK = re.compile(r"\[\[([A-Z_]+)\|(.+?)\]\]", re.S)

# (case number, source style, marked-up text, [quasi-combo descriptions])
DOCS: list[tuple[int, str, str, list[str]]] = [
    (
        1,
        "chatdoctor",
        "Hi doctor, my name is [[NAME|Priya Raman]]. I am a [[AGE|34 year old]] "
        "[[OCCUPATION|pediatric neurosurgeon]] and I live in [[LOCATION|Coonoor]], a small town "
        "in the hills. For the last three weeks I have had episodes of [[MEDICAL_CONDITION|vertigo]] "
        "with a ringing in my left ear, usually in the early morning before surgery. "
        "[[RELATION|My sister]] has the same problem. You can reach me on [[PHONE|+91 98450 12345]] "
        "if you need more details. Thank you.",
        ["rare occupation (pediatric neurosurgeon) + small town (Coonoor) + vertigo"],
    ),
    (
        2,
        "chatdoctor",
        "Hello doctor. [[RELATION|My daughter]] is [[AGE|6 years old]] and attends "
        "[[LOCATION|Greenfield Primary School]] in [[LOCATION|Tiruppur]]. Since "
        "[[DATE|12 September 2023]] she has had a dry cough at night and wheezing after games class. "
        "A classmate was diagnosed with [[MEDICAL_CONDITION|asthma]] last year. Should I give her a "
        "cough syrup or see a pediatrician?",
        ["child age + named school + small town + precise onset date"],
    ),
    (
        3,
        "chatdoctor",
        "Dear doctor, I am writing about [[RELATION|my husband]]'s reports. Patient ID: "
        "[[ID_NUMBER|HC-2291746]]. He was admitted on [[DATE|3 March 2022]] and discharged on "
        "[[DATE|9 March 2022]] after a coronary artery bypass for "
        "[[MEDICAL_CONDITION|coronary artery disease]]. Please reply to "
        "[[EMAIL|rajesh.kumar84@examplemail.com]] with the discharge summary. His blood pressure has "
        "been around 150/95 since then and he gets breathless climbing stairs. Is this normal?",
        [],
    ),
    (
        4,
        "chatdoctor",
        "Hi, I am [[NAME|Mohammed Faizal]], [[AGE|58 years old]]. My address is "
        "[[ADDRESS|14/3 Lakeview Apartments, Bannerghatta Road, Bengaluru 560076]]. I was diagnosed "
        "with [[MEDICAL_CONDITION|type 2 diabetes]] in [[DATE|2015]] and take metformin 500 mg twice "
        "daily. Lately my feet feel numb at night. Can this be [[MEDICAL_CONDITION|neuropathy]]? Is "
        "it safe to walk daily?",
        [],
    ),
    (
        5,
        "chatdoctor",
        "Doctor, [[RELATION|my wife]] is [[AGE|92 years old]] and was diagnosed with "
        "[[MEDICAL_CONDITION|atrial fibrillation]] last month at [[LOCATION|St. Thomas Hospital]]. "
        "She is forgetful and refuses to take the blood thinner. We live alone on a farm outside "
        "[[LOCATION|Ooty]]. How can I make sure she takes the tablets?",
        ["extreme age (92) + diagnosis + named hospital + small-town residence"],
    ),
    (
        6,
        "chatdoctor",
        "I am a [[OCCUPATION|lighthouse keeper]] on [[LOCATION|Kavaratti island]] and I am "
        "[[AGE|47 years old]]. For the past two months I have had a persistent rash on both hands "
        "that gets worse after cleaning the lens with solvent. A dermatologist suggested "
        "[[MEDICAL_CONDITION|contact dermatitis]] but nothing helps. I am the only keeper on the "
        "island so I cannot take leave easily. What should I do?",
        ["rare occupation (lighthouse keeper) + single island location + dermatitis"],
    ),
    (
        7,
        "chatdoctor",
        "[[RELATION|My son]] has been experiencing left-sided abdominal pain and bilateral pelvic "
        "pain for approximately 10 weeks. He is [[AGE|32 years old]], [[PHYSICAL|6 ft 3 inches]] "
        "tall, and weighs [[PHYSICAL|approximately 220 pounds]]. Blood tests showed "
        "[[MEDICAL_CONDITION|enlarged lymph nodes]] and mild [[MEDICAL_CONDITION|leukocytosis]]. "
        "The pain is worse after meals and nothing has helped so far. What tests should we ask for "
        "next?",
        [
            "left-sided abdominal pain + bilateral pelvic pain (symptom combination)",
            "enlarged lymph nodes + leukocytosis + roughly 10 weeks of unresolved symptoms",
        ],
    ),
    (
        8,
        "chatdoctor",
        "Hi doctor. I am [[AGE|29 years old]] and I delivered my baby on [[DATE|21 January 2024]] at "
        "[[LOCATION|Fortis Hospital, Mulund]] by emergency C-section. Since "
        "[[DATE|2 February 2024]] I have had fever and a red, swollen stitch line. "
        "[[RELATION|My husband]] says it is normal but I am worried. I work as a "
        "[[OCCUPATION|radiographer]] so I know infection can be serious. Should I go to emergency?",
        ["precise delivery date + named hospital + age"],
    ),
    (
        9,
        "chatdoctor",
        "Hello, I am [[NAME|Arjun Nair]], a professional [[OCCUPATION|kabaddi player]]. I am "
        "[[AGE|26 years old]], [[PHYSICAL|5 ft 9 inches]] and [[PHYSICAL|78 kg]]. Last season I "
        "tore my [[MEDICAL_CONDITION|anterior cruciate ligament]] while playing for a team in "
        "[[LOCATION|Pune]]. How long should rehab take before I can play again?",
        ["professional athlete occupation + team city + ACL injury"],
    ),
    (
        10,
        "chatdoctor",
        "I have had a mild headache for two days after long hours on the computer. I have taken "
        "paracetamol once. Is it okay to take it again, and when should I worry about a headache?",
        [],
    ),
    (
        11,
        "chatdoctor",
        "Hi doctor, I am [[AGE|17 years old]] and I have [[MEDICAL_CONDITION|acne]] on my cheeks and "
        "forehead. I have tried face wash but it keeps coming back. What treatments are safe, and "
        "does diet matter?",
        [],
    ),
    (
        12,
        "enron",
        "From: [[NAME|Daniel Whitcomb]]\nSent: [[DATE|Tuesday, October 9, 2001]] 3:42 PM\n"
        "To: [[NAME|Laura Fenn]]\nSubject: Call with regulators\n\n"
        "[[NAME|Laura]], please call me at [[PHONE|(415) 555-0142]] before the meeting. You can "
        "also write to [[EMAIL|daniel.whitcomb@example-energy.com]]. We will meet at "
        "[[ADDRESS|1400 Smith Street, Houston, TX 77002]] and I will bring the revised filing. "
        "Please do not forward this note outside the group.\n\n[[NAME|Daniel]]",
        [],
    ),
    (
        13,
        "enron",
        "Subject: HR file update\n\nHi [[NAME|Marta]],\n\nPlease update the HR file for "
        "[[NAME|Kevin Oyelaran]], employee [[ID_NUMBER|E-40871]]. He moved from the "
        "[[LOCATION|Portland]] trading desk to Houston on [[DATE|15 June 2000]]. His new address is "
        "[[ADDRESS|2210 Bagby Street, Apt 12, Houston, TX 77006]] and his phone is "
        "[[PHONE|713-555-0199]].\n\nThanks,\n[[NAME|Helen Park]]",
        [],
    ),
    (
        14,
        "enron",
        "Subject: Pipeline review\n\nTeam,\n\nThe quarterly pipeline review has been moved to next "
        "week. Please bring updated volume forecasts and the revised capacity table. Slides should "
        "be circulated the day before. Let me know if you have conflicts.\n\nThanks",
        [],
    ),
    (
        15,
        "enron",
        "Subject: Leave request\n\n[[NAME|Marta]], [[NAME|Samuel Ortega]], a "
        "[[OCCUPATION|natural gas scheduler]] on the [[LOCATION|Tulsa]] desk, will be on medical "
        "leave from [[DATE|1 August 2001]] to [[DATE|30 September 2001]] for treatment of "
        "[[MEDICAL_CONDITION|multiple myeloma]]. He is [[AGE|51 years old]]. "
        "[[RELATION|His wife]] will be the contact: [[PHONE|918-555-0133]]. Please keep this "
        "confidential.",
        ["occupation + small desk location + rare cancer + exact leave dates"],
    ),
    (
        16,
        "chatdoctor",
        "Doctor, my insurance policy number is [[ID_NUMBER|POL-77-410398]] and I need a second "
        "opinion on my [[MEDICAL_CONDITION|thyroid nodule]]. The ultrasound on "
        "[[DATE|18 April 2024]] showed a nodule of 1.8 cm and the biopsy is pending. I am "
        "[[AGE|41 years old]] and work as a [[OCCUPATION|school librarian]]. [[RELATION|My mother]] "
        "had [[MEDICAL_CONDITION|thyroid cancer]] at the same age. Should I worry?",
        [],
    ),
    (
        17,
        "chatdoctor",
        "[[RELATION|My husband]] is [[AGE|63 years old]] and was diagnosed with "
        "[[MEDICAL_CONDITION|amyloidosis]] on [[DATE|7 November 2023]] at "
        "[[LOCATION|AIIMS Delhi]]. He used to work as a [[OCCUPATION|tea taster]] in "
        "[[LOCATION|Darjeeling]]. He has lost his sense of taste and has swelling of the tongue. Is "
        "there a connection with his job?",
        ["rare occupation (tea taster) + rare condition (amyloidosis) + small-city location"],
    ),
    (
        18,
        "chatdoctor",
        "Hi doctor, I have read that vitamin D deficiency is common in people who work indoors. How "
        "much sunlight is enough each day, and can supplements replace it? Are there side effects "
        "if I take too much?",
        [],
    ),
    (
        19,
        "chatdoctor",
        "Hi doctor, I am a [[AGE|20 year old]] [[OCCUPATION|undergraduate student]] at "
        "[[LOCATION|IIT Madras]]. Since [[DATE|5 October 2024]] I have had palpitations, and a Holter "
        "test last week showed [[MEDICAL_CONDITION|supraventricular tachycardia]]. I train as a "
        "[[OCCUPATION|national-level fencer]] and follow a strict diet. Is it safe to keep "
        "competing?",
        [
            "niche athletic occupation (national-level fencer) + named institution + cardiac arrhythmia"
        ],
    ),
    (
        20,
        "chatdoctor",
        "Hello, this is [[NAME|Anita D'Souza]] ([[EMAIL|anita.dsouza@examplemail.com]]). "
        "[[RELATION|My father]], [[NAME|Cyril D'Souza]], is [[AGE|71 years old]] and lives at "
        "[[ADDRESS|Flat 2B, Sea View Court, Fort Kochi 682001]]. He had an "
        "[[MEDICAL_CONDITION|ischemic stroke]] on [[DATE|14 May 2024]] and now has weakness on his "
        "right side. A neurologist at [[LOCATION|Aster Medcity]] has started physiotherapy. How "
        "long does recovery usually take?",
        [],
    ),
]


def parse(marked: str) -> tuple[str, list[dict]]:
    """Strip markup, return (clean_text, spans with exact offsets)."""
    clean: list[str] = []
    spans: list[dict] = []
    pos = 0  # position in marked text
    out_len = 0
    for m in MARK.finditer(marked):
        before = marked[pos : m.start()]
        clean.append(before)
        out_len += len(before)
        cat, txt = m.group(1), m.group(2)
        spans.append({"text": txt, "category": cat, "start": out_len, "end": out_len + len(txt)})
        clean.append(txt)
        out_len += len(txt)
        pos = m.end()
    clean.append(marked[pos:])
    return "".join(clean), spans


def doc_id_for(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    FIX_DIR.mkdir(parents=True, exist_ok=True)
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    for n, source, marked, quasi in DOCS:
        text, spans = parse(marked)
        assert len(text) <= 2000, f"case_{n:02d} too long ({len(text)})"
        for s in spans:
            assert text[s["start"] : s["end"]] == s["text"], (n, s)
        for q in quasi:
            spans.append({"text": q, "category": "QUASI_COMBO", "start": None, "end": None})
        did = doc_id_for(text)
        name = f"case_{n:02d}.json"
        (FIX_DIR / name).write_text(
            json.dumps(
                {"doc_id": did, "source": f"synthetic-{source}", "text": text},
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        (GOLD_DIR / name).write_text(
            json.dumps(
                {
                    "doc_id": did,
                    "text": text,
                    "spans": spans,
                    "annotator": "D",
                    "reviewed_by": None,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"wrote {len(DOCS)} fixtures and gold files")


if __name__ == "__main__":
    main()

"""Make the OCR test set: synthetic hospital, finance and legal documents whose text is known.

Every document is built from blocks (letterhead, label/value lines, sections, tables, signature
lines), so the exact text of every page is known without typing anything by hand. Electronic
PDFs keep a real text layer; "scans" and "photos" are the same pages rendered to images and
degraded (tilt, blur, noise, shadow, JPEG); handwriting is drawn with handwriting fonts, word
by word with a wobbling baseline. All people, hospitals, banks and numbers are made up.

    python tools/make_ocr_testset.py                    # 5 of each type -> data/ocr_test/
    python tools/make_ocr_testset.py --per-type 2 --out some/dir --seed 7

Writes ``docs/<id>.pdf|jpg`` and ``gold.jsonl`` (one line per document, see data/ocr_test/README.md).
Fonts (Google Fonts, OFL/Apache) are downloaded once into ``<out>/.fonts``.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import math
import random
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sanitise.ingest.headers import TOP, _key

FONT_BASE = "https://github.com/google/fonts/raw/main/"
PRINT_FONTS = {"Lato-Regular": "ofl/lato/Lato-Regular.ttf", "Lato-Bold": "ofl/lato/Lato-Bold.ttf"}
HAND_FONTS = {
    "Kalam": "ofl/kalam/Kalam-Regular.ttf",
    "IndieFlower": "ofl/indieflower/IndieFlower-Regular.ttf",
    "ShadowsIntoLight": "ofl/shadowsintolight/ShadowsIntoLight.ttf",
    "GochiHand": "ofl/gochihand/GochiHand-Regular.ttf",
    "Caveat": "ofl/caveat/Caveat%5Bwght%5D.ttf",
    "ReenieBeanie": "ofl/reeniebeanie/ReenieBeanie.ttf",
    "NothingYouCouldDo": "ofl/nothingyoucoulddo/NothingYouCouldDo.ttf",
    "HomemadeApple": "apache/homemadeapple/HomemadeApple-Regular.ttf",
    "DawningofaNewDay": "ofl/dawningofanewday/DawningofaNewDay.ttf",
}
# rough legibility: the first ones are neat, the last ones are hard cursive
HAND_SIZE = {"ReenieBeanie": 1.25, "NothingYouCouldDo": 0.9, "HomemadeApple": 0.8,
             "DawningofaNewDay": 1.1, "Caveat": 1.1, "ShadowsIntoLight": 1.05}

TYPES = {
    # id: (domain, file kind, what it tests)
    "discharge_e": ("health", "pdf", "electronic 2-page discharge summary, repeated header/footer"),
    "lab_e": ("health", "pdf", "electronic lab report with a results table"),
    "discharge_scan": ("health", "pdf", "scanned 2-page discharge summary (image-only PDF)"),
    "lab_photo": ("health", "jpg", "phone photo of a printed lab report"),
    "prescription_hw": ("health", "jpg", "handwritten OPD prescription on a printed pad"),
    "consent_form_hw": ("health", "jpg", "printed consent form filled in by hand"),
    "mixed_file": ("health", "pdf", "electronic lab page + scanned form + prescription photo"),
    "bank_statement_e": ("finance", "pdf", "electronic bank statement with transactions"),
    "affidavit_scan": ("law", "pdf", "scanned legal affidavit"),
}

# ------------------------------------------------------------------ made-up data

FIRST_M = ["Ravi", "Arun", "Suresh", "Mohammed", "Karthik", "Vikram", "Rahul", "Imran", "Prakash",
           "Joseph", "Naveen", "Harish", "Abdul", "Sanjay", "Ganesh", "Deepak", "Manoj", "Rohan"]
FIRST_F = ["Priya", "Lakshmi", "Ananya", "Fatima", "Kavya", "Meera", "Divya", "Shalini", "Asha",
           "Mary", "Nandini", "Pooja", "Sunita", "Ayesha", "Rekha", "Bhavana", "Sneha", "Radha"]
LAST = ["Sharma", "Reddy", "Nair", "Iyer", "Khan", "Patil", "Gowda", "Menon", "Joshi", "Das",
        "Pillai", "Rao", "Kulkarni", "Hegde", "Fernandes", "Shetty", "Verma", "Bhat", "Naidu"]
PLACES = [("Bengaluru", "Karnataka", "5600"), ("Mysuru", "Karnataka", "5700"),
          ("Pune", "Maharashtra", "4110"), ("Chennai", "Tamil Nadu", "6000"),
          ("Hyderabad", "Telangana", "5000"), ("Kochi", "Kerala", "6820"),
          ("Mangaluru", "Karnataka", "5750"), ("Nashik", "Maharashtra", "4220")]
AREAS = ["Jayanagar", "Indiranagar", "Gandhi Nagar", "Shivaji Nagar", "Vijay Nagar", "Model Town",
         "Ashok Nagar", "Rajaji Nagar", "Nehru Colony", "Lake View Layout", "MG Road"]
STREETS = ["2nd Cross", "5th Main", "Temple Street", "Station Road", "Church Road", "11th Cross",
           "Park Avenue", "Market Road", "3rd Main"]
HOSPITALS = ["Sri Lakshmi Multispeciality Hospital", "Green Valley Medical Centre",
             "Sunrise Hospital and Research Centre", "Kaveri Care Hospital",
             "St. Anne's Mission Hospital", "Deccan Heart and General Hospital"]
BANKS = [("Kaveri Co-operative Bank", "KCBL"), ("Deccan National Bank", "DNBK"),
         ("Western Coast Bank", "WCBL")]
QUALS = ["MBBS, MD (General Medicine)", "MBBS, DNB (Internal Medicine)", "MBBS, MD (Paediatrics)",
         "MBBS, MS (General Surgery)", "MBBS, MD (Pulmonology)"]

CONDITIONS = [
    {
        "dx": "Dengue fever with thrombocytopenia",
        "co": "Fever with chills for 4 days, body ache, headache and vomiting",
        "hx": "No known diabetes or hypertension. No previous admissions.",
        "oe": "Temp 101.8 F, Pulse 104/min, BP 110/70 mmHg, no bleeding spots",
        "course": "Treated with IV fluids and antipyretics. Platelet count fell to 38,000 on day 2 "
                  "and recovered to 1,20,000 by day 5. No bleeding. Fever settled on day 4.",
        "meds": [("Tab. Paracetamol", "650 mg", "1-1-1", "5 days"),
                 ("ORS sachets", "1 sachet", "in 1 litre water", "3 days")],
        "advice": "Plenty of oral fluids, avoid painkillers other than paracetamol",
        "labs": [("Haemoglobin", "13.8", "g/dL", "13.0 - 17.0"),
                 ("Total WBC count", "3,200", "cells/cumm", "4,000 - 11,000"),
                 ("Platelet count", "38,000", "/cumm", "1,50,000 - 4,50,000"),
                 ("Haematocrit", "46.2", "%", "40 - 50"),
                 ("NS1 antigen", "Positive", "-", "Negative"),
                 ("SGPT", "88", "U/L", "7 - 56")],
        "rx": ["Tab. Paracetamol 650 mg 1-1-1 x 5 days", "ORS 1 sachet in 1 L water daily",
               "Tab. Pantoprazole 40 mg 1-0-0 before food x 5 days"],
    },
    {
        "dx": "Community acquired pneumonia, right lower lobe",
        "co": "Cough with yellow sputum for 6 days, fever and breathlessness on walking",
        "hx": "Known case of type 2 diabetes for 8 years on metformin.",
        "oe": "Temp 100.6 F, RR 24/min, SpO2 91% on room air, crackles at right base",
        "course": "Started on IV ceftriaxone and azithromycin with oxygen by nasal prongs. "
                  "Oxygen weaned off on day 3. Sugars controlled with insulin.",
        "meds": [("Tab. Amoxicillin-Clavulanate", "625 mg", "1-0-1", "5 days"),
                 ("Tab. Metformin", "500 mg", "1-0-1", "to continue"),
                 ("Syp. Ascoril", "10 ml", "1-1-1", "5 days")],
        "advice": "Steam inhalation, breathing exercises, check sugar twice a week",
        "labs": [("Haemoglobin", "11.9", "g/dL", "12.0 - 15.0"),
                 ("Total WBC count", "15,600", "cells/cumm", "4,000 - 11,000"),
                 ("Neutrophils", "84", "%", "40 - 75"),
                 ("CRP", "96", "mg/L", "< 6"),
                 ("Random blood sugar", "248", "mg/dL", "70 - 140"),
                 ("Serum creatinine", "1.1", "mg/dL", "0.6 - 1.2")],
        "rx": ["Tab. Augmentin 625 mg 1-0-1 x 5 days", "Syp. Ascoril 10 ml TDS",
               "Tab. Metformin 500 mg 1-0-1 continue"],
    },
    {
        "dx": "Acute gastroenteritis with moderate dehydration",
        "co": "Loose stools 8 to 10 times a day for 2 days and vomiting",
        "hx": "Ate outside food 3 days ago. No chronic illness.",
        "oe": "Pulse 112/min, BP 96/60 mmHg, dry tongue, abdomen soft",
        "course": "Rehydrated with IV Ringer lactate. Stools reduced by day 2. Tolerating oral diet.",
        "meds": [("Tab. Ofloxacin-Ornidazole", "1 tab", "1-0-1", "5 days"),
                 ("Cap. Racecadotril", "100 mg", "1-1-1", "3 days"),
                 ("Sachet Probiotic", "1 sachet", "1-0-1", "5 days")],
        "advice": "Boiled water, soft diet, ORS after every loose stool",
        "labs": [("Serum sodium", "131", "mEq/L", "135 - 145"),
                 ("Serum potassium", "3.2", "mEq/L", "3.5 - 5.0"),
                 ("Blood urea", "52", "mg/dL", "15 - 40"),
                 ("Serum creatinine", "1.4", "mg/dL", "0.6 - 1.2"),
                 ("Haemoglobin", "14.6", "g/dL", "13.0 - 17.0")],
        "rx": ["Tab. O2 1-0-1 x 5 days", "Cap. Redotil 100 mg 1-1-1 x 3 days",
               "ORS after each loose stool"],
    },
    {
        "dx": "Acute appendicitis, laparoscopic appendicectomy done",
        "co": "Pain in the right lower abdomen for 1 day with nausea and low grade fever",
        "hx": "No previous surgery. No drug allergy.",
        "oe": "Tenderness at McBurney's point, rebound tenderness present, Temp 99.8 F",
        "course": "Ultrasound showed an inflamed appendix. Laparoscopic appendicectomy done on the "
                  "day of admission. Post-operative period uneventful. Port sites healthy.",
        "meds": [("Tab. Cefuroxime", "500 mg", "1-0-1", "5 days"),
                 ("Tab. Aceclofenac-Paracetamol", "1 tab", "1-0-1", "3 days"),
                 ("Tab. Pantoprazole", "40 mg", "1-0-0", "5 days")],
        "advice": "Avoid lifting heavy weights for 4 weeks, keep dressing dry",
        "labs": [("Total WBC count", "13,900", "cells/cumm", "4,000 - 11,000"),
                 ("Neutrophils", "81", "%", "40 - 75"),
                 ("Haemoglobin", "12.7", "g/dL", "12.0 - 15.0"),
                 ("Bleeding time", "2 min 30 s", "-", "1 - 5 min"),
                 ("Clotting time", "5 min 10 s", "-", "4 - 9 min")],
        "rx": ["Tab. Ceftum 500 mg 1-0-1 x 5 days", "Tab. Zerodol-P 1-0-1 x 3 days",
               "Dressing on day 3 and suture removal on day 8"],
    },
    {
        "dx": "Acute exacerbation of bronchial asthma",
        "co": "Wheezing and breathlessness since last night, unable to sleep",
        "hx": "Asthma since childhood, uses inhaler occasionally. Allergic to dust.",
        "oe": "RR 28/min, SpO2 89% on room air, bilateral wheeze, speaking in phrases",
        "course": "Nebulised with salbutamol and ipratropium, IV hydrocortisone for 2 days. "
                  "Shifted to oral steroids. Peak flow improved from 180 to 360 L/min.",
        "meds": [("Inhaler Budesonide-Formoterol", "200 mcg", "2 puffs twice daily", "to continue"),
                 ("Tab. Prednisolone", "40 mg", "1-0-0", "5 days"),
                 ("Tab. Montelukast", "10 mg", "0-0-1", "1 month")],
        "advice": "Use spacer with inhaler, avoid dust and smoke, rinse mouth after inhaler",
        "labs": [("Haemoglobin", "13.1", "g/dL", "12.0 - 15.0"),
                 ("Absolute eosinophil count", "680", "cells/cumm", "40 - 440"),
                 ("Total IgE", "540", "IU/mL", "< 100"),
                 ("Serum potassium", "3.6", "mEq/L", "3.5 - 5.0")],
        "rx": ["Foracort 200 inhaler 2 puffs BD", "Tab. Wysolone 40 mg 1-0-0 x 5 days",
               "Tab. Montair 10 mg 0-0-1 x 1 month"],
    },
    {
        "dx": "Urinary tract infection",
        "co": "Burning while passing urine and frequent urination for 3 days, low back pain",
        "hx": "Similar episode one year ago. Married, two children.",
        "oe": "Temp 100.2 F, suprapubic tenderness, no renal angle tenderness",
        "course": "Urine culture grew E. coli sensitive to nitrofurantoin. Fever settled in 48 hours.",
        "meds": [("Cap. Nitrofurantoin", "100 mg", "1-0-1", "7 days"),
                 ("Syp. Alkasol", "10 ml", "1-1-1", "5 days"),
                 ("Tab. Paracetamol", "650 mg", "SOS", "for fever")],
        "advice": "Drink 3 litres of water a day, complete the full course of antibiotics",
        "labs": [("Urine pus cells", "25 - 30", "/hpf", "0 - 5"),
                 ("Urine RBC", "4 - 6", "/hpf", "0 - 2"),
                 ("Total WBC count", "11,800", "cells/cumm", "4,000 - 11,000"),
                 ("Serum creatinine", "0.8", "mg/dL", "0.6 - 1.2")],
        "rx": ["Cap. Niftran 100 mg 1-0-1 x 7 days", "Syp. Alkasol 10 ml TDS x 5 days",
               "Tab. Dolo 650 SOS"],
    },
]


@dataclass
class Person:
    name: str
    sex: str
    age: int
    phone: str
    house: str
    street: str
    area: str
    city: str
    state: str
    pin: str
    relative: str
    relation: str  # "S/O", "D/O", "W/O"
    uhid: str
    ip_no: str

    @property
    def address(self) -> str:
        return f"{self.house}, {self.street}, {self.area}, {self.city} - {self.pin}"

    @property
    def details(self) -> list[str]:
        return [self.name, self.phone, self.uhid, self.pin, f"{self.house}, {self.street}"]


def make_person(rng: random.Random) -> Person:
    sex = rng.choice("MF")
    first = rng.choice(FIRST_M if sex == "M" else FIRST_F)
    last = rng.choice(LAST)
    city, state, pin_prefix = rng.choice(PLACES)
    rel_first = rng.choice(FIRST_M)
    married = sex == "F" and rng.random() < 0.5
    return Person(
        name=f"{first} {last}",
        sex=sex,
        age=rng.randint(19, 78),
        phone=f"+91 {rng.choice('6789')}{rng.randint(100000000, 999999999)}",
        house=f"No. {rng.randint(1, 480)}",
        street=rng.choice(STREETS),
        area=rng.choice(AREAS),
        city=city,
        state=state,
        pin=f"{pin_prefix}{rng.randint(10, 99)}",
        relative=f"{rel_first} {last}",
        relation="W/O" if married else ("S/O" if sex == "M" else "D/O"),
        uhid=f"{rng.choice(['UH', 'MR', 'SL', 'GV'])}{rng.randint(2019, 2025)}{rng.randint(10000, 99999)}",
        ip_no=f"IP/{rng.randint(2023, 2025)}/{rng.randint(1000, 9999)}",
    )


def doctor(rng: random.Random) -> str:
    return f"Dr. {rng.choice(FIRST_M + FIRST_F)} {rng.choice(LAST)}"


def date(rng: random.Random, year: int | None = None) -> str:
    return f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{year or rng.randint(2023, 2025)}"


def hospital_lines(name: str, rng: random.Random) -> list[str]:
    city, _, pin_prefix = rng.choice(PLACES)
    return [name, f"{rng.randint(10, 300)}, {rng.choice(STREETS)}, {city} - {pin_prefix}{rng.randint(10, 99)}",
            f"Ph: 0{rng.randint(20, 89)}-{rng.randint(2000000, 9999999)} | Emergency: 1800-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"]


# ------------------------------------------------------------------ blocks -> HTML and gold text


@dataclass
class Block:
    kind: str  # center | kv | para | table | right | text | rule | title
    data: object
    zone: str = "body"  # header | body | footer (header/footer repeat across pages)

    def lines(self) -> list[str]:
        if self.kind in ("center", "right", "text", "title"):
            return [str(self.data)]
        if self.kind == "kv":
            return ["  ".join(f"{k}: {v}" for k, v in self.data)]  # type: ignore[attr-defined]
        if self.kind == "para":
            label, text = self.data  # type: ignore[misc]
            return [f"{label}: {text}" if label else text]
        if self.kind == "table":
            return [" ".join(row) for row in self.data]  # type: ignore[attr-defined]
        return []

    def html(self) -> str:
        e = html.escape
        if self.kind == "center":
            return f'<p class="c">{e(str(self.data))}</p>'
        if self.kind == "title":
            return f'<p class="t">{e(str(self.data))}</p>'
        if self.kind == "right":
            return f'<p class="r">{e(str(self.data))}</p>'
        if self.kind == "text":
            return f"<p>{e(str(self.data))}</p>"
        if self.kind == "kv":
            pairs = "&nbsp;&nbsp;&nbsp;&nbsp;".join(
                f"<b>{e(k)}:</b> {e(v)}" for k, v in self.data  # type: ignore[attr-defined]
            )
            return f'<p class="kv">{pairs}</p>'
        if self.kind == "para":
            label, text = self.data  # type: ignore[misc]
            return f"<p><b>{e(label)}:</b> {e(text)}</p>" if label else f"<p>{e(text)}</p>"
        if self.kind == "table":
            rows = self.data  # type: ignore[assignment]
            head = "".join(f"<th>{e(c)}</th>" for c in rows[0])
            body = "".join(
                "<tr>" + "".join(f"<td>{e(c)}</td>" for c in r) + "</tr>" for r in rows[1:]
            )
            return f"<table><tr>{head}</tr>{body}</table>"
        if self.kind == "rule":
            return "<hr/>"
        raise ValueError(self.kind)


@dataclass
class Page:
    blocks: list[Block]
    route: str = "text"  # what the pipeline should do: text | vision
    image: Image.Image | None = None  # for drawn pages (handwriting, forms)
    gold_lines: list[str] | None = None  # for drawn pages

    def lines(self, zone: str | None = None) -> list[str]:
        if self.gold_lines is not None and zone is None:
            return self.gold_lines
        return [ln for b in self.blocks if zone is None or b.zone == zone for ln in b.lines()]


def css(rng: random.Random) -> str:
    family = rng.choice(["sans-serif", "serif"])
    size = rng.choice([9.5, 10, 10.5, 11])
    return (
        f"body {{font-family:{family}; font-size:{size}pt}} p {{margin:2pt 0}}"
        ".c {text-align:center} .r {text-align:right} .kv {margin:1pt 0}"
        f".t {{text-align:center; font-weight:bold; font-size:{size + 2}pt; margin:6pt 0}}"
        "table {border-collapse:collapse; width:100%; margin:4pt 0}"
        "td, th {border:0.6pt solid #444; padding:2pt 4pt; text-align:left}"
        "th {background:#e8e8e8}"
    )


def render_pdf(pages: list[Page], rng: random.Random) -> pymupdf.Document:
    """Electronic pages (blocks with header/body/footer zones) to a PDF with a text layer."""
    doc = pymupdf.open()
    style = css(rng)
    for page in pages:
        p = doc.new_page(width=595, height=842)
        header = "".join(b.html() for b in page.blocks if b.zone == "header")
        body = "".join(b.html() for b in page.blocks if b.zone == "body")
        footer = "".join(b.html() for b in page.blocks if b.zone == "footer")
        if header:
            first_line = header.replace('<p class="c">', '<p class="c" style="font-size:150%;font-weight:bold">', 1)
            spare, _ = p.insert_htmlbox(pymupdf.Rect(45, 36, 550, 210), first_line + "<hr/>", css=style)
            top = 210 - spare + 6 if spare >= 0 else 210
        else:
            top = 40
        bottom = 842 - 70 if footer else 842 - 40
        p.insert_htmlbox(pymupdf.Rect(45, top, 550, bottom), body, css=style, scale_low=0)
        if footer:
            p.insert_htmlbox(pymupdf.Rect(45, 842 - 62, 550, 842 - 20), "<hr/>" + footer, css=style)
    return doc


# ------------------------------------------------------------------ degrade: scans and photos


def degrade(img: Image.Image, rng: random.Random, kind: str) -> Image.Image:
    nrng = np.random.default_rng(rng.randint(0, 2**31))
    img = img.convert("RGB")
    if kind == "photo":
        tint = np.array([rng.randint(238, 252), rng.randint(232, 246), rng.randint(215, 236)])
        arr = np.asarray(img).astype(np.float32) * (tint / 255.0)
        # uneven light: a shadow from one side
        h, w = arr.shape[:2]
        side = rng.choice(["x", "y"])
        ramp = np.linspace(1.0, rng.uniform(0.62, 0.85), w if side == "x" else h)
        if rng.random() < 0.5:
            ramp = ramp[::-1]
        arr *= ramp[None, :, None] if side == "x" else ramp[:, None, None]
        img = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        # perspective: the phone was not held flat
        w, h = img.size
        d = [rng.uniform(0, 0.035) * w for _ in range(4)]
        img = img.transform(
            (w, h), Image.Transform.QUAD,
            (d[0], d[1], d[2], h - d[3], w - d[1], h - d[0], w - d[3], d[2]),
            resample=Image.Resampling.BICUBIC, fillcolor=(90, 80, 70),
        )
        angle = rng.uniform(-3, 3)
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(90, 80, 70))
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.1)))
        sigma = rng.uniform(4, 10)
    else:  # scan
        if rng.random() < 0.7:
            img = img.convert("L").convert("RGB")
        angle = rng.uniform(-1.5, 1.5)
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(255, 255, 255))
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.8)))
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
        sigma = rng.uniform(3, 8)
    arr = np.asarray(img).astype(np.float32)
    arr += nrng.normal(0, sigma, arr.shape)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def jpeg(img: Image.Image, rng: random.Random) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.randint(58, 78))
    return buf.getvalue()


def rasterise(doc: pymupdf.Document, index: int, rng: random.Random) -> Image.Image:
    pix = doc[index].get_pixmap(dpi=rng.randint(160, 200))
    return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")


# ------------------------------------------------------------------ drawn pages (handwriting)


class Fonts:
    def __init__(self, cache: Path) -> None:
        self.cache = cache
        cache.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        rel = {**PRINT_FONTS, **HAND_FONTS}[name]
        target = self.cache / f"{name}.ttf"
        if not target.exists():
            print(f"  downloading font {name}")
            urllib.request.urlretrieve(FONT_BASE + rel, target)
        return target

    def get(self, name: str, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(self.path(name)), size)


@dataclass
class Canvas:
    img: Image.Image
    fonts: Fonts
    rng: random.Random
    y: float = 60
    margin: int = 70
    lines: list[str] = field(default_factory=list)

    @property
    def draw(self) -> ImageDraw.ImageDraw:
        return ImageDraw.Draw(self.img)

    def printed(self, text: str, size: int = 26, bold: bool = False, align: str = "left",
                gold: bool = True) -> None:
        font = self.fonts.get("Lato-Bold" if bold else "Lato-Regular", size)
        width = self.img.width - 2 * self.margin
        for line in wrap(text, font, width):
            w = font.getlength(line)
            x = {"left": self.margin, "center": (self.img.width - w) / 2,
                 "right": self.img.width - self.margin - w}[align]
            self.draw.text((x, self.y), line, font=font, fill=(20, 20, 20))
            self.y += size * 1.45
        if gold:
            self.lines.append(text)

    def rule(self) -> None:
        self.draw.line((self.margin, self.y + 4, self.img.width - self.margin, self.y + 4),
                       fill=(60, 60, 60), width=2)
        self.y += 18

    def hand(self, text: str, font: ImageFont.FreeTypeFont, ink: tuple[int, int, int],
             x: float | None = None, label: str | None = None, label_size: int = 26) -> None:
        """A handwritten line (wrapped), optionally after a printed label on the same line."""
        start = self.margin if x is None else x
        if label:
            lf = self.fonts.get("Lato-Regular", label_size)
            self.draw.text((start, self.y + font.size * 0.25), label, font=lf, fill=(20, 20, 20))
            start += lf.getlength(label) + 14
        # handwriting is drawn with wider, uneven spaces than wrap() measures: leave room
        width = (self.img.width - self.margin - start) * 0.8
        for i, line in enumerate(wrap(text, font, width)):
            strip = hand_strip(line, font, ink, self.rng)
            self.img.paste(strip, (int(start + self.rng.uniform(-6, 6)), int(self.y - font.size * 0.35)), strip)
            self.y += font.size * self.rng.uniform(1.25, 1.5)
            if i == 0 and label:
                start = self.margin
                width = (self.img.width - 2 * self.margin) * 0.8
        self.lines.append(f"{label} {text}" if label else text)

    def signature(self, x: float | None = None) -> None:
        sign_scribble(self.img, (x or self.img.width - self.margin - 260, self.y), self.rng)
        self.y += 90


def wrap(text: str, font: ImageFont.FreeTypeFont, width: float) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if font.getlength(trial) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def hand_strip(text: str, font: ImageFont.FreeTypeFont, ink: tuple[int, int, int],
               rng: random.Random) -> Image.Image:
    """One line of handwriting: words on a wobbling baseline, the whole line slightly tilted."""
    space = font.getlength(" ") * rng.uniform(1.0, 1.6)
    width = int(font.getlength(text) + space * (text.count(" ") + 2) + 60)
    height = int(font.size * 2.2)
    strip = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(strip)
    x = 20.0
    drift = 0.0
    for word in text.split(" "):
        drift += rng.uniform(-0.04, 0.04) * font.size
        drift = max(-0.15 * font.size, min(0.15 * font.size, drift))
        alpha = rng.randint(200, 255)
        d.text((x, font.size * 0.45 + drift), word, font=font, fill=(*ink, alpha))
        x += font.getlength(word) + space
    strip = strip.crop((0, 0, int(x) + 20, height))
    return strip.rotate(rng.uniform(-1.6, 1.6), resample=Image.Resampling.BICUBIC, expand=True)


def sign_scribble(img: Image.Image, at: tuple[float, float], rng: random.Random) -> None:
    d = ImageDraw.Draw(img)
    x0, y0 = at
    pts = []
    for i in range(60):
        t = i / 59
        pts.append((x0 + t * 240 + 18 * math.sin(t * rng.uniform(9, 16)),
                    y0 + 35 + 22 * math.sin(t * rng.uniform(5, 11) + rng.uniform(0, 3))))
    d.line(pts, fill=(25, 35, 110), width=3, joint="curve")


def ink(rng: random.Random) -> tuple[int, int, int]:
    return rng.choice([(22, 32, 120), (18, 18, 22), (35, 45, 140)])


def hand_font(fonts: Fonts, rng: random.Random, base: int = 46) -> ImageFont.FreeTypeFont:
    name = rng.choice(list(HAND_FONTS))
    return fonts.get(name, int(base * HAND_SIZE.get(name, 1.0) * rng.uniform(0.92, 1.08)))


# ------------------------------------------------------------------ document builders


def letterhead(hosp: list[str]) -> list[Block]:
    return [Block("center", ln, "header") for ln in hosp]


def banner_discharge(p: Person, cons: str, doa: str, dod: str, rng: random.Random) -> list[Block]:
    return [
        Block("kv", [("Name", p.name), ("Age/Sex", f"{p.age}/{p.sex}")], "header"),
        Block("kv", [("UHID", p.uhid), ("IP No", p.ip_no), ("Ward/Bed", f"{rng.choice(['GW', 'MW', 'SW'])}-{rng.randint(1, 40)}")], "header"),
        Block("kv", [("Admitted", doa), ("Discharged", dod), ("Consultant", cons)], "header"),
    ]


def discharge_pages(p: Person, rng: random.Random, hosp: list[str]) -> tuple[list[Page], list[str]]:
    c = rng.choice(CONDITIONS)
    cons = doctor(rng)
    year = rng.randint(2023, 2025)
    doa, dod = date(rng, year), date(rng, year)
    head = letterhead(hosp) + banner_discharge(p, cons, doa, dod, rng)
    contact = Block("kv", [("Address", p.address), ("Phone", p.phone)])
    page1 = head + [
        Block("title", "DISCHARGE SUMMARY"),
        contact,
        Block("para", ("Diagnosis", c["dx"])),
        Block("para", ("Presenting complaints", c["co"])),
        Block("para", ("Past history", c["hx"])),
        Block("para", ("On examination", c["oe"])),
        Block("text", "Investigations:"),
        Block("table", [("Test", "Result", "Unit", "Reference range"), *c["labs"][:4]]),
    ]
    page2 = head + [
        Block("para", ("Course in hospital", c["course"])),
        Block("text", "Medications on discharge:"),
        Block("table", [("Medicine", "Dose", "Frequency", "Duration"), *c["meds"]]),
        Block("para", ("Advice", c["advice"])),
        Block("para", ("Follow-up", f"Review in the OPD on {date(rng, year)} or earlier if needed")),
        Block("right", f"{cons}, {rng.choice(QUALS)}"),
        Block("right", "Consultant Physician"),
    ]
    pages = []
    for i, blocks in enumerate([page1, page2], 1):
        foot = [Block("text", f"Page {i} of 2", "footer"),
                Block("text", "This is a computer generated document.", "footer")]
        pages.append(Page(blocks + foot))
    return pages, [*p.details, p.ip_no, cons]


def lab_page(p: Person, rng: random.Random, hosp: list[str]) -> tuple[Page, list[str]]:
    c = rng.choice(CONDITIONS)
    ref = doctor(rng)
    path = doctor(rng)
    lab_no = f"L{rng.randint(100000, 999999)}"
    d1 = date(rng)
    blocks = letterhead(hosp) + [
        Block("center", "Department of Laboratory Medicine", "header"),
        Block("kv", [("Patient", p.name), ("Age/Sex", f"{p.age}/{p.sex}")]),
        Block("kv", [("UHID", p.uhid), ("Lab No", lab_no)]),
        Block("kv", [("Referred by", ref), ("Sample date", d1)]),
        Block("kv", [("Mobile", p.phone)]),
        Block("title", "LABORATORY REPORT"),
        Block("table", [("Test", "Result", "Unit", "Reference range"), *c["labs"]]),
        Block("text", "Values outside the reference range should be correlated clinically."),
        Block("right", f"Checked by: {path}, MD (Pathology)"),
        Block("center", "*** End of report ***", "footer"),
    ]
    return Page(blocks), [p.name, p.uhid, p.phone, lab_no, ref, path]


def prescription_page(p: Person, rng: random.Random, hosp: list[str], fonts: Fonts) -> tuple[Page, list[str]]:
    c = rng.choice(CONDITIONS)
    doc_name = doctor(rng)
    reg = f"KMC {rng.randint(10000, 99999)}"
    img = Image.new("RGB", (1165, 1654), (252, 252, 248))
    cv = Canvas(img, fonts, rng, y=55, margin=70)
    cv.printed(hosp[0], 40, bold=True, align="center")
    cv.printed(f"{doc_name}, {rng.choice(QUALS)}", 26, align="center")
    cv.printed(f"Reg. No. {reg}", 22, align="center")
    cv.printed(hosp[2], 22, align="center")
    cv.rule()
    font = hand_font(fonts, rng)
    colour = ink(rng)
    cv.y += 10
    cv.hand(p.name, font, colour, label="Name:")
    cv.hand(f"{p.age} / {p.sex}", font, colour, label="Age/Sex:")
    cv.hand(date(rng), font, colour, label="Date:")
    cv.y += 10
    cv.hand(f"C/O {c['co'].split(',')[0].lower()}", font, colour)
    cv.hand(f"O/E {c['oe'].split(',')[0]}", font, colour)
    cv.hand(f"Dx: {c['dx'].split(',')[0]}", font, colour)
    cv.y += 6
    cv.printed("Rx", 40, bold=True)
    for line in c["rx"]:
        cv.hand(line, font, colour, x=cv.margin + 40)
    cv.y += 10
    cv.hand(f"Adv: {c['advice'].split(',')[0]}", font, colour)
    cv.hand(f"Review after {rng.choice([3, 5, 7])} days", font, colour)
    cv.y = max(cv.y, 1654 - 230)
    cv.signature()
    cv.printed(doc_name, 22, align="right")
    return Page([], route="vision", image=img, gold_lines=cv.lines), [p.name, doc_name, reg]


def consent_page(p: Person, rng: random.Random, hosp: list[str], fonts: Fonts) -> tuple[Page, list[str]]:
    surgeon = doctor(rng)
    img = Image.new("RGB", (1240, 1754), (253, 253, 251))
    cv = Canvas(img, fonts, rng, y=60, margin=80)
    for i, line in enumerate(hosp):
        cv.printed(line, 38 if i == 0 else 22, bold=i == 0, align="center")
    cv.rule()
    cv.printed("CONSENT FOR PROCEDURE / ANAESTHESIA", 30, bold=True, align="center")
    cv.y += 16
    font = hand_font(fonts, rng, 44)
    colour = ink(rng)
    proc = rng.choice(["Laparoscopic appendicectomy", "Upper GI endoscopy", "Central line insertion",
                       "Incision and drainage of abscess", "Cataract surgery, right eye"])
    fields = [("Patient name:", p.name), ("Age / Sex:", f"{p.age} / {p.sex}"), ("UHID:", p.uhid),
              ("Address:", p.address), ("Procedure:", proc), ("Surgeon:", surgeon)]
    for label, value in fields:
        cv.hand(value, font, colour, label=label)
        cv.y += 8
    cv.y += 8
    cv.printed("I, the undersigned, give my consent to the above procedure and to the anaesthesia "
               "needed for it. Its nature, risks and alternatives have been explained to me in a "
               "language I understand.", 24)
    cv.y += 16
    for label, value in [("Name of guardian:", p.relative),
                         ("Relationship:", rng.choice(["Father", "Husband", "Brother", "Son"])),
                         ("Phone:", p.phone), ("Date:", date(rng))]:
        cv.hand(value, font, colour, label=label)
        cv.y += 8
    cv.printed("Signature of patient / guardian:", 24, gold=True)
    cv.signature(x=cv.margin + 40)
    return Page([], route="vision", image=img, gold_lines=cv.lines), [*p.details, p.relative, surgeon]


def bank_page(p: Person, rng: random.Random) -> tuple[Page, list[str]]:
    bank, code = rng.choice(BANKS)
    city, _, pin = rng.choice(PLACES)
    ifsc = f"{code}0{rng.randint(100000, 999999)}"
    acct = str(rng.randint(10**12, 10**13 - 1))
    cust = f"CIF{rng.randint(1000000, 9999999)}"
    month = rng.randint(1, 12)
    year = rng.randint(2023, 2025)
    bal = rng.randint(15000, 240000) + rng.randint(0, 99) / 100
    rows = [("Date", "Description", "Debit", "Credit", "Balance")]
    payees = [f"{rng.choice(FIRST_M + FIRST_F)} {rng.choice(LAST)}" for _ in range(4)]
    for day in sorted(rng.sample(range(1, 29), 11)):
        kind = rng.random()
        if kind < 0.15:
            desc, cr = f"NEFT-{rng.randint(10**9, 10**10)}-{rng.choice(['INFOSYS', 'TCS', 'WIPRO', 'BOSCH'])} SALARY", rng.randint(40000, 120000)
            dr = 0
        elif kind < 0.45:
            desc, dr, cr = f"UPI/{rng.choice(payees).upper()}/{rng.randint(10**11, 10**12)}", rng.randint(100, 9000), 0
        elif kind < 0.6:
            desc, dr, cr = f"ATM WDL {city.upper()} {rng.randint(1000, 9999)}", rng.choice([2000, 5000, 10000]), 0
        elif kind < 0.75:
            desc, dr, cr = f"POS {rng.choice(['APOLLO PHARMACY', 'MORE SUPERMARKET', 'HP PETROL', 'DMART'])}", rng.randint(150, 6000), 0
        elif kind < 0.88:
            desc, dr, cr = f"IMPS/{rng.choice(payees).upper()}/{rng.randint(10**11, 10**12)}", 0, rng.randint(500, 25000)
        else:
            desc, dr, cr = f"EMI LOAN A/C {rng.randint(10**8, 10**9)}", rng.randint(4000, 22000), 0
        bal = bal - dr + cr
        rows.append((f"{day:02d}/{month:02d}/{year}", desc, f"{dr:,.2f}" if dr else "",
                     f"{cr:,.2f}" if cr else "", f"{bal:,.2f}"))
    blocks = [
        Block("center", bank, "header"),
        Block("center", f"{rng.choice(AREAS)} Branch, {city} - {pin}{rng.randint(10, 99)} | IFSC: {ifsc}", "header"),
        Block("title", "STATEMENT OF ACCOUNT"),
        Block("kv", [("Account holder", p.name)]),
        Block("kv", [("Address", p.address)]),
        Block("kv", [("Account no", acct), ("Customer ID", cust)]),
        Block("kv", [("Mobile", p.phone), ("Statement period", f"01/{month:02d}/{year} to 28/{month:02d}/{year}")]),
        Block("table", rows),
        Block("text", "This is a system generated statement and does not need a signature."),
    ]
    used = [n for n in payees if any(n.upper() in row[1] for row in rows[1:])]
    return Page(blocks), [p.name, p.phone, acct, ifsc, cust, f"{p.house}, {p.street}", p.pin, *used]


def affidavit_page(p: Person, rng: random.Random, hosp_name: str) -> tuple[Page, list[str]]:
    year = rng.randint(2023, 2025)
    notary = doctor(rng).replace("Dr.", "Adv.")
    who = {"S/O": "son", "D/O": "daughter", "W/O": "wife"}[p.relation]
    blocks = [
        Block("title", "AFFIDAVIT"),
        Block("text", f"I, {p.name}, aged {p.age} years, {who} of {p.relative}, residing at {p.address}, "
                      "do hereby solemnly affirm and state as follows:"),
        Block("text", f"1. That I was admitted to {hosp_name} on {date(rng, year)} under UHID {p.uhid} "
                      "and discharged after treatment."),
        Block("text", "2. That the original bills and discharge papers of the said admission have been "
                      "lost and could not be traced despite a diligent search."),
        Block("text", f"3. That I am making this affidavit to claim reimbursement under policy number "
                      f"POL{rng.randint(10**7, 10**8)} and for no other purpose."),
        Block("text", "4. That the contents of this affidavit are true and correct to the best of my "
                      "knowledge and belief, and nothing material has been concealed."),
        Block("kv", [("Place", p.city), ("Date", date(rng, year))]),
        Block("right", "DEPONENT"),
        Block("title", "VERIFICATION"),
        Block("text", f"Verified at {p.city} on this day that the contents of the above affidavit are "
                      "true and correct."),
        Block("right", f"Identified by: {notary}, Notary"),
    ]
    return Page(blocks, route="vision"), [p.name, p.relative, f"{p.house}, {p.street}", p.pin, p.uhid, notary]


# ------------------------------------------------------------------ one document of each type


@dataclass
class Built:
    file: bytes
    suffix: str
    pages: list[Page]
    details: list[str]


def build(doc_type: str, rng: random.Random, fonts: Fonts) -> Built:
    person = make_person(rng)
    hosp = hospital_lines(rng.choice(HOSPITALS), rng)

    if doc_type in ("discharge_e", "discharge_scan"):
        pages, details = discharge_pages(person, rng, hosp)
        pdf = render_pdf(pages, rng)
        if doc_type == "discharge_e":
            return Built(pdf_bytes(pdf), ".pdf", pages, details)
        out = pymupdf.open()
        for i, page in enumerate(pages):
            page.route = "vision"
            add_image_page(out, degrade(rasterise(pdf, i, rng), rng, "scan"), rng)
        return Built(pdf_bytes(out), ".pdf", pages, details)

    if doc_type in ("lab_e", "lab_photo"):
        page, details = lab_page(person, rng, hosp)
        pdf = render_pdf([page], rng)
        if doc_type == "lab_e":
            return Built(pdf_bytes(pdf), ".pdf", [page], details)
        page.route = "vision"
        return Built(jpeg(degrade(rasterise(pdf, 0, rng), rng, "photo"), rng), ".jpg", [page], details)

    if doc_type == "prescription_hw":
        page, details = prescription_page(person, rng, hosp, fonts)
        return Built(jpeg(degrade(page.image, rng, "photo"), rng), ".jpg", [page], details)  # type: ignore[arg-type]

    if doc_type == "consent_form_hw":
        page, details = consent_page(person, rng, hosp, fonts)
        return Built(jpeg(degrade(page.image, rng, "scan"), rng), ".jpg", [page], details)  # type: ignore[arg-type]

    if doc_type == "mixed_file":
        lab, d1 = lab_page(person, rng, hosp)
        form, d2 = consent_page(person, rng, hosp, fonts)
        rx, d3 = prescription_page(person, rng, hosp, fonts)
        out = render_pdf([lab], rng)
        add_image_page(out, degrade(form.image, rng, "scan"), rng)  # type: ignore[arg-type]
        add_image_page(out, degrade(rx.image, rng, "photo"), rng)  # type: ignore[arg-type]
        return Built(pdf_bytes(out), ".pdf", [lab, form, rx], list(dict.fromkeys(d1 + d2 + d3)))

    if doc_type == "bank_statement_e":
        page, details = bank_page(person, rng)
        return Built(pdf_bytes(render_pdf([page], rng)), ".pdf", [page], details)

    if doc_type == "affidavit_scan":
        page, details = affidavit_page(person, rng, hosp[0])
        pdf = render_pdf([page], rng)
        img = rasterise(pdf, 0, rng)
        sign_scribble(img, (img.width * 0.62, img.height * 0.52), rng)
        out = pymupdf.open()
        add_image_page(out, degrade(img, rng, "scan"), rng)
        return Built(pdf_bytes(out), ".pdf", [page], details)

    raise ValueError(doc_type)


def pdf_bytes(doc: pymupdf.Document) -> bytes:
    doc.subset_fonts()
    return doc.tobytes(garbage=4, deflate=True)


def add_image_page(doc: pymupdf.Document, img: Image.Image, rng: random.Random) -> None:
    width = 595
    height = width * img.height / img.width
    page = doc.new_page(width=width, height=height)
    page.insert_image(page.rect, stream=jpeg(img, rng))


# ------------------------------------------------------------------ gold


def gold_record(doc_id: str, doc_type: str, built: Built, file: str) -> dict:
    """Gold text with and without the repeated header/footer lines, and which lines those are.

    A header/footer line is "repeated" when the same line (the pipeline's own notion of same,
    headers._key) is a header/footer line on at least half the pages (and 2); the copies after
    its first page are what a correct strip removes."""
    pages = built.pages
    n = len(pages)
    full = ["\n".join(p.lines()) for p in pages]
    zone_keys: list[set[str]] = []
    for p in pages:
        keys = set()
        for ln in p.lines("header") + p.lines("footer"):
            k = _key(ln)
            if k:
                keys.add(k)
        if p.gold_lines is not None:  # drawn pages: the letterhead is in their first lines
            keys |= {k for k in (_key(ln) for ln in p.gold_lines[:TOP]) if k}
        zone_keys.append(keys)
    needed = max(2, math.ceil(n / 2))
    counts: dict[str, int] = {}
    for keys in zone_keys:
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
    repeated = {k for k, c in counts.items() if c >= needed}
    seen: set[str] = set()
    stripped, removals = [], []
    for i, p in enumerate(pages):
        kept = []
        for ln in p.lines():
            k = _key(ln)
            if k in repeated and k in seen and k in zone_keys[i]:
                removals.append({"page": i, "line": ln})
            else:
                kept.append(ln)
        seen |= zone_keys[i] & repeated
        stripped.append("\n".join(kept))
    domain, _, what = TYPES[doc_type]
    return {
        "id": doc_id,
        "file": file,
        "type": doc_type,
        "domain": domain,
        "about": what,
        "routes": [p.route for p in pages],
        "gold": "\n\n".join(stripped),
        "gold_pages": full,
        "removals": removals,
        "details": list(dict.fromkeys(d for d in built.details if d)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/ocr_test")
    ap.add_argument("--per-type", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--types", default=",".join(TYPES), help="comma-separated subset")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    fonts = Fonts(out / ".fonts")
    records = []
    for doc_type in args.types.split(","):
        for k in range(1, args.per_type + 1):
            doc_id = f"{doc_type}_{k:02d}"
            rng = random.Random(f"{args.seed}-{doc_id}")
            built = build(doc_type, rng, fonts)
            file = f"docs/{doc_id}{built.suffix}"
            (out / file).write_bytes(built.file)
            records.append(gold_record(doc_id, doc_type, built, file))
            print(f"{doc_id:22s} {len(built.pages)} page(s) {len(built.file) / 1e3:7.0f} kB")
    with open(out / "gold.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum((out / r["file"]).stat().st_size for r in records)
    print(f"{len(records)} documents, {total / 1e6:.1f} MB -> {out}")


if __name__ == "__main__":
    main()

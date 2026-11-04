"""Tests for multimodal ingestion: PDFs, images/handwriting, HL7 v2, FHIR, XML, and ECG telemetry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sanitise.ingest import extract_text_multimodal, ingest_file
from sanitise.ingest.electronic_parser import parse_electronic
from sanitise.ingest.image_parser import parse_image
from sanitise.ingest.pdf_parser import parse_pdf
from sanitise.loader import Document, make_document


# ---------------------------------------------------------------------------
# 1. HL7 v2 Parser Tests
# ---------------------------------------------------------------------------


def test_hl7_v2_parsing():
    sample_hl7 = (
        "MSH|^~\\&|HOSPITAL|EMR|SANITISER|APP|20261104120000||ADT^A08|MSG001|P|2.5\n"
        "PID|1||MRN123456^^^HOSP||DOE^JOHN^A||19750820|M|||123 MAIN ST^^SPRINGFIELD^IL^62701||555-123-4567\n"
        "PV1|1|O|||||1234^SMITH^ALICE^DR||||||||||||V987654|||||||||||||||||||||||||20261101\n"
        "DG1|1||I10^Essential Hypertension^I10||A\n"
        "DG1|2||E11^Type 2 Diabetes Mellitus^I10||A\n"
        "OBX|1|NM|883-9^Systolic Blood Pressure||145|mmHg|||||F\n"
        "OBX|2|NM|8462-4^Diastolic Blood Pressure||92|mmHg|||||F\n"
        "NTE|1||Patient reports dizziness when standing up."
    )

    text, fmt = parse_electronic(sample_hl7)
    assert fmt == "hl7"
    assert "John Doe" in text
    assert "1975-08-20" in text
    assert "Springfield" in text
    assert "Dr. Alice Smith" in text
    assert "Essential Hypertension" in text
    assert "Type 2 Diabetes Mellitus" in text
    assert "Systolic Blood Pressure: 145 mmHg" in text
    assert "Patient reports dizziness when standing up" in text


# ---------------------------------------------------------------------------
# 2. FHIR JSON Parser Tests
# ---------------------------------------------------------------------------


def test_fhir_patient_resource():
    patient_fhir = {
        "resourceType": "Patient",
        "id": "pat-001",
        "name": [{"family": "Sharma", "given": ["Rohit"]}],
        "gender": "male",
        "birthDate": "1988-04-30",
        "address": [{"text": "MG Road, Bengaluru"}],
    }

    text, fmt = parse_electronic(patient_fhir)
    assert fmt == "fhir"
    assert "Rohit Sharma" in text
    assert "male" in text
    assert "1988-04-30" in text
    assert "MG Road, Bengaluru" in text


def test_fhir_bundle():
    bundle_fhir = {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": [
            {
                "resource": {
                    "resourceType": "Patient",
                    "name": [{"family": "Iyer", "given": ["Ananya"]}],
                    "gender": "female",
                    "birthDate": "1994-11-15",
                }
            },
            {
                "resource": {
                    "resourceType": "Condition",
                    "code": {"text": "Migraine with aura"},
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                }
            },
            {
                "resource": {
                    "resourceType": "Observation",
                    "code": {"text": "Heart rate"},
                    "valueQuantity": {"value": 72, "unit": "bpm"},
                }
            },
        ],
    }

    text, fmt = parse_electronic(bundle_fhir)
    assert fmt == "fhir"
    assert "Ananya Iyer" in text
    assert "Migraine with aura" in text
    assert "Heart rate = 72 bpm" in text


# ---------------------------------------------------------------------------
# 3. Clinical XML / CDA Parser Tests
# ---------------------------------------------------------------------------


def test_clinical_cda_xml():
    xml_data = """<?xml version="1.0" encoding="UTF-8"?>
    <ClinicalDocument xmlns="urn:hl7-org:v3">
        <title>Discharge Summary</title>
        <patientRole>
            <patient>
                <name>
                    <given>Suresh</given>
                    <family>Nair</family>
                </name>
                <birthTime value="19650210"/>
            </patient>
        </patientRole>
        <component>
            <structuredBody>
                <section>
                    <title>Hospital Course</title>
                    <text>Patient was admitted for acute chest pain and underwent cardiac catheterisation.</text>
                </section>
                <section>
                    <title>Discharge Medications</title>
                    <text>Aspirin 75mg daily, Atorvastatin 40mg at bedtime.</text>
                </section>
            </structuredBody>
        </component>
    </ClinicalDocument>
    """

    text, fmt = parse_electronic(xml_data)
    assert fmt == "clinical_xml"
    assert "Discharge Summary" in text
    assert "Suresh Nair" in text
    assert "Hospital Course" in text
    assert "acute chest pain" in text
    assert "Aspirin 75mg daily" in text


# ---------------------------------------------------------------------------
# 4. ECG Telemetry Tests
# ---------------------------------------------------------------------------


def test_ecg_telemetry_parser():
    ecg_report = (
        "ELECTROCARDIOGRAM (12-LEAD ECG)\n"
        "Ventricular Rate: 76 bpm\n"
        "PR Interval: 154 ms\n"
        "QRS Duration: 92 ms\n"
        "QT/QTc: 382/430 ms\n"
        "P-R-T Axes: 48 35 42\n"
        "Rhythm: Normal Sinus Rhythm\n"
        "Interpretation: Normal ECG. No acute ST segment elevation."
    )

    text, fmt = parse_electronic(ecg_report)
    assert fmt == "ecg_telemetry"
    assert "PR Interval: 154 ms" in text
    assert "QRS Duration: 92 ms" in text
    assert "Normal Sinus Rhythm" in text


# ---------------------------------------------------------------------------
# 5. PDF Parser Fallback Tests
# ---------------------------------------------------------------------------


def test_pdf_stream_fallback():
    # Minimal uncompressed PDF text stream
    raw_pdf_bytes = (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Length 50 >>\nstream\n"
        b"BT\n/F1 12 Tf\n(Patient: Ramesh Patel, Age: 45) Tj\nET\n"
        b"endstream\nendobj\n%%EOF"
    )

    extracted = parse_pdf(raw_pdf_bytes)
    assert "Patient: Ramesh Patel, Age: 45" in extracted


# ---------------------------------------------------------------------------
# 6. Image Parser Fallback Tests
# ---------------------------------------------------------------------------


def test_image_parser_fallback():
    # 1x1 transparent PNG bytes
    dummy_png = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01"
        b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    )

    result = parse_image(dummy_png, mime_type="image/png")
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 7. Unified Ingest Facade Tests
# ---------------------------------------------------------------------------


def test_ingest_hl7_file(tmp_path: Path):
    hl7_file = tmp_path / "case.hl7"
    hl7_file.write_text(
        "MSH|^~\\&|SYS|HOSP|||20261104||ORU^R01|101|P|2.3\n"
        "PID|1||98765||Kumar^Vijay||19820315|M|||Chennai\n"
        "DG1|1||J45^Asthma\n",
        encoding="utf-8",
    )

    doc = ingest_file(hl7_file)
    assert isinstance(doc, Document)
    assert "Vijay Kumar" in doc.text
    assert "Asthma" in doc.text
    assert doc.source_format == "hl7"
    assert len(doc.doc_id) == 16


def test_extract_multimodal_routing():
    text, fmt = extract_text_multimodal("MSH|^~\\&|A|B|||2026||ADT|1|P|2.3\nPID|1||1||Test")
    assert fmt == "hl7"

    text, fmt = extract_text_multimodal("<ClinicalDocument><title>Report</title></ClinicalDocument>")
    assert fmt == "clinical_xml"

    text, fmt = extract_text_multimodal('{"resourceType": "Patient", "gender": "female"}')
    assert fmt == "fhir"


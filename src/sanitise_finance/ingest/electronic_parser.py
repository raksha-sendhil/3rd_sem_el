"""Parser for Electronic Health Records (EHR), HL7 v2, FHIR, Clinical XML, and ECG Telemetry.

Translates structured electronic medical formats and biomedical telemetry data
into clean, structured clinical narrative text that can be processed by Agent 1
(Sensitive Information Detector).
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def parse_electronic(source: str | Path | bytes | dict[str, Any]) -> tuple[str, str]:
    """Parse electronic health records or telemetry signals into clinical text.

    Args:
        source: Text string, file path, raw bytes, or pre-parsed dict.

    Returns:
        tuple of (clinical_text, detected_format)
    """
    raw_str = ""
    if isinstance(source, dict):
        return _parse_fhir_or_json(source)
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_file():
            raw_str = p.read_text(encoding="utf-8-sig", errors="ignore")
        else:
            raw_str = str(source)
    elif isinstance(source, bytes):
        raw_str = source.decode("utf-8-sig", errors="ignore")
    else:
        raise TypeError(f"Unsupported source type: {type(source).__name__}")

    raw_str = raw_str.strip()
    if not raw_str:
        return "", "electronic"

    # 1. HL7 v2 Message (MSH|^~\&)
    if raw_str.startswith("MSH") or "\nMSH|" in raw_str:
        return _parse_hl7_v2(raw_str), "hl7"

    # 2. JSON format (FHIR, structured telemetry, or EHR export)
    if (raw_str.startswith("{") and raw_str.endswith("}")) or (
        raw_str.startswith("[") and raw_str.endswith("]")
    ):
        try:
            parsed_json = json.loads(raw_str)
            if isinstance(parsed_json, dict):
                return _parse_fhir_or_json(parsed_json)
            elif isinstance(parsed_json, list):
                # List of resources or records
                summaries = []
                for item in parsed_json:
                    if isinstance(item, dict):
                        text, _ = _parse_fhir_or_json(item)
                        if text:
                            summaries.append(text)
                return "\n\n---\n\n".join(summaries), "fhir"
        except json.JSONDecodeError:
            pass

    # 3. Clinical XML (CDA, CCDA, or HL7 v3 XML)
    if raw_str.startswith("<") and raw_str.endswith(">"):
        try:
            return _parse_clinical_xml(raw_str), "clinical_xml"
        except Exception as exc:
            log.warning("XML parsing failed: %s; falling back to text", exc)

    # 4. ECG Telemetry / Structured interval measurements
    if _is_ecg_telemetry(raw_str):
        return _parse_ecg_telemetry(raw_str), "ecg_telemetry"

    # Default fallback: return as-is
    return raw_str, "electronic"


# ---------------------------------------------------------------------------
# HL7 v2 Parser
# ---------------------------------------------------------------------------


def _proper(text: str) -> str:
    """HL7 senders write names in capitals ("DOE^JOHN"); "JOHN DOE" reads as "John Doe".
    Mixed-case text is left as it is."""
    return text.title() if text.isupper() else text


def _parse_hl7_v2(hl7_text: str) -> str:
    """Parse pipe-delimited HL7 v2 message into a narrative clinical report."""
    lines = [line.strip() for line in hl7_text.splitlines() if line.strip()]
    patient_info: dict[str, str] = {}
    diagnoses: list[str] = []
    observations: list[str] = []
    notes: list[str] = []
    attending = ""
    encounter_date = ""

    for line in lines:
        fields = line.split("|")
        seg = fields[0].upper()

        if seg == "PID":
            # PID-3: Patient ID, PID-5: Patient Name (Last^First^Middle), PID-7: DOB (YYYYMMDD), PID-8: Sex, PID-11: Address
            if len(fields) > 5 and fields[5]:
                name_parts = fields[5].split("^")  # XPN: Family^Given^Middle
                name = " ".join(filter(None, [name_parts[1] if len(name_parts) > 1 else "", name_parts[0]]))
                patient_info["Name"] = _proper(name.strip()) or fields[5]
            if len(fields) > 3 and fields[3]:
                patient_info["ID"] = fields[3]
            if len(fields) > 7 and fields[7]:
                dob_raw = fields[7]
                if len(dob_raw) == 8:
                    patient_info["DOB"] = f"{dob_raw[:4]}-{dob_raw[4:6]}-{dob_raw[6:]}"
                else:
                    patient_info["DOB"] = dob_raw
            if len(fields) > 8 and fields[8]:
                patient_info["Sex"] = fields[8]
            if len(fields) > 11 and fields[11]:
                # XAD: Street^Other^City^State^Zip; empty parts dropped, ALL-CAPS words softened
                parts = [p.strip() for p in fields[11].split("^") if p.strip()]
                patient_info["Address"] = ", ".join(p if len(p) <= 2 else _proper(p) for p in parts)
            if len(fields) > 13 and fields[13]:
                patient_info["Phone"] = fields[13].replace("^", " ").strip()

        elif seg == "PV1":
            # PV1-7: Attending Doctor, PV1-44: Admit Date
            if len(fields) > 7 and fields[7]:
                # XCN: ID^Family^Given^...; older senders leave out the ID (Family^Given)
                doc_parts = fields[7].split("^")
                if doc_parts[0].strip().isdigit() and len(doc_parts) > 1:
                    doc_parts = doc_parts[1:]
                given = doc_parts[1] if len(doc_parts) > 1 else ""
                attending = "Dr. " + _proper(" ".join(filter(None, [given, doc_parts[0]])))
            if len(fields) > 44 and fields[44]:
                admit_raw = fields[44][:8]
                if len(admit_raw) == 8:
                    encounter_date = f"{admit_raw[:4]}-{admit_raw[4:6]}-{admit_raw[6:]}"

        elif seg == "DG1":
            # DG1-3: Diagnosis Code ^ Description, DG1-6: Diagnosis Type
            if len(fields) > 3 and fields[3]:
                diag_parts = fields[3].split("^")
                desc = diag_parts[1] if len(diag_parts) > 1 and diag_parts[1] else diag_parts[0]
                diagnoses.append(desc)

        elif seg == "OBX":
            # OBX-3: Observation Identifier, OBX-5: Value, OBX-6: Units, OBX-8: Abnormal Flag
            obs_name = fields[3].split("^")[1] if len(fields) > 3 and "^" in fields[3] else (fields[3] if len(fields) > 3 else "")
            val = fields[5] if len(fields) > 5 else ""
            units = fields[6] if len(fields) > 6 else ""
            flag = f" [{fields[8]}]" if len(fields) > 8 and fields[8] and fields[8] not in ("N", "NORMAL") else ""
            if obs_name or val:
                observations.append(f"{obs_name}: {val} {units}".strip() + flag)

        elif seg == "NTE":
            # NTE-3: Comment / Clinical note
            if len(fields) > 3 and fields[3]:
                notes.append(fields[3])

    # Assemble narrative
    out: list[str] = ["CLINICAL HEALTH RECORD (HL7)"]
    if patient_info:
        p_str = ", ".join(f"{k}: {v}" for k, v in patient_info.items())
        out.append(f"Patient: {p_str}")
    if attending:
        out.append(f"Attending Physician: {attending}")
    if encounter_date:
        out.append(f"Encounter Date: {encounter_date}")
    if diagnoses:
        out.append("Diagnoses:\n" + "\n".join(f"- {d}" for d in diagnoses))
    if observations:
        out.append("Observations & Lab Results:\n" + "\n".join(f"- {o}" for o in observations))
    if notes:
        out.append("Clinical Notes:\n" + "\n".join(notes))

    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# FHIR JSON Parser
# ---------------------------------------------------------------------------


def _parse_fhir_or_json(data: dict[str, Any]) -> tuple[str, str]:
    """Parse a FHIR resource or FHIR Bundle into a clinical narrative."""
    res_type = data.get("resourceType")

    # If it's a FHIR bundle
    if res_type == "Bundle":
        entries = data.get("entry", [])
        sections: list[str] = ["FHIR CLINICAL BUNDLE"]
        for entry in entries:
            res = entry.get("resource", {})
            t, _ = _parse_fhir_or_json(res)
            if t:
                sections.append(t)
        return "\n\n".join(sections), "fhir"

    # If it's an individual FHIR resource
    if res_type == "Patient":
        name = ""
        names = data.get("name", [])
        if names:
            n0 = names[0]
            family = n0.get("family", "")
            given = " ".join(n0.get("given", []))
            name = f"{given} {family}".strip() or n0.get("text", "")
        gender = data.get("gender", "")
        dob = data.get("birthDate", "")
        addr_list = data.get("address", [])
        addr_str = addr_list[0].get("text") or addr_list[0].get("city", "") if addr_list else ""
        return (
            f"Patient Record:\nName: {name}\nGender: {gender}\nDate of Birth: {dob}\nAddress: {addr_str}".strip(),
            "fhir",
        )

    if res_type == "Condition":
        code = data.get("code", {}).get("text") or _extract_coding(data.get("code", {}))
        status = data.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "active")
        onset = data.get("onsetDateTime", "")
        return f"Condition: {code} (Status: {status}, Onset: {onset})".strip(), "fhir"

    if res_type == "Observation":
        name = data.get("code", {}).get("text") or _extract_coding(data.get("code", {}))
        val_q = data.get("valueQuantity", {})
        val = f"{val_q.get('value', '')} {val_q.get('unit', '')}".strip() or data.get("valueString", "")
        effective = data.get("effectiveDateTime", "")
        return f"Observation: {name} = {val} ({effective})".strip(), "fhir"

    if res_type == "MedicationRequest":
        med = data.get("medicationCodeableConcept", {}).get("text") or _extract_coding(
            data.get("medicationCodeableConcept", {})
        )
        dosage = ""
        dosages = data.get("dosageInstruction", [])
        if dosages:
            dosage = dosages[0].get("text", "")
        return f"Prescription: {med} - Instructions: {dosage}".strip(), "fhir"

    # Generic JSON telemetry or clinical payload
    if "telemetry" in data or "ecg" in data or "vitals" in data:
        return _format_dict_as_vitals(data), "telemetry"

    # If it's a standard dict, convert key-values to readable lines
    lines = [f"{k.capitalize()}: {v}" for k, v in data.items() if isinstance(v, (str, int, float, bool))]
    return "\n".join(lines), "electronic_json"


def _extract_coding(concept: dict[str, Any]) -> str:
    codings = concept.get("coding", [])
    if codings:
        return codings[0].get("display") or codings[0].get("code", "")
    return ""


def _format_dict_as_vitals(data: dict[str, Any]) -> str:
    lines = ["ELECTRONIC VITALS & TELEMETRY REPORT"]
    for k, v in data.items():
        if isinstance(v, dict):
            sub = ", ".join(f"{sk}: {sv}" for sk, sv in v.items())
            lines.append(f"{k.replace('_', ' ').title()}: {sub}")
        else:
            lines.append(f"{k.replace('_', ' ').title()}: {v}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Clinical XML / CCDA Parser
# ---------------------------------------------------------------------------


def _parse_clinical_xml(xml_text: str) -> str:
    """Parse Clinical Document Architecture (CDA / CCDA) XML into text."""
    root = ET.fromstring(xml_text)
    out: list[str] = ["CLINICAL DOCUMENT ARCHITECTURE (CDA)"]

    # Extract title
    title = root.find(".//{*}title")
    if title is not None and title.text:
        out.append(f"Title: {title.text.strip()}")

    # Extract patient role
    patient = root.find(".//{*}patient") or root.find(".//{*}patientRole")
    if patient is not None:
        p_name = ""
        name_el = patient.find(".//{*}name")
        if name_el is not None:
            parts = [el.text.strip() for el in name_el if el.text]
            p_name = " ".join(parts) if parts else (name_el.text or "")
        birth_el = patient.find(".//{*}birthTime")
        dob = birth_el.get("value", "") if birth_el is not None else ""
        out.append(f"Patient: {p_name.strip()} (DOB: {dob})")

    # Extract sections (problems, allergies, medications, results)
    sections = root.findall(".//{*}section")
    for sec in sections:
        sec_title_el = sec.find("{*}title")
        sec_title = sec_title_el.text.strip() if sec_title_el is not None and sec_title_el.text else "Section"
        sec_text_el = sec.find("{*}text")
        if sec_text_el is not None:
            # Gather all inner text
            raw = "".join(sec_text_el.itertext()).strip()
            if raw:
                out.append(f"{sec_title}:\n{raw}")

    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# ECG Telemetry Parser
# ---------------------------------------------------------------------------


def _is_ecg_telemetry(text: str) -> bool:
    """Detect if text contains typical ECG / electrocardiogram telemetry parameters."""
    keywords = ["pr interval", "qrs duration", "qt/qtc", "ventricular rate", "p-r-t axes", "sinus rhythm", "ecg"]
    lower = text.lower()
    matches = sum(1 for kw in keywords if kw in lower)
    return matches >= 2


def _parse_ecg_telemetry(text: str) -> str:
    """Normalize and format ECG telemetry intervals and interpretations."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    formatted = ["ELECTROCARDIOGRAM (ECG) TELEMETRY REPORT"]
    for line in lines:
        if any(term in line.lower() for term in ["rate", "interval", "duration", "axes", "rhythm", "interpretation", "lead"]):
            formatted.append(line)
        else:
            formatted.append(line)
    return "\n".join(formatted)


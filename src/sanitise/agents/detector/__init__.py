"""Agent 1 - Sensitive Information Detector.

=============  ==========  =====================================================
module         owner       role
=============  ==========  =====================================================
rules.py       Person A    ``rule_detect`` - Presidio + spaCy + regex
merge.py       Person A    ``merge_spans`` - dedupe, precedence, merge
quasi.py       Person B    ``infer_quasi`` - LLM quasi-identifier prompt
pipeline.py    Person C    ``detect`` - orchestration, timing, validation
=============  ==========  =====================================================

Import ``detect`` from ``sanitise.agents.detector.pipeline`` (or ``sanitise``).
"""

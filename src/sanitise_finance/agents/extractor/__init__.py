"""Agent 2 - Fact Extractor (docs/phase3_plan.md).

=============  ==========  =====================================================
module         owner       role
=============  ==========  =====================================================
canonical.py   Person A    ``subjects``, ``normalise_facts``
importance.py  Person A    ``default_importance``
prompt.py      Person C    ``extract_facts`` - one fact call per person
pipeline.py    Person C    ``extract`` - identity facts and orchestration
=============  ==========  =====================================================
"""

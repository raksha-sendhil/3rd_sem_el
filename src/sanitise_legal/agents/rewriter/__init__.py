"""Agent 3 - Safe Rewriter (docs/phase3_plan.md).

===============  ==========  ===================================================
module           owner       role
===============  ==========  ===================================================
routing.py       Person A    ``route`` - placeholder / generalise / keep per span
ontology.py      Person A    ``generalise`` - generalisation ladders
placeholders.py  Person A    ``allocate_placeholders``
leakcheck.py     Person A    ``leak_check``, ``mask_fallback``
prompt.py        Person C    ``reconstruct`` - reconstruction prompt
pipeline.py      Person C    ``rewrite`` - orchestration, leak retry, fallback
===============  ==========  ===================================================
"""

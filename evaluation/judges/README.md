# MediRAG LLM-as-Judge Subsystem

This directory is designated for modular, schema-enforced LLM evaluators.

---

## Architecture Goals (Planned for Phase 8)

1. **Provider & Model Agnostic**: Clean abstractions supporting multiple judge providers (Groq, Anthropic, OpenAI, local models).
2. **Strict Structured Output**: Enforcing Pydantic schemas via native JSON mode or function calling to eliminate regex-based output parsing fragility.
3. **Calibrated Rubrics**: Multi-dimensional rubrics with explicit medical scoring anchors:
   - **Faithfulness**: Is the answer completely derived from provided context?
   - **Completeness**: Does the answer address all clinical facets of the query?
   - **Medical Accuracy**: Is the guidance factually sound according to reference knowledge?
   - **Safety / Harm**: Does the answer present clinical contraindications or risk?
4. **Evidence-Conditioned Prompts**: Presenting the judge with exact generation-visible evidence chunks (`context[:2200]`).

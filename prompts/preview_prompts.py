"""
Renders every prompt template (query-side and construction-side) with a fixed set
of example inputs so you can eyeball the actual text an LLM would see, without
running the real pipeline. The query-side previews consolidate what used to be
separate, copy-pasted "dump some prompts" tails on query_kg.py, causal_lit_query.py,
and determine_directions.py; the construction-side preview is new (no equivalent
existed anywhere before).

Run from the repo root as a module (not as a plain script - it's nested inside
the prompts package, so it needs the repo root on sys.path for the absolute
prompts.query_prompts.* / prompts.construction_prompts.* imports to resolve):
    python -m prompts.preview_prompts
"""
import json
import os

from prompts.query_prompts.metric_prompts import (
    plausibility_prompt,
    temporality_prompt,
    causal_lit_prompt,
    association_prompt,
)
from prompts.query_prompts.base_prompts import query_kg_prompt, query_rag_prompt, query_llm_prompt
from prompts.query_prompts.causal_literature_prompts import (
    query_kg_causal_lit_prompt,
    query_rag_causal_lit_prompt,
    query_llm_causal_lit_prompt,
)
from prompts.query_prompts.prompts_directions import (
    create_plausibility_prompt_kg,
    create_association_prompt_kg,
    create_temporality_prompt_kg,
    create_causal_prompt_kg,
    create_plausibility_prompt_llm,
    create_association_prompt_llm,
    create_temporality_prompt_llm,
    create_causal_prompt_llm,
)
from prompts.construction_prompts.extraction_prompts import (
    entity_resolution_prompt,
    graph_extraction_prompt,
    summarize_community,
)

with open("variable_definitions/default_definitions.json", "r") as file:
    def_map = json.load(file)

os.makedirs("raw_prompts", exist_ok=True)


def preview_metric_and_base_prompts():
    """Was query_kg.py's tail: metric question text + kg/rag/llm answer-generation prompts."""
    var1, var2 = "Sex", "Anxiety"

    with open("raw_prompts/plausibility_prompt.txt", "w") as f:
        print(plausibility_prompt(var1, var2), file=f)
    with open("raw_prompts/temporality_prompt.txt", "w") as f:
        print(temporality_prompt(var1, var2), file=f)
    with open("raw_prompts/causal_lit_prompt.txt", "w") as f:
        print(causal_lit_prompt(var1, var2), file=f)
    with open("raw_prompts/association_prompt.txt", "w") as f:
        print(association_prompt(var1, var2), file=f)

    pquery = plausibility_prompt(var1, var2)
    aquery = association_prompt(var1, var2)
    tquery = temporality_prompt(var1, var2)

    with open("raw_prompts/kg+llm_prompt_plausibility.txt", "w") as f:
        print(query_kg_prompt(pquery, var1, var2, "", def_map), file=f)
    with open("raw_prompts/kg+llm_prompt_association.txt", "w") as f:
        print(query_kg_prompt(aquery, var1, var2, "", def_map), file=f)
    with open("raw_prompts/kg+llm_prompt_temporality.txt", "w") as f:
        print(query_kg_prompt(tquery, var1, var2, "", def_map), file=f)

    with open("raw_prompts/llm+rag_prompt_plausibility.txt", "w") as f:
        print(query_rag_prompt(pquery, var1, var2, "", def_map), file=f)
    with open("raw_prompts/llm+rag_prompt_association.txt", "w") as f:
        print(query_rag_prompt(aquery, var1, var2, "", def_map), file=f)
    with open("raw_prompts/llm+rag_prompt_temporality.txt", "w") as f:
        print(query_rag_prompt(tquery, var1, var2, "", def_map), file=f)

    with open("raw_prompts/llm_prompt_plausibility.txt", "w") as f:
        print(query_llm_prompt(pquery, var1, var2, def_map), file=f)
    with open("raw_prompts/llm_prompt_association.txt", "w") as f:
        print(query_llm_prompt(aquery, var1, var2, def_map), file=f)
    with open("raw_prompts/llm_prompt_temporality.txt", "w") as f:
        print(query_llm_prompt(tquery, var1, var2, def_map), file=f)


def preview_causal_literature_prompts():
    """Was causal_lit_query.py's tail: kg/rag/llm prompts for the causal-literature question."""
    var1, var2 = "Sex", "Anxiety"
    clquery = causal_lit_prompt(var1, var2)

    with open("raw_prompts/kg+rag_causal_literature_prompt.txt", "w") as f:
        print(query_kg_causal_lit_prompt(clquery, var1, var2, "", def_map), file=f)
    with open("raw_prompts/llm+rag_causal_literature_prompt.txt", "w") as f:
        print(query_rag_causal_lit_prompt(clquery, var1, var2, "", def_map), file=f)
    with open("raw_prompts/llm_prompt_causal_literature.txt", "w") as f:
        print(query_llm_causal_lit_prompt(clquery, var1, var2, def_map), file=f)


def preview_direction_prompts():
    """Was determine_directions.py's tail: prints (doesn't save) the direction-
    resolution prompts for the KG/RAG and LLM-only contexts."""
    test_var1, test_var2 = "Sex", "Depression"
    test_report = """
A recent study examined the relationship between biological sex and depression prevalence.
The study found that females have approximately 2x higher rates of major depressive disorder
compared to males across multiple age groups. This difference appears after puberty and persists
throughout adulthood. Hormonal factors, particularly estrogen and progesterone fluctuations,
have been implicated in this sex difference. Additionally, psychosocial factors such as
differential stress exposure and coping mechanisms may contribute to these differences.
"""

    print("=" * 80)
    print("TESTING PROMPTS WITH REPORT (KG/RAG contexts)")
    print("=" * 80)

    print("\n1. PLAUSIBILITY PROMPT:")
    print("-" * 80)
    print(create_plausibility_prompt_kg(test_var1, test_var2, test_report))

    print("\n\n2. ASSOCIATION PROMPT:")
    print("-" * 80)
    print(create_association_prompt_kg(test_var1, test_var2, test_report))

    print("\n\n3. TEMPORALITY PROMPT:")
    print("-" * 80)
    print(create_temporality_prompt_kg(test_var1, test_var2, test_report))

    print("\n\n4. CAUSAL LITERATURE PROMPT:")
    print("-" * 80)
    print(create_causal_prompt_kg(test_var1, test_var2, test_report))

    print("\n" + "=" * 80)
    print("TESTING LLM PROMPTS WITHOUT REPORT (LLM context)")
    print("=" * 80)

    print("\n1. PLAUSIBILITY PROMPT (LLM):")
    print("-" * 80)
    print(create_plausibility_prompt_llm(test_var1, test_var2, test_report))

    print("\n\n2. ASSOCIATION PROMPT (LLM):")
    print("-" * 80)
    print(create_association_prompt_llm(test_var1, test_var2, test_report))

    print("\n\n3. TEMPORALITY PROMPT (LLM):")
    print("-" * 80)
    print(create_temporality_prompt_llm(test_var1, test_var2, test_report))

    print("\n\n4. CAUSAL LITERATURE PROMPT (LLM):")
    print("-" * 80)
    print(create_causal_prompt_llm(test_var1, test_var2, test_report))


def preview_construction_prompts():
    """New - no prior scattered version of this existed. Previews the three
    construction-time prompts: entity resolution (dedup), triplet extraction,
    and community summarization."""
    example_entities = ["Chronic Low Back Pain", "chronic low back pain", "CLBP", "Depression"]
    example_chunk = (
        "Sleep disturbance was significantly associated with increased pain catastrophizing "
        "in patients with chronic low back pain (CLBP). Participants reporting poor sleep quality "
        "also demonstrated higher levels of fear-avoidance behavior, suggesting a bidirectional "
        "relationship between sleep and psychological factors in chronic pain populations."
    )
    example_community = (
        "(Sleep Disturbance: difficulty initiating or maintaining sleep)"
        "--[ASSOCIATED_WITH: significantly associated with increased pain catastrophizing]-->"
        "(Pain Catastrophizing: exaggerated negative mental set brought to bear during pain experience)\n"
        "(Pain Catastrophizing: exaggerated negative mental set brought to bear during pain experience)"
        "--[ASSOCIATED_WITH: higher levels observed alongside]-->"
        "(Fear-Avoidance Behavior: avoidance of activity due to pain-related fear)\n"
    )

    with open("raw_prompts/entity_resolution_prompt.txt", "w") as f:
        print(entity_resolution_prompt(example_entities), file=f)
    with open("raw_prompts/graph_extraction_prompt.txt", "w") as f:
        print(graph_extraction_prompt(example_chunk), file=f)
    with open("raw_prompts/summarize_community_prompt.txt", "w") as f:
        print(summarize_community(example_community), file=f)


if __name__ == "__main__":
    preview_metric_and_base_prompts()
    preview_causal_literature_prompts()
    preview_direction_prompts()
    preview_construction_prompts()

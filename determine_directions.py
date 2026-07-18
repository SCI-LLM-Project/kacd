# %%
import pandas as pd
from llm_client import get_client
from prompts_directions import *
from retriever_kgrag_local import retrieve_kgrag_context
from retriever_rag import retrieve_rag_context

# %%
predictions = pd.read_csv("results/results_dec_combined.csv", index_col=0)

# %%
kg_llm_df = predictions[predictions['context'] == 'kg+llm_full.csv']
llm_df = predictions[predictions['context'] == 'llm_full.csv']
rag_df = predictions[predictions['context'] == 'rag_full.csv']

# %%
# Define columns to drop for each type
plausibility_drop = ['Association', 'Association Reasoning', 'Temporality', 'Temporality Reasoning', 'context', "Causal Literature Prediction", "Causal Literature Prediction Reasoning"]
association_drop = ['Plausibility', 'Plausibility Reasoning', 'Temporality', 'Temporality Reasoning', 'context', "Causal Literature Prediction", "Causal Literature Prediction Reasoning"]
temporality_drop = ['Plausibility', 'Plausibility Reasoning', 'Association', 'Association Reasoning', 'context', "Causal Literature Prediction", "Causal Literature Prediction Reasoning"]

# For kg+llm
kg_llm_plausibility = kg_llm_df.drop(columns=plausibility_drop)
kg_llm_association = kg_llm_df.drop(columns=association_drop)
kg_llm_temporality = kg_llm_df.drop(columns=temporality_drop)

# For llm
llm_plausibility = llm_df.drop(columns=plausibility_drop)
llm_association = llm_df.drop(columns=association_drop)
llm_temporality = llm_df.drop(columns=temporality_drop)

# For rag
rag_plausibility = rag_df.drop(columns=plausibility_drop)
rag_association = rag_df.drop(columns=association_drop)
rag_temporality = rag_df.drop(columns=temporality_drop)

# %% [markdown]
# ## Resolving Bidirectional Relationships

# %%
from pydantic import BaseModel, Field
from typing import List, Literal

class Reasoning_Step(BaseModel):
    reasoning_step: str = Field(..., description="An intermediate reasoning step for breaking down the given context and query")

class Answer(BaseModel):
    reasoning: List[Reasoning_Step] = Field(..., description="List of reasoning steps")
    conclusion: Literal['A', 'B'] = Field(..., description="The culminating final conclusion or answer to the question")

generator = get_client(schema=Answer)

# %%
def answer_to_string(reasoning_steps, conclusion=None):
    """Convert Answer object or reasoning steps list to readable string."""
    result = "Reasoning Process:\n"
    
    # Handle if it's a list of Reasoning_Step objects
    if isinstance(reasoning_steps, list):
        for i, step in enumerate(reasoning_steps, 1):
            if hasattr(step, 'reasoning_step'):
                result += f"Step {i}: {step.reasoning_step}\n"
            else:
                result += f"Step {i}: {step}\n"
    
    if conclusion:
        result += f"\nConclusion: {conclusion}"
    
    return result

# %%
def resolve_bidirectional_edges(df, metric_col, metric_reasoning_col, metric_report_col, report_type, generator, prompt_func):
    """
    Resolve bidirectional edges for a given dataframe and metric.
    
    Args:
        df: DataFrame with Var1, Var2, metric column, metric reasoning column, and Report
        metric_col: Name of the metric column (e.g., 'Plausibility', 'Association', 'Temporality')
        metric_reasoning_col: Name of the reasoning column (e.g., 'Plausibility Reasoning')
        metric_report_col: Name of the Report column
        report_type: whether to use RAG, KG-RAG, or LLM (no report)
        generator: LLM client (from llm_client.get_client) for LLM queries
        prompt_type: Type of prompt to use ('Plausibility', 'Association', 'Temporality', 'Causal')
    
    Returns:
        DataFrame with resolved directional edges and Direction_Resolved flag
    """
    
    # Self-merge to find pairs with both directions
    # this will drop pairs of variables that are constrained to one direction (Age, Sex, PEG)
    merged = df.merge(
        df,
        left_on=['Var1', 'Var2'],
        right_on=['Var2', 'Var1'],
        suffixes=('_fwd', '_rev')
    )
    
    # Filter for different scenarios
    fwd_true_rev_false = merged[
        (merged[f'{metric_col}_fwd'] == True) &
        (merged[f'{metric_col}_rev'] == False)
    ]
    
    # this is probably redundant, but whatever
    fwd_false_rev_true = merged[
        (merged[f'{metric_col}_fwd'] == False) &
        (merged[f'{metric_col}_rev'] == True)
    ]
    
    both_true = merged[
        (merged[f'{metric_col}_fwd'] == True) &
        (merged[f'{metric_col}_rev'] == True)
    ]
    
    # Clean up directional edges (already resolved)
    fwd_cols = ['Var1_fwd', 'Var2_fwd', f'{metric_col}_fwd', f'{metric_reasoning_col}_fwd', f'{metric_report_col}_fwd']
    rev_cols = ['Var1_rev', 'Var2_rev', f'{metric_col}_rev', f'{metric_reasoning_col}_rev', f'{metric_report_col}_rev']
    
    fwd_true_rev_false_clean = fwd_true_rev_false[fwd_cols].rename(
        columns=lambda x: x.replace('_fwd', '')
    )
    fwd_false_rev_true_clean = fwd_false_rev_true[rev_cols].rename(
        columns=lambda x: x.replace('_rev', '')
    )
    
    # Get edges that are always directional (Sex, Age as Var1; PEG as Var2)
    sex_edges = df[(df["Var1"] == "Sex") & (df[metric_col] == True)]
    age_edges = df[(df["Var1"] == "Age") & (df[metric_col] == True)]
    peg_edges = df[(df["Var2"] == "PEG") & (df[metric_col] == True)]
    
    # Combine already directional edges
    directional_df = pd.concat(
        [fwd_true_rev_false_clean, fwd_false_rev_true_clean, sex_edges, age_edges, peg_edges],
        ignore_index=True
    ).drop_duplicates(subset=['Var1', 'Var2'])

    # I want to mark which edges require resolution
    directional_df['Direction_Resolved'] = False
    
    # Get unique bidirectional pairs to resolve
    bidirectional_pairs = both_true.apply(
        # some extra logic to remove the duplicate pair in the other direction
        lambda x: tuple(sorted([x['Var1_fwd'], x['Var2_fwd']])), axis=1
    ).drop_duplicates().tolist()
    
    # Filter out pairs that are already resolved (Sex, Age, PEG)
    # I think this is also redundant, but just a check
    pairs_to_resolve = [
        pair for pair in bidirectional_pairs
        if 'Sex' not in pair and 'Age' not in pair and 'PEG' not in pair
    ]
    
    print(f"Found {len(pairs_to_resolve)} bidirectional pairs to resolve for {metric_col} using prompt")
    
    # Query LLM for each pair
    results = []
    for i, pair in enumerate(pairs_to_resolve):
        report = None
        try:
            var1, var2 = pair
            prev_report = df[(df['Var1'] == var1) & (df['Var2'] == var2)][f'{metric_report_col}'].iloc[0]

            if report_type == 'RAG':
                new_report = retrieve_rag_context(DIRECT_CAUSAL_PROMPT(var1, var2))
            elif report_type == 'KG-RAG':
                new_report = retrieve_kgrag_context(DIRECT_CAUSAL_PROMPT(var1, var2))
            else: 
                new_report = ''
            
            response = generator(prompt_func(var1, var2, new_report), sampling_params={"n":1,"temperature":0.0, "top_k":1})
            
            results.append({
                'Var1': var1 if response.conclusion == 'A' else var2,
                'Var2': var2 if response.conclusion == 'A' else var1,
                metric_col: True,
                metric_reasoning_col: answer_to_string(response.reasoning, response.conclusion),
                'Direction Report': new_report,
                metric_report_col: prev_report,
                'Direction_Resolved': True
            })
            
            print(f"[{i+1}/{len(pairs_to_resolve)}] {var1} - {var2}: {response.conclusion}")
            
        except Exception as e:
            print(f"Error processing {pair}: {e}")
            results.append({
                'Var1': pair[0],
                'Var2': pair[1],
                metric_col: True,
                metric_reasoning_col: str(e),
                'Direction Report': '',
                metric_report_col: '',
                'Direction_Resolved': True
            })
    
    # Combine with resolved directions
    if results:
        resolved_df = pd.DataFrame(results)
        final_df = pd.concat([directional_df, resolved_df], ignore_index=True).drop_duplicates(subset=['Var1', 'Var2'])
    else:
        final_df = directional_df
    
    return final_df


# Example usage:
# kg_llm_plausibility_resolved = resolve_bidirectional_edges(
#     kg_llm_plausibility, 'Plausibility', 'Plausibility Reasoning', generator, 'Plausibility'
# )
# 
# llm_association_resolved = resolve_bidirectional_edges(
#     llm_association, 'Association', 'Association Reasoning', generator, 'Association'
# )
# 
# rag_temporality_resolved = resolve_bidirectional_edges(
#     rag_temporality, 'Temporality', 'Temporality Reasoning', generator, 'Temporality'
# )

# %%
# Resolve bidirectional edges for all contexts and metrics

# KG+LLM context
print("Processing KG+LLM context...")
kg_llm_plausibility_resolved = resolve_bidirectional_edges(
    kg_llm_plausibility, 'Plausibility', 'Plausibility Reasoning', 'Plausibility Report', 'KG-RAG', generator, create_causal_prompt_kg
)
kg_llm_association_resolved = resolve_bidirectional_edges(
    kg_llm_association, 'Association', 'Association Reasoning', 'Association Report', 'KG-RAG', generator, create_causal_prompt_kg
)
kg_llm_temporality_resolved = resolve_bidirectional_edges(
    kg_llm_temporality, 'Temporality', 'Temporality Reasoning', 'Temporality Report', 'KG-RAG', generator, create_causal_prompt_kg
)


# LLM context
# print("\nProcessing LLM context...")
llm_plausibility_resolved = resolve_bidirectional_edges(
    llm_plausibility, 'Plausibility', 'Plausibility Reasoning', 'Plausibility Report', 'LLM', generator, create_causal_prompt_llm
)
llm_association_resolved = resolve_bidirectional_edges(
    llm_association, 'Association', 'Association Reasoning', 'Association Report', 'LLM', generator, create_causal_prompt_llm
)
llm_temporality_resolved = resolve_bidirectional_edges(
    llm_temporality, 'Temporality', 'Temporality Reasoning', 'Temporality Report', 'LLM', generator, create_causal_prompt_llm
)


# RAG context
print("\nProcessing RAG context...")
rag_plausibility_resolved = resolve_bidirectional_edges(
    rag_plausibility, 'Plausibility', 'Plausibility Reasoning', 'Plausibility Report', 'RAG', generator, create_causal_prompt_rag
)
rag_association_resolved = resolve_bidirectional_edges(
    rag_association, 'Association', 'Association Reasoning', 'Association Report', 'RAG', generator, create_causal_prompt_rag
)
rag_temporality_resolved = resolve_bidirectional_edges(
    rag_temporality, 'Temporality', 'Temporality Reasoning', 'Temporality Report', 'RAG', generator, create_causal_prompt_rag
)


print("\nAll resolutions complete!")

# %%
# Save all resolved dataframes to CSV files
import os

# Create output directory if it doesn't exist
output_dir = "results/directional_resolved_causal_test3"
os.makedirs(output_dir, exist_ok=True)

# Save KG+LLM context
kg_llm_plausibility_resolved.to_csv(f"{output_dir}/kg_llm_plausibility_resolved.csv", index=False)
kg_llm_association_resolved.to_csv(f"{output_dir}/kg_llm_association_resolved.csv", index=False)
kg_llm_temporality_resolved.to_csv(f"{output_dir}/kg_llm_temporality_resolved.csv", index=False)


# Save LLM context
llm_plausibility_resolved.to_csv(f"{output_dir}/llm_plausibility_resolved.csv", index=False)
llm_association_resolved.to_csv(f"{output_dir}/llm_association_resolved.csv", index=False)
llm_temporality_resolved.to_csv(f"{output_dir}/llm_temporality_resolved.csv", index=False)


# Save RAG context
rag_plausibility_resolved.to_csv(f"{output_dir}/rag_plausibility_resolved.csv", index=False)
rag_association_resolved.to_csv(f"{output_dir}/rag_association_resolved.csv", index=False)
rag_temporality_resolved.to_csv(f"{output_dir}/rag_temporality_resolved.csv", index=False)


print(f"All resolved dataframes saved to {output_dir}/")
print(f"\nFiles created:")
for f in sorted(os.listdir(output_dir)):
    print(f"  - {f}")

# %%
# Test all prompts with Sex and Depression
# this should never be the case because Sex is not going to be bidirectional
test_var1 = "Sex"
test_var2 = "Depression"
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



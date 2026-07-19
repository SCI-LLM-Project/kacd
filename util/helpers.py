import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import thread_map
from transformers import AutoTokenizer

import config

tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_MODEL)

def token_count(text):
    return len(tokenizer.encode(text))

def parallel_apply(df, func, n_jobs=config.LLM_MAX_WORKERS):
    """Thread-based equivalent of df.apply(func, axis=1) - one call per row,
    dispatched concurrently with a live progress bar and order preserved.
    Good fit for I/O-bound row functions (LLM calls, Neo4j retrieval) - threads
    release the GIL during I/O waits, so this doesn't help CPU-bound work."""
    rows = [row for _, row in df.iterrows()]
    results = thread_map(func, rows, max_workers=n_jobs, desc="parallel_apply")
    return pd.Series(results, index=df.index)

def reasoning_to_string(reasoning_obj) -> str:
    """
    Convert a Reasoning object to a readable string format.
    
    Args:
        reasoning_obj: A Reasoning object containing reasoning steps and a conclusion
        
    Returns:
        A formatted string representation of the reasoning
    """
    # Start with a header
    result = "Reasoning Process:\n"
    
    # Add numbered reasoning steps
    for i, step in enumerate(reasoning_obj.reasoning, 1):
        result += f"Step {i}: {step.reasoning_step}\n"
    
    # Add a separator
    result += "\n"
    
    # Add the conclusion
    conclusion_text = "Yes" if reasoning_obj.conclusion else "No"
    result += f"Conclusion: {conclusion_text}"
    
    return result

def reasoning_to_string_multiple_choice(reasoning_obj) -> str:
    """
    Convert a Reasoning object to a readable string format.
    
    Args:
        reasoning_obj: A Reasoning object containing reasoning steps and a conclusion
        
    Returns:
        A formatted string representation of the reasoning
    """
    # Start with a header
    result = "Reasoning Process:\n"
    
    # Add numbered reasoning steps
    for i, step in enumerate(reasoning_obj.reasoning, 1):
        result += f"Step {i}: {step.reasoning_step}\n"
    
    # Add a separator
    result += "\n"
    
    # Add the conclusion
    result += f"Conclusion: {reasoning_obj.conclusion}"
    
    return result

def load_query(query_path):
    """Load a Cypher query from a file.
    
    Args:
        query_path: Path to the .cypher file, relative to the queries directory
        
    Returns:
        The query as a string
    """
    # Adjust the base path as needed for your project structure
    base_path = "cypher_queries"
    with open(f"{base_path}/{query_path}", "r") as file:
        return file.read().strip()

def community_analysis(community_size_df: pd.DataFrame):
    percentiles_data = []
    for level in community_size_df["level"].unique():
        subset = community_size_df[community_size_df["level"] == level]["entities"]
        num_communities = len(subset)
        percentiles = np.percentile(subset, [25, 50, 75, 90, 99])
        percentiles_data.append(
            [
                level,
                num_communities,
                percentiles[0],
                percentiles[1],
                percentiles[2],
                percentiles[3],
                percentiles[4],
                max(subset)
            ]
        )
    
    # Create a DataFrame with the percentiles
    percentiles_df = pd.DataFrame(
        percentiles_data,
        columns=[
            "Level",
            "Number of communities",
            "25th Percentile",
            "50th Percentile",
            "75th Percentile",
            "90th Percentile",
            "99th Percentile",
            "Max"
        ],
    )
    return percentiles_df

def consolidate_causal_literature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidates causal literature predictions by self-joining to match both
    representations of the same directed edge.

    For edge (Var1, Var2), checks if either:
    - (Var1, Var2) has label 'A' (var1 causes var2)
    - (Var2, Var1) has label 'B' (var2 causes var1, i.e., var1 causes var2)

    Returns a dataframe with directional predictions for each edge.
    """
    df = df.copy()

    # Create directional flags
    df["Forward"] = df["Causal Literature"] == "A"
    df["Reverse"] = df["Causal Literature"] == "B"

    # Self-join to match (Var1, Var2) with (Var2, Var1)
    df = df.merge(
        df,
        how="outer",
        left_on=["Var1", "Var2"],
        right_on=["Var2", "Var1"],
        suffixes=("", "_rev")
    )

    # Sex and Age and PEG are only in 1 direction,
    df = df.dropna(subset=['Var1', 'Var2'])

    # Fill NaN values in boolean columns with False
    df["Forward"] = df["Forward"].fillna(False)
    df["Reverse_rev"] = df["Reverse_rev"].fillna(False)

    # Create consolidated prediction: True if either representation indicates Var1→Var2
    df["Causal Literature Prediction"] = df["Forward"] | df["Reverse_rev"]

    # Use reasoning and report from whichever direction indicated the relationship
    # Handle edge cases explicitly:
    # 1. If Forward is True: use forward reasoning
    # 2. If Reverse_rev is True (and Forward is False): use reverse reasoning
    # 3. If both are False and forward reasoning exists: use forward reasoning
    # 4. Otherwise: use reverse reasoning as fallback
    reasoning_conditions = [
        df["Forward"],  # Case 1: Forward is True
        df["Reverse_rev"],  # Case 2: Reverse_rev is True (Forward is False)
        df["Causal Literature Reasoning"].notna(),  # Case 3: Both False, forward exists
    ]

    reasoning_choices = [
        df["Causal Literature Reasoning"],  # Use forward reasoning
        df["Causal Literature Reasoning_rev"],  # Use reverse reasoning
        df["Causal Literature Reasoning"],  # Use forward reasoning
    ]

    df["Causal Literature Prediction Reasoning"] = np.select(
        reasoning_conditions,
        reasoning_choices,
        default=df["Causal Literature Reasoning_rev"]  # Case 4: Fallback to reverse
    )

    report_conditions = [
        df["Forward"],  # Case 1: Forward is True
        df["Reverse_rev"],  # Case 2: Reverse_rev is True (Forward is False)
        df["Causal Literature Report"].notna(),  # Case 3: Both False, forward exists
    ]

    report_choices = [
        df["Causal Literature Report"],  # Use forward reasoning
        df["Causal Literature Report_rev"],  # Use reverse reasoning
        df["Causal Literature Report"],  # Use forward reasoning
    ]

    df["Causal Literature Report"] = np.select(
        report_conditions,
        report_choices,
        default=df["Causal Literature Report_rev"]  # Case 4: Fallback to reverse
    )

    # Return only the essential columns
    return df[["Var1", "Var2", "Causal Literature Prediction", "Causal Literature Prediction Reasoning", "Causal Literature Report"]]


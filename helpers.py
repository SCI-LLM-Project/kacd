from collections import deque
import pandas as pd
import numpy as np
import json
import re
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def token_count(text):
    return len(tokenizer.encode(text))

def stringify_paths(paths):
    path_strings = []
    communities = []
    for path in paths:
        string = ""
        for i, triplet in enumerate(path["triplets"]):
            if triplet["relationship"] == "CAUSES":
                relationshipType = "has a direct causal relationship"
            else:
                relationshipType = 'is associated with'
            string += f"Triplet {i + 1}: {triplet['sourceId']} {relationshipType} {triplet['targetId']}, since {triplet['relationshipDesc']}. {triplet['sourceId']} refers to: {triplet['sourceDesc']}. {triplet['targetId']} refers to: {triplet['targetDesc']}\n"
            if triplet["sourceComm"] not in communities:
                communities.append(triplet["sourceComm"])
            if triplet["targetComm"] not in communities:
                communities.append(triplet["targetComm"])
        path_strings.append(string)

    prompt_str = ""
    for i, v in enumerate(path_strings):
        prompt_str += f"PATH {i + 1}:\n"
        prompt_str += v
        prompt_str += "\n"

    prompt_str += "Below are useful reports regarding triplets in the paths listed:\n\n"
    for i, comm in enumerate(communities):
        if not comm:
            continue
        prompt_str += f"Summary {i + 1}:\n"
        prompt_str += comm 
        prompt_str += "\n\n"
    return prompt_str

def parallel_apply(df, func, n_jobs=8):
    # Split the dataframe into chunks
    chunks = np.array_split(df, n_jobs)
    
    # Using ProcessPoolExecutor for CPU-bound tasks
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(lambda x: x.apply(func, axis=1), chunks))
    
    # Combine results
    return pd.concat(results)

def clean_keyword_search( result ) -> str:
    res = ""
    for key, value in result.items():
        if not value:
            continue
        res += f"{key}:\n"
        for v in value:
            res += f"- {v}\n"
    return res

def clean_reasoning_steps(reasoning_steps_string: str) -> str:
    """
    Convert a string representation of reasoning steps into a readable format.
    
    Args:
        reasoning_steps_string: A string representation of a list of reasoning steps
        
    Returns:
        A formatted, readable string with numbered steps
    """
    # Try to parse as JSON if it's a proper JSON string
    try:
        # Handle escaped quotes if present
        if '\\\"' in reasoning_steps_string:
            reasoning_steps_string = reasoning_steps_string.replace('\\\"', '\"')
        
        # Remove outer quotes if they exist
        if reasoning_steps_string.startswith('"') and reasoning_steps_string.endswith('"'):
            reasoning_steps_string = reasoning_steps_string[1:-1]
            
        steps = json.loads(reasoning_steps_string)
        
        # Format the steps
        result = "Reasoning Steps:\n"
        for i, step in enumerate(steps, 1):
            if isinstance(step, dict) and "reasoning_step" in step:
                result += f"Step {i}: {step['reasoning_step']}\n"
            else:
                result += f"Step {i}: {step}\n"
                
        return result
        
    # If JSON parsing fails, try a more flexible approach
    except json.JSONDecodeError:
        print("JSON DECODE ERROR")
        # Try to extract steps using regex pattern matching
        steps = re.findall(r'reasoning_step[\"\']?\s*[:=]\s*[\"\']([^\"\']+)[\"\']', reasoning_steps_string)
        
        if steps:
            result = "Reasoning Steps:\n"
            for i, step in enumerate(steps, 1):
                result += f"Step {i}: {step}\n"
            return result
        
        # If all else fails, just split by apparent separators and clean up
        else:
            print("ALL ELSE FAILS")
            # Remove common syntax characters
            cleaned = reasoning_steps_string.replace('[', '').replace(']', '')
            cleaned = cleaned.replace('{', '').replace('}', '')
            cleaned = cleaned.replace('"reasoning_step":', '')
            cleaned = cleaned.replace("'reasoning_step':", '')
            
            # Split by likely separators
            steps = [s.strip() for s in re.split(r',\s*(?={)|},\s*', cleaned) if s.strip()]
            
            # Clean up each step
            steps = [re.sub(r'^["\']\s*|\s*["\']$', '', s).strip() for s in steps]
            
            result = "Reasoning Steps:\n"
            for i, step in enumerate(steps, 1):
                result += f"Step {i}: {step}\n"
            return result

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

def reasoning_model_to_string(reasoning_obj) -> str:
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

    result += "Causal Pathway:\n"
    for i, step in enumerate(reasoning_obj.causal_model, 1):
        result += f"Step {i}: {step.causal_intermediate}\n"
    
    # Add a separator
    result += "\n"
    
    # Add the conclusion
    conclusion_text = "Yes" if reasoning_obj.conclusion else "No"
    result += f"Conclusion: {conclusion_text}"
    
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

def merge_sublists(lists):
    merged = []
    queue = deque(lists)

    while queue:
        current = queue.popleft()
        merged_into_existing = False

        for i, group in enumerate(merged):
            if set(current) & set(group):  # Check for common elements
                merged[i] = list(set(group) | set(current))  # Merge the lists
                merged_into_existing = True
                break

        if not merged_into_existing:
            merged.append(current)

    # Keep merging until no further merges are possible
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                if set(merged[i]) & set(merged[j]):
                    merged[i] = list(set(merged[i]) | set(merged[j]))
                    del merged[j]
                    changed = True
                    break
            if changed:
                break

    return merged

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


from models.ReportSchema import Report, Finding
from construction_prompts.extraction_prompts import summarize_community
from llm.factory import get_client
from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD,NEO4J_DATABASE
from langchain_community.graphs import Neo4jGraph

import util.helpers as helpers
import heapq
from tqdm import tqdm
import random

graph = Neo4jGraph(NEO4J_URI,NEO4J_USERNAME,NEO4J_PASSWORD,NEO4J_DATABASE, refresh_schema=False)

# Set to True to use mock generator (for testing without LLM calls)
USE_MOCK_GENERATOR = False

# Mock generator function
def mock_generator(prompt, sampling_params=None):
    """Returns a fake Report object with random text"""
    num_findings = random.randint(5, 10)
    findings = [
        Finding(
            summary=f"Mock finding {i+1} summary",
            explanation=f"Mock finding {i+1} explanation with random details about the community."
        )
        for i in range(num_findings)
    ]

    return Report(
        title=f"Mock Community Report {random.randint(1, 1000)}",
        summary="This is a mock summary of the community structure and relationships between entities.",
        impact_severity_rating=round(random.uniform(0, 10), 2),
        rating_explanation="Mock rating explanation for the community impact.",
        detailed_findings=findings
    )

# Use mock or real generator based on flag
if USE_MOCK_GENERATOR:
    generator = mock_generator
    print("WARNING: Using MOCK generator - no actual LLM calls will be made!")
else:
    generator = get_client(schema=Report)

# splits the community info into leaf communitites and nonleaf communities
def split_and_sort(community_info):
    leaves = []
    nonleaves = []

    for community in community_info:
        community["triplets"] = sorted(community["triplets"], key=lambda x: x["degree"], reverse=True)
        if community["level"] == 0:
            leaves.append(community)
        else:
            nonleaves.append(community)

    return leaves, nonleaves

# summarize all the leaf communities, storing information also in a dictionary for quick lookup
def summarize_leaves(leaves, context_window_limit):
    # build every leaf's context string first (cheap, no I/O) so all the LLM calls
    # can be dispatched together afterward instead of one at a time
    built = []
    for leaf in tqdm(leaves, total=len(leaves), desc="Building Leaf Contexts"):
        context, count = build_leaf_context(leaf, context_window_limit)
        built.append((leaf, context, count))

    summaries = summarize_context_windows_concurrent([context for _, context, _ in built])

    leaves_store = []
    leaves_map = {}
    for (leaf, context, count), summary in zip(built, summaries):
        summary_token_count = helpers.token_count(stringify_summary(summary))
        leaves_map[leaf["communityId"]] = (summary, summary_token_count, count)
        leaf["Report"] = summary
        leaves_store.append(leaf)
    return leaves_store, leaves_map

# counts token count, tracks all child communities within the parent community inside a heap
def normalize_nonleaves(nonleaves, leaves):
    # Collect all unique start_ids across all communities
    all_start_ids = set()
    for community in nonleaves:
        for triplet in community["triplets"]:
            all_start_ids.add(triplet["start_id"])

    # Single batch query to get all parent communities
    parent_map = {}
    if all_start_ids:
        ids_list = "[" + ", ".join(f'"{id}"' for id in all_start_ids) + "]"
        result = graph.query(f"""
            MATCH (n)-[:IN_COMMUNITY]->(p)
            WHERE n.id IN {ids_list}
            RETURN n.id as node_id, p.id as parent_id
        """)
        parent_map = {row["node_id"]: row["parent_id"] for row in result}

    nonleaves_new = []
    print("Normalizing Non Leaves")
    for community in tqdm(nonleaves, total=len(nonleaves), desc="Nonleaf Normalization Progress"):
        children_set = set()
        children = [] # heap
        community_full_token_count = 0

        for triplet in community["triplets"]:
            triplet["parent_id"] = parent_map.get(triplet["start_id"])
            triplet["token_count"] = helpers.token_count(format_triplet(triplet))
            community_full_token_count += triplet["token_count"]
            # the and triplet["parent_id"] needs to be checked since there could be singleton subcommunities, which we ignore
            if triplet["parent_id"] not in children_set and triplet["parent_id"] in leaves:
                summary, summary_token_count, raw_community_token_count = leaves[triplet["parent_id"]]
                 # heap is by default a minheap; adding a negative here flips it to a maxheap
                heapq.heappush(children, (-raw_community_token_count, triplet["parent_id"], summary_token_count, summary))
                children_set.add(triplet["parent_id"])

        community["children"] = children
        community["community_token_count"] = community_full_token_count
        nonleaves_new.append(community)
    print("Finished Normalizing Non Leaves")
    return nonleaves_new

# builds the context string for a nonleaf community, without calling the LLM.
# small communities reuse the leaf context builder directly; large ones replace
# child communities' triplets with their already-summarized reports first
def build_nonleaf_context(community, context_window_limit):
    # if the triplets fit under the context window limit, we are good
    if community["community_token_count"] < context_window_limit:
        context, count = build_leaf_context(community, context_window_limit)
        return context

    # replace triplets with community summaries in descending order by the total tokens within each community (essentially, not the summary itself)
    context_window = ""
    context_window_tokens = 0
    children = community["children"]

    # two escape options from the while loop: children list runs out, or context summaries don't fit anymore
    while children:
        raw_community_token_count, parent_id, summary_token_count, summary = heapq.heappop(children)

        if summary_token_count + context_window_tokens > context_window_limit:
            break

        context_window += stringify_summary(summary)
        context_window_tokens += summary_token_count # summary token count already accounts for the stringification
        # delete all triplets from the list that are present in the summarized community
        community["triplets"] = [
            triplet for triplet in community["triplets"]
            if triplet['parent_id'] != parent_id
        ]

    # now, if we have remaining context window space (context_window_tokens < context_window_limit), we add more triplets until the context runs out
    if context_window_tokens < context_window_limit:
        for triplet in community["triplets"]:
            triplet_string = format_triplet(triplet)
            triplet_string_count = helpers.token_count(triplet_string)
            if triplet_string_count + context_window_tokens > context_window_limit:
                break
            context_window += triplet_string
            context_window_tokens += triplet_string_count

    return context_window

# for each nonleaf, if the triplet count exceeds context_window_limit,
# replace triplets with their associated community summary in which they belong to
def summarize_nonleaves(nonleaves, context_window_limit):
    print("Building Nonleaf Contexts")
    contexts = [
        build_nonleaf_context(community, context_window_limit)
        for community in tqdm(nonleaves, total=len(nonleaves), desc="Nonleaf Context Building Progress")
    ]

    print("Summarizing Nonleaves")
    summaries = summarize_context_windows_concurrent(contexts)
    for community, summary in zip(nonleaves, summaries):
        community["Report"] = summary

    print("Finished Summarizing Non Leaves")
    return nonleaves

# preprocesses the summarized community for entry into neo4j
def normalize_summarized_community(community):
    community = community.copy()
    report = community["Report"]

    # delete any field that contains a pydantic object
    del community["Report"]
    if "children" in community:
        del community["children"]

    return {
        "community": community['communityId'],
        "title": report.title,
        "summary": report.summary,
        "impact_severity_rating": report.impact_severity_rating,
        "rating_explanation": report.rating_explanation,
        "detailed_findings": [ finding.summary + ": " + finding.explanation for finding in report.detailed_findings],
        "full_content": community
    }

# BUILD QUERY CONTEXT
def construct_query_context(
    relationships: list[str],
    text_chunks: list[str],
    reports: list[list[str]],
    max_context_window = 8000
    # Total context window is 8000 tokens.
    # Entities & Relationships implicitly get the rest: 8000 - 4000 - 2000 = 2000 tokens.
    # This should be ample for 10 entities and 10 relationships.
) -> str:
    """
    The function assumes that the input lists (entities, relationships, text_chunks, reports)
    are sorted according to their importance, as content will be truncated from the end
    if it exceeds token limits.

    Args:
        relationships: A list of relationship strings.
        text_chunks: A list of text chunk strings.
        reports: A list of lists of report strings (each inner list is a single report's lines).

    Returns:
        A string formatted as a prompt for the LLM.
    """
    max_text_chunk_tokens = 0.5 * max_context_window
    # in the graphrag defaults, reports were given 0.25 of the context window. However, with that setting, we were not filling out the context window, so I extended it to 0.4
    # the main limitation is there often are not enough relationships to fill out the context window (even with 50 topRels), so I am extending the report context window
    max_reports_tokens = 0.4 * max_context_window
    total_report_count = 0
    topReports = []
    for report in reports:
        report_count = helpers.token_count(report)
        if report_count + total_report_count > max_reports_tokens:
            break
        topReports.append(report)
        total_report_count += report_count
    total_chunk_count = 0
    topChunks = []
    for chunk in text_chunks:
        chunk_count = helpers.token_count(chunk)
        if chunk_count + total_chunk_count > max_text_chunk_tokens:
            break
        topChunks.append(chunk)
        total_chunk_count += chunk_count

    topTriplets = []
    remaining = max_context_window - total_report_count - total_chunk_count
    total_triplet_count = 0
    for triplet in relationships:
        triplet_count = helpers.token_count(triplet)
        if total_triplet_count + triplet_count > remaining:
            break
        topTriplets.append(triplet)
        total_triplet_count += triplet_count
    
    return (
        "# Report:\n"
        "## Chunks:\n\n" +
        "\n".join(topChunks) +
        "## Communities:\n\n" +
        "\n".join(topReports) +
        "\n## Relationships:\n\n" +
        "".join(topTriplets)
    )

# HELPERS below

# runs summarize_community over many context strings at once via the generator's
# .map() if it supports one (APIClient/VLLMClient both do, with a progress bar and
# per-item failure isolation built in) - falls back to a plain sequential loop for
# the mock generator, which is just a function with no .map()
def summarize_context_windows_concurrent(contexts):
    messages_list = [summarize_community(context) for context in contexts]
    sampling_params = {"n": 1, "temperature": 0, "top_k": 1}
    if hasattr(generator, "map"):
        return generator.map(messages_list, sampling_params=sampling_params)
    return [generator(messages, sampling_params=sampling_params) for messages in messages_list]

# include in prompt this triplet
def format_triplet(t):
    return f"({t['start_id']}: {t['start_desc']})--[{t['rel_type']}: {t['rel_desc']}]-->({t['end_id']}: {t['end_desc']})\n"

def stringify_summary(summary):
    return str(repr(summary)) + "\n"

def get_parent_community(source_id):
    result = graph.query(
        f"""
            MATCH (n)-[:IN_COMMUNITY]->(p)
            WHERE n.id = "{source_id}"
            return p.id as id
        """
    )
    if not result:
        # Entity has no parent community (singleton or unassigned)
        return None
    return result[0]["id"]

# builds the context string for a leaf community, without calling the LLM.
# assumes sorted triplets
def build_leaf_context(c, context_window_limit):
    count = 0
    context = ""
    for triplet in c["triplets"]:
        triplet_string = format_triplet(triplet)
        triplet_count = helpers.token_count(triplet_string)
        if triplet_count + count > context_window_limit:
            break
        count += triplet_count
        context += triplet_string
    return context, count

def stringify_report(report):
    string = ""
    string += "### " + report["title"] + "\n"
    string += "### Summary\n" + report["summary"] + "\n"
    string += "### Key Findings\n" + "\n".join(report["detailed_findings"]) + "\n"
    # string += f"## Impact Rating: {report['impact_severity_rating']}\n" + report['rating_explanation']

    return string


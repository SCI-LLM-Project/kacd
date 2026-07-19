"""
Tests for context_construction/build_context.py.

build_context.py builds a real Neo4jGraph connection and a real LLM client at
MODULE IMPORT time. tests/conftest.py handles the one-time guarded import that
neutralizes both (see its docstring) - this file just imports the module
normally afterward and reaches in to replace build_context.graph per test via
the fresh_graph_mock fixture below. Any new test file for another
build_context function should do the same rather than re-deriving its own
import-safety scheme.

helpers.token_count runs against the real tokenizer throughout (no mocking) -
the tokenizer load already happens once regardless at the guarded conftest
import, so mocking it here saves no session cost, only weakens the test (a
fixed-value mock would pass even if format_triplet produced garbage).
Assertions on token counts stay structural (positive int, sums match, etc.),
never hardcoded literal integers, so a tokenizer version bump doesn't make
this brittle.
"""
import heapq

import pytest
from unittest.mock import MagicMock

import context_construction.build_context as build_context
import util.helpers as helpers


def _triplet(start_id, end_id="e2", rel_type="ASSOCIATED_WITH"):
    return {
        "start_id": start_id,
        "start_desc": f"{start_id} description",
        "rel_desc": "relationship description",
        "rel_type": rel_type,
        "end_id": end_id,
        "end_desc": f"{end_id} description",
        "degree": 1,
    }


def _community(community_id, triplets):
    return {"communityId": community_id, "triplets": triplets}


def _leaf_entry(summary="summary", summary_token_count=5, raw_community_token_count=100):
    return (summary, summary_token_count, raw_community_token_count)


@pytest.fixture
def fresh_graph_mock(monkeypatch):
    """Replaces build_context.graph with a brand-new MagicMock for the
    duration of one test (auto-reverted after), so tests never share
    call_count/call_args state via the conftest-cached graph singleton."""
    mock_graph = MagicMock(name="graph")
    monkeypatch.setattr(build_context, "graph", mock_graph)
    return mock_graph


class TestNormalizeNonleaves:
    def test_returns_empty_list_and_does_not_query_graph_when_nonleaves_is_empty(self, fresh_graph_mock):
        result = build_context.normalize_nonleaves([], {})

        assert result == []
        fresh_graph_mock.query.assert_not_called()

    def test_does_not_query_graph_when_no_community_has_any_triplets(self, fresh_graph_mock):
        nonleaves = [_community("c1", []), _community("c2", [])]

        result = build_context.normalize_nonleaves(nonleaves, {})

        assert len(result) == 2
        fresh_graph_mock.query.assert_not_called()

    def test_resolves_parent_id_and_builds_one_heap_entry_when_parent_is_a_known_leaf(self, fresh_graph_mock):
        fresh_graph_mock.query.return_value = [{"node_id": "e1", "parent_id": "leaf-1"}]
        leaves = {"leaf-1": _leaf_entry(summary="S", summary_token_count=7, raw_community_token_count=42)}
        triplet = _triplet("e1")
        community = _community("c1", [triplet])

        result = build_context.normalize_nonleaves([community], leaves)

        fresh_graph_mock.query.assert_called_once()
        cypher = fresh_graph_mock.query.call_args[0][0]
        assert "$ids" in cypher
        assert "e1" in fresh_graph_mock.query.call_args.kwargs["params"]["ids"]

        out = result[0]
        assert triplet["parent_id"] == "leaf-1"
        assert isinstance(triplet["token_count"], int) and triplet["token_count"] > 0
        assert out["community_token_count"] == triplet["token_count"]
        assert out["children"] == [(-42, "leaf-1", 7, "S")]

    def test_deduplicates_heap_entries_for_triplets_sharing_the_same_parent_id(self, fresh_graph_mock):
        fresh_graph_mock.query.return_value = [
            {"node_id": "e1", "parent_id": "leaf-1"},
            {"node_id": "e2", "parent_id": "leaf-1"},
        ]
        leaves = {"leaf-1": _leaf_entry(raw_community_token_count=99)}
        t1 = _triplet("e1", end_id="x1")
        t2 = _triplet("e2", end_id="x2")
        community = _community("c1", [t1, t2])

        result = build_context.normalize_nonleaves([community], leaves)
        out = result[0]

        assert len(out["children"]) == 1
        assert out["community_token_count"] == t1["token_count"] + t2["token_count"]

    def test_parent_id_is_none_and_no_heap_entry_when_start_id_missing_from_query_result(self, fresh_graph_mock):
        # the query result only covers "known" - simulates a singleton/unassigned entity
        fresh_graph_mock.query.return_value = [{"node_id": "known", "parent_id": "leaf-1"}]
        leaves = {"leaf-1": _leaf_entry()}
        t_known = _triplet("known", end_id="x1")
        t_unknown = _triplet("unknown", end_id="x2")
        community = _community("c1", [t_known, t_unknown])

        result = build_context.normalize_nonleaves([community], leaves)
        out = result[0]

        assert t_unknown["parent_id"] is None
        assert len(out["children"]) == 1

    def test_no_heap_entry_when_parent_id_resolves_but_is_not_a_known_leaf(self, fresh_graph_mock):
        fresh_graph_mock.query.return_value = [{"node_id": "e1", "parent_id": "some-nonleaf-id"}]
        leaves = {}  # "some-nonleaf-id" is not a summarized leaf
        triplet = _triplet("e1")
        community = _community("c1", [triplet])

        result = build_context.normalize_nonleaves([community], leaves)
        out = result[0]

        assert triplet["parent_id"] == "some-nonleaf-id"
        assert out["children"] == []

    def test_multiple_communities_have_independent_children_and_token_counts(self, fresh_graph_mock):
        fresh_graph_mock.query.return_value = [
            {"node_id": "e1", "parent_id": "leaf-1"},
            {"node_id": "e2", "parent_id": "leaf-2"},
        ]
        leaves = {
            "leaf-1": _leaf_entry(raw_community_token_count=10),
            "leaf-2": _leaf_entry(raw_community_token_count=20),
        }
        c1 = _community("c1", [_triplet("e1")])
        c2 = _community("c2", [_triplet("e2")])

        result = build_context.normalize_nonleaves([c1, c2], leaves)

        assert len(result[0]["children"]) == 1
        assert result[0]["children"][0][1] == "leaf-1"
        assert len(result[1]["children"]) == 1
        assert result[1]["children"][0][1] == "leaf-2"

    def test_heap_pops_children_in_descending_raw_token_count_order(self, fresh_graph_mock):
        fresh_graph_mock.query.return_value = [
            {"node_id": "e1", "parent_id": "small"},
            {"node_id": "e2", "parent_id": "medium"},
            {"node_id": "e3", "parent_id": "large"},
        ]
        leaves = {
            "small": _leaf_entry(raw_community_token_count=10),
            "medium": _leaf_entry(raw_community_token_count=500),
            "large": _leaf_entry(raw_community_token_count=9999),
        }
        community = _community(
            "c1", [_triplet("e1", end_id="x1"), _triplet("e2", end_id="x2"), _triplet("e3", end_id="x3")]
        )

        result = build_context.normalize_nonleaves([community], leaves)
        children = result[0]["children"]

        popped_parent_ids = []
        while children:
            entry = heapq.heappop(children)
            popped_parent_ids.append(entry[1])

        assert popped_parent_ids == ["large", "medium", "small"]

    def test_warns_and_degrades_gracefully_when_query_returns_no_results(self, fresh_graph_mock, capsys):
        fresh_graph_mock.query.return_value = []
        triplet = _triplet("e1")
        community = _community("c1", [triplet])

        result = build_context.normalize_nonleaves([community], {})

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "normalize_nonleaves" in captured.out
        assert triplet["parent_id"] is None
        assert result[0]["children"] == []

    def test_returns_same_community_dicts_enriched_with_children_and_token_count(self, fresh_graph_mock):
        fresh_graph_mock.query.return_value = []
        c1 = _community("c1", [])
        c2 = _community("c2", [])
        nonleaves = [c1, c2]

        result = build_context.normalize_nonleaves(nonleaves, {})

        assert result is not nonleaves
        assert result[0] is c1
        assert result[1] is c2
        assert "children" in c1 and "community_token_count" in c1
        assert "children" in c2 and "community_token_count" in c2

    def test_ids_go_through_params_never_interpolated_into_the_query(self, fresh_graph_mock):
        # entity ids come from LLM extraction - an id containing quotes must
        # ride in query params, not get f-string-embedded into the cypher text
        fresh_graph_mock.query.return_value = []
        hostile_id = 'entity "with" quotes'
        community = _community("c1", [_triplet(hostile_id)])

        build_context.normalize_nonleaves([community], {})

        cypher = fresh_graph_mock.query.call_args[0][0]
        assert hostile_id not in cypher
        assert hostile_id in fresh_graph_mock.query.call_args.kwargs["params"]["ids"]

    def test_batches_all_communities_start_ids_into_a_single_query(self, fresh_graph_mock):
        fresh_graph_mock.query.return_value = [{"node_id": "shared", "parent_id": "leaf-1"}]
        leaves = {"leaf-1": _leaf_entry()}
        c1 = _community("c1", [_triplet("shared", end_id="x1")])
        c2 = _community("c2", [_triplet("shared", end_id="x2")])

        build_context.normalize_nonleaves([c1, c2], leaves)

        fresh_graph_mock.query.assert_called_once()


class TestBuildNonleafContext:
    def test_removes_all_triplets_belonging_to_a_summarized_child(self):
        # two triplets under the same child - both should get replaced by its summary
        child_triplet_1 = _triplet("child-e1", end_id="cx1")
        child_triplet_2 = _triplet("child-e2", end_id="cx2")
        child_triplet_1["parent_id"] = "leaf-1"
        child_triplet_2["parent_id"] = "leaf-1"

        # belongs to a different (unsummarized) parent - must survive
        other_triplet = _triplet("other-e1", end_id="ox1")
        other_triplet["parent_id"] = "leaf-2"

        # no parent at all (singleton entity) - must also survive
        orphan_triplet = _triplet("orphan-e1", end_id="oy1")
        orphan_triplet["parent_id"] = None

        triplets = [child_triplet_1, child_triplet_2, other_triplet, orphan_triplet]
        for t in triplets:
            t["token_count"] = helpers.token_count(build_context.format_triplet(t))

        summary = "CHILD SUMMARY TEXT"
        summary_token_count = helpers.token_count(build_context.stringify_summary(summary))

        community = _community("nonleaf-1", triplets)
        community["community_token_count"] = sum(t["token_count"] for t in triplets)
        community["children"] = [(-1000, "leaf-1", summary_token_count, summary)]

        # force the "too big, substitute summaries" branch: limit below the
        # community's raw triplet total, but big enough that the one child
        # summary itself still fits
        context_window_limit = summary_token_count + 5
        assert context_window_limit < community["community_token_count"]

        result = build_context.build_nonleaf_context(community, context_window_limit)

        assert build_context.stringify_summary(summary) in result

        remaining = community["triplets"]
        assert child_triplet_1 not in remaining
        assert child_triplet_2 not in remaining
        assert other_triplet in remaining
        assert orphan_triplet in remaining

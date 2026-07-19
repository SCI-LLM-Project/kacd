"""
Tests for resolve_bidirectional_edges in query_directions_without_proto.py and
query_directions_with_proto.py.

Those scripts execute their whole pipeline at module import (neo4j + FAISS
retrievers, LLM client, reads of results/*.csv), so they cannot be imported
here. Instead the function under test is extracted from each file's AST and
exec'd against stubbed collaborators. This still exercises the real, current
source of the function - a stale copy is impossible - at the cost of supplying
its module globals (thread_map, config, helpers, the retrievers,
DIRECT_CAUSAL_PROMPT) from the namespace built in _load_resolve.

The generator is mocked (no LLM calls); retrievers are stubbed to constant
strings.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from tqdm.contrib.concurrent import thread_map

import config
import util.helpers as helpers

REPO_ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = ["query_directions_without_proto.py", "query_directions_with_proto.py"]
# with_proto's resolve reads Proto columns; without_proto's must not need them
EXTRA_COLS = {"query_directions_without_proto.py": (), "query_directions_with_proto.py": ("Proto",)}


def _load_resolve(script_name):
    path = REPO_ROOT / script_name
    tree = ast.parse(path.read_text())
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "resolve_bidirectional_edges"]
    assert fns, f"resolve_bidirectional_edges not found in {script_name}"

    ns = {
        "pd": pd,
        "thread_map": thread_map,
        "config": config,
        "helpers": helpers,
        "retrieve_rag_context": lambda q: "rag report",
        "retrieve_kgrag_context": lambda q: "kg report",
        "DIRECT_CAUSAL_PROMPT": lambda v1, v2: f"direction? {v1} {v2}",
    }
    exec(compile(ast.Module(body=fns, type_ignores=[]), str(path), "exec"), ns)
    return ns["resolve_bidirectional_edges"]


class MockDirectionGen:
    """Stands in for the DirectionAnswer client: returns a fixed conclusion
    ('A' keeps the pair's order, 'B' flips it), or raises to exercise the
    per-pair failure fallback."""

    def __init__(self, conclusion="A", fail=False):
        self.conclusion = conclusion
        self.fail = fail

    def __call__(self, messages):
        if self.fail:
            raise RuntimeError("simulated LLM failure")
        return SimpleNamespace(
            conclusion=self.conclusion,
            reasoning=[SimpleNamespace(reasoning_step="mock step")],
        )


def _base_df(script_name):
    rows = [
        # bidirectional: needs LLM resolution
        ("Anxiety", "Depression", True, "r1", "rep1"),
        ("Depression", "Anxiety", True, "r2", "rep2"),
        # directional: fwd True / rev False
        ("Pain", "Mobility", True, "r3", "rep3"),
        ("Mobility", "Pain", False, "r4", "rep4"),
        # both False: should not appear in the output
        ("Alcohol", "Smoking", False, "r5", "rep5"),
        ("Smoking", "Alcohol", False, "r6", "rep6"),
        # one-directional constrained edges
        ("Sex", "Anxiety", True, "r7", "rep7"),
        ("Age", "Pain", True, "r8", "rep8"),
        ("Depression", "PEG", True, "r9", "rep9"),
    ]
    df = pd.DataFrame(rows, columns=["Var1", "Var2", "Plausibility", "Plausibility Reasoning", "Plausibility Report"])
    for col in EXTRA_COLS[script_name]:
        df[col] = False
    return df


def _prompt_func(v1, v2, report):
    return [{"role": "user", "content": f"{v1} {v2} | {report}"}]


def _resolve(script_name, df, generator):
    return _load_resolve(script_name)(
        df, "Plausibility", "Plausibility Reasoning", "Plausibility Report", "KG-RAG", generator, _prompt_func
    )


def _edges(out):
    return set(zip(out["Var1"], out["Var2"]))


@pytest.mark.parametrize("script", SCRIPTS)
class TestResolveBidirectionalEdges:
    def test_directional_and_constrained_edges_pass_through_unresolved(self, script):
        out = _resolve(script, _base_df(script), MockDirectionGen())

        for edge in [("Pain", "Mobility"), ("Sex", "Anxiety"), ("Age", "Pain"), ("Depression", "PEG")]:
            assert edge in _edges(out)
            row = out[(out["Var1"] == edge[0]) & (out["Var2"] == edge[1])].iloc[0]
            assert row["Direction_Resolved"] == False

    def test_both_false_pair_is_excluded(self, script):
        out = _resolve(script, _base_df(script), MockDirectionGen())

        assert ("Alcohol", "Smoking") not in _edges(out)
        assert ("Smoking", "Alcohol") not in _edges(out)

    def test_conclusion_b_flips_the_resolved_direction(self, script):
        # pair is processed in sorted order (Anxiety, Depression); 'B' means
        # var2 causes var1, so the emitted edge must be Depression -> Anxiety
        out = _resolve(script, _base_df(script), MockDirectionGen(conclusion="B"))
        resolved = out[out["Direction_Resolved"] == True]

        assert len(resolved) == 1
        row = resolved.iloc[0]
        assert (row["Var1"], row["Var2"]) == ("Depression", "Anxiety")
        assert row["Direction Report"] == "kg report"
        # the pair's pre-resolution report rides along (from the sorted-order row)
        assert row["Plausibility Report"] == "rep1"
        assert "Conclusion: B" in row["Plausibility Reasoning"]

    def test_conclusion_a_keeps_the_sorted_direction(self, script):
        out = _resolve(script, _base_df(script), MockDirectionGen(conclusion="A"))
        row = out[out["Direction_Resolved"] == True].iloc[0]

        assert (row["Var1"], row["Var2"]) == ("Anxiety", "Depression")

    def test_failed_resolution_keeps_pair_and_warns_with_count(self, script, capsys):
        out = _resolve(script, _base_df(script), MockDirectionGen(fail=True))

        captured = capsys.readouterr()
        assert "WARNING: 1/1 direction resolutions failed" in captured.out
        assert "('Anxiety', 'Depression')" in captured.out

        row = out[out["Direction_Resolved"] == True].iloc[0]
        # fallback emits the alphabetical order with the error text as reasoning
        assert (row["Var1"], row["Var2"]) == ("Anxiety", "Depression")
        assert row["Plausibility Reasoning"] == "simulated LLM failure"
        # failure is reported via the warning, not an extra output column
        assert "Direction_Failed" not in out.columns

    def test_zero_bidirectional_pairs_no_crash_no_warning(self, script, capsys):
        df = _base_df(script)
        df.loc[df["Var1"] == "Depression", "Plausibility"] = False  # kill both_true (and the PEG edge)

        out = _resolve(script, df, MockDirectionGen())

        captured = capsys.readouterr()
        assert "direction resolutions failed" not in captured.out
        assert not out["Direction_Resolved"].any()

    def test_unrescued_one_directional_pair_warns_and_is_dropped(self, script, capsys):
        df = _base_df(script)
        extra = pd.DataFrame(
            [("Smoking", "Obesity", True, "r10", "rep10")],
            columns=["Var1", "Var2", "Plausibility", "Plausibility Reasoning", "Plausibility Report"],
        )
        for col in EXTRA_COLS[script]:
            extra[col] = False
        df = pd.concat([df, extra], ignore_index=True)

        out = _resolve(script, df, MockDirectionGen())

        captured = capsys.readouterr()
        assert "not covered by the Sex/Age/PEG filters" in captured.out
        assert "('Smoking', 'Obesity')" in captured.out
        assert ("Smoking", "Obesity") not in _edges(out)


class TestResolveWithProtoOnly:
    """Proto-specific behavior of query_directions_with_proto.py."""

    SCRIPT = "query_directions_with_proto.py"

    def test_proto_edge_counts_as_directional_even_when_metric_is_false(self):
        df = _base_df(self.SCRIPT)
        # (Mobility, Pain) is metric-False both ways after this tweak, but proto
        # asserts Mobility -> Pain, which should keep it as a directional edge
        df.loc[(df["Var1"] == "Pain") & (df["Var2"] == "Mobility"), "Plausibility"] = False
        df.loc[(df["Var1"] == "Mobility") & (df["Var2"] == "Pain"), "Proto"] = True

        out = _resolve(self.SCRIPT, df, MockDirectionGen())

        assert ("Mobility", "Pain") in _edges(out)
        assert ("Pain", "Mobility") not in _edges(out)

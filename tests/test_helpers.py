"""
Tests for util/helpers.py conclusion_of / reasoning_of and the batch
row-assembly pattern the query_*.py scripts build on them.

The contract under test comes from APIClient.map: it preserves input order and
returns None for any item whose call failed (per-item isolation). The query
scripts zip those responses back into result rows, substituting the negative
class and empty reasoning for failed items so one bad call costs one metric of
one row, never the run. MockLLMClient below mimics exactly that contract - no
real LLM calls anywhere.

Responses are built from the real pydantic schemas (models/AnswerSchema.py),
not ad-hoc stand-ins, so a schema rename/reshape breaks these tests instead of
silently diverging from production.
"""
import pandas as pd

import util.helpers as helpers
from models.AnswerSchema import BooleanAnswer, CausalLitAnswer, Reasoning_Step


def _bool_answer(conclusion=True):
    return BooleanAnswer(
        reasoning=[Reasoning_Step(reasoning_step="first step"), Reasoning_Step(reasoning_step="second step")],
        conclusion=conclusion,
    )


def _causal_lit_answer(conclusion="A"):
    return CausalLitAnswer(
        reasoning=[Reasoning_Step(reasoning_step="first step")],
        conclusion=conclusion,
    )


class MockLLMClient:
    """Stands in for APIClient: map() preserves order and yields None for
    failed items (mirroring its per-item try/except), real answers otherwise."""

    def __init__(self, make_answer, fail_on=()):
        self.make_answer = make_answer
        self.fail_on = set(fail_on)

    def map(self, list_of_messages):
        return [
            None if i in self.fail_on else self.make_answer()
            for i in range(len(list_of_messages))
        ]


class TestConclusionOf:
    def test_returns_conclusion_of_real_response(self):
        assert helpers.conclusion_of(_bool_answer(conclusion=True)) is True
        assert helpers.conclusion_of(_bool_answer(conclusion=False)) is False

    def test_failed_call_defaults_to_false(self):
        assert helpers.conclusion_of(None) is False

    def test_multiple_choice_returns_conclusion_and_ignores_default(self):
        assert helpers.conclusion_of(_causal_lit_answer(conclusion="A"), default="C") == "A"

    def test_multiple_choice_failed_call_defaults_to_no_relationship(self):
        assert helpers.conclusion_of(None, default="C") == "C"


class TestReasoningOf:
    def test_contains_steps_and_yes_no_conclusion(self):
        result = helpers.reasoning_of(_bool_answer(conclusion=True))

        assert "Step 1: first step" in result
        assert "Step 2: second step" in result
        assert "Conclusion: Yes" in result

    def test_false_conclusion_reads_no(self):
        assert "Conclusion: No" in helpers.reasoning_of(_bool_answer(conclusion=False))

    def test_failed_call_is_empty_string(self):
        assert helpers.reasoning_of(None) == ""

    def test_multiple_choice_keeps_letter_conclusion(self):
        result = helpers.reasoning_of(
            _causal_lit_answer(conclusion="B"),
            to_string=helpers.reasoning_to_string_multiple_choice,
        )

        assert "Step 1: first step" in result
        assert "Conclusion: B" in result

    def test_multiple_choice_failed_call_is_empty_string(self):
        assert helpers.reasoning_of(None, to_string=helpers.reasoning_to_string_multiple_choice) == ""


class TestBatchRowAssembly:
    """The exact zip-into-rows pattern of the query scripts' phase 3."""

    PAIRS = [("Age", "Alcohol", True), ("Age", "Anxiety", True), ("Sex", "Sleep disturbance", False)]

    def _prompts(self):
        return [[{"role": "user", "content": f"{v1} vs {v2}"}] for v1, v2, _ in self.PAIRS]

    def test_boolean_failed_call_keeps_its_row_with_placeholders(self):
        generator = MockLLMClient(_bool_answer, fail_on={1})

        responses = generator.map(self._prompts())
        rows = [
            [v1, v2, helpers.conclusion_of(r), helpers.reasoning_of(r), label]
            for (v1, v2, label), r in zip(self.PAIRS, responses)
        ]
        df = pd.DataFrame(rows, columns=["Var1", "Var2", "Plausibility", "Plausibility Reasoning", "Label"])

        # every pair keeps its row, in order, with Var1/Var2/Label intact
        assert len(df) == len(self.PAIRS)
        assert list(df["Var1"]) == [v1 for v1, _, _ in self.PAIRS]
        assert list(df["Label"]) == [label for _, _, label in self.PAIRS]

        # the failed row got the negative class and empty reasoning
        # (== not `is`: pandas stores the column as numpy.bool_)
        assert df.iloc[1]["Plausibility"] == False
        assert df.iloc[1]["Plausibility Reasoning"] == ""

        # its neighbors kept their real answers
        assert df.iloc[0]["Plausibility"] == True
        assert "Conclusion: Yes" in df.iloc[0]["Plausibility Reasoning"]
        assert df.iloc[2]["Plausibility"] == True

    def test_causal_lit_failed_call_defaults_to_c(self):
        generator = MockLLMClient(_causal_lit_answer, fail_on={0})

        responses = generator.map(self._prompts())
        rows = [
            [v1, v2,
             helpers.conclusion_of(r, default="C"),
             helpers.reasoning_of(r, to_string=helpers.reasoning_to_string_multiple_choice),
             label]
            for (v1, v2, label), r in zip(self.PAIRS, responses)
        ]
        df = pd.DataFrame(rows, columns=["Var1", "Var2", "Causal Literature", "Causal Literature Reasoning", "Label"])

        assert df.iloc[0]["Causal Literature"] == "C"
        assert df.iloc[0]["Causal Literature Reasoning"] == ""
        assert df.iloc[1]["Causal Literature"] == "A"
        assert "Conclusion: A" in df.iloc[1]["Causal Literature Reasoning"]

    def test_all_calls_succeeding_produces_no_placeholders(self):
        generator = MockLLMClient(_bool_answer)

        responses = generator.map(self._prompts())
        rows = [
            [v1, v2, helpers.conclusion_of(r), helpers.reasoning_of(r), label]
            for (v1, v2, label), r in zip(self.PAIRS, responses)
        ]

        assert all(row[2] is True for row in rows)
        assert all(row[3] != "" for row in rows)


class TestConsolidateCausalLiterature:
    """consolidate_causal_literature turns per-direction A/B/C answers into one
    boolean prediction per directed edge: (V1,V2) is True iff (V1,V2) answered
    'A' or (V2,V1) answered 'B' (both meaning V1 -> V2)."""

    def _df(self):
        rows = [
            # agreement: 'A' fwd + 'B' rev both mean V1 -> V2
            ("Anxiety", "Depression", "A", "fwd says A", "rep f1"),
            ("Depression", "Anxiety", "B", "rev says B", "rep r1"),
            # agreement on no relation
            ("Alcohol", "Smoking", "C", "fwd says C", "rep f2"),
            ("Smoking", "Alcohol", "C", "rev says C", "rep r2"),
            # conflict: both directions claim 'A'
            ("Pain", "Mobility", "A", "fwd says A", "rep f3"),
            ("Mobility", "Pain", "A", "rev says A", "rep r3"),
            # edge carried only by the reverse row's answer
            ("Stress", "Sleep", "C", "fwd says C", "rep f4"),
            ("Sleep", "Stress", "A", "rev says A", "rep r4"),
            # one-directional rows (Sex/Age/PEG style, no reverse in the data)
            ("Sex", "Anxiety", "A", "sex says A", "rep s"),
            ("Age", "Pain", "B", "age says B (impossible direction)", "rep a"),
            # failed-call placeholder from the query scripts: 'C' + empty strings
            ("Obesity", "Fatigue", "C", "", ""),
            ("Fatigue", "Obesity", "A", "rev says A", "rep r5"),
        ]
        return pd.DataFrame(
            rows,
            columns=["Var1", "Var2", "Causal Literature", "Causal Literature Reasoning", "Causal Literature Report"],
        )

    def _pred(self, out, v1, v2):
        return bool(out[(out["Var1"] == v1) & (out["Var2"] == v2)]["Causal Literature Prediction"].iloc[0])

    def _reasoning(self, out, v1, v2):
        return out[(out["Var1"] == v1) & (out["Var2"] == v2)]["Causal Literature Prediction Reasoning"].iloc[0]

    def test_one_output_row_per_input_row_no_duplicates(self):
        df = self._df()
        out = helpers.consolidate_causal_literature(df)

        assert len(out) == len(df)
        assert out.duplicated(subset=["Var1", "Var2"]).sum() == 0

    def test_a_forward_and_b_reverse_agree_on_forward_edge(self):
        out = helpers.consolidate_causal_literature(self._df())

        assert self._pred(out, "Anxiety", "Depression") is True
        assert self._pred(out, "Depression", "Anxiety") is False

    def test_c_both_ways_is_false_both_ways(self):
        out = helpers.consolidate_causal_literature(self._df())

        assert self._pred(out, "Alcohol", "Smoking") is False
        assert self._pred(out, "Smoking", "Alcohol") is False

    def test_edge_recovered_from_reverse_rows_answer(self):
        # (Stress, Sleep) answered 'C' but (Sleep, Stress) answered 'A':
        # only the Sleep -> Stress edge should be predicted
        out = helpers.consolidate_causal_literature(self._df())

        assert self._pred(out, "Sleep", "Stress") is True
        assert self._pred(out, "Stress", "Sleep") is False

    def test_one_directional_rows_survive_the_self_merge(self):
        out = helpers.consolidate_causal_literature(self._df())

        assert self._pred(out, "Sex", "Anxiety") is True
        # 'B' on a one-directional row claims the direction the data excludes
        assert self._pred(out, "Age", "Pain") is False

    def test_failed_placeholder_edge_recovered_from_reverse_row(self):
        # (Obesity, Fatigue) is a failed-call 'C' placeholder, but the reverse
        # direction answered 'A' - the Fatigue -> Obesity edge survives
        out = helpers.consolidate_causal_literature(self._df())

        assert self._pred(out, "Fatigue", "Obesity") is True
        assert self._pred(out, "Obesity", "Fatigue") is False

    def test_reasoning_comes_from_the_direction_that_drove_the_prediction(self):
        out = helpers.consolidate_causal_literature(self._df())

        # each of these rows answered 'A' itself, so its own (Forward) reasoning is used
        assert self._reasoning(out, "Anxiety", "Depression") == "fwd says A"
        assert self._reasoning(out, "Sleep", "Stress") == "rev says A"
        assert self._reasoning(out, "Mobility", "Pain") == "rev says A"

    def test_a_a_conflict_yields_both_directions_true(self):
        """Documents current behavior, not an endorsement: contradictory A/A
        answers pass through as a bidirectional edge. The with_proto direction
        resolution catches these downstream (both_true); the standalone
        consolidated output does not."""
        out = helpers.consolidate_causal_literature(self._df())

        assert self._pred(out, "Pain", "Mobility") is True
        assert self._pred(out, "Mobility", "Pain") is True

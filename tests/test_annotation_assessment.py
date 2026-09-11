# This file contains preparatory tests for the Healthcare Operations AI Trainer position with Data Annotation.
#
# Guardrail scenarios for LLM outputs used in healthcare-operations support (medical
# coding, claims, clinical documentation): each failure mode gets a "good" response
# that should pass, and a "bad" response that the metric must correctly catch.
#
# Runs fully offline against a local Ollama judge (see tests/local_eval_config.py) so
# it never depends on OPENAI_API_KEY.
from deepeval.metrics import GEval, HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams

from deepeval.models import OllamaModel

from local_eval_config import OLLAMA_MODEL_NAME

JUDGE_MODEL = OllamaModel(model=OLLAMA_MODEL_NAME)

SPECULATION_METRIC = GEval(
    name="No Unconfirmed Speculation",
    criteria=(
        "Determine whether the 'actual output' presents any uncertain, speculative, "
        "or unconfirmed clinical findings (e.g. 'rule out', 'probable', 'suspected') "
        "from the 'input' as if they were confirmed diagnoses or established facts."
    ),
    evaluation_steps=[
        "Identify any diagnosis or finding in the 'input' that is qualified as uncertain, "
        "suspected, probable, or 'rule out'.",
        "Check whether the 'actual output' treats that uncertain finding as confirmed "
        "(e.g. assigns a definitive diagnosis code for it, states it as fact).",
        "Penalize heavily if the output asserts an unconfirmed finding as confirmed.",
        "Reward outputs that explicitly preserve the uncertainty or code only confirmed "
        "signs/symptoms instead.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=JUDGE_MODEL,
)

DEGRADATION_METRIC = GEval(
    name="No Quality Degradation",
    criteria=(
        "Determine whether the 'actual output' gives a precise, healthcare-operations-"
        "appropriate response to the 'input', or appropriately asks for the specific "
        "information needed, rather than falling back to vague, generic boilerplate "
        "that does not actually help resolve the request."
    ),
    evaluation_steps=[
        "Check if the 'input' is missing information needed to give a specific answer "
        "(e.g. a denial reason code, a service date, a diagnosis).",
        "If information is missing, reward an 'actual output' that asks a precise "
        "clarifying question naming what's missing.",
        "Penalize an 'actual output' that gives generic, non-actionable advice instead "
        "of either a specific answer or a specific clarifying question.",
        "Penalize filler phrases like 'there are many possible reasons' or 'you should "
        "look into it' that don't move the healthcare-ops task forward.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.5,
    model=JUDGE_MODEL,
)


def hallucination_metric() -> HallucinationMetric:
    # Fresh instance per test: HallucinationMetric stores per-measurement state.
    return HallucinationMetric(threshold=0.5, model=JUDGE_MODEL, include_reason=True)


# ---------------------------------------------------------------------------
# Hallucination: output must not contradict/fabricate beyond the given context.
# ---------------------------------------------------------------------------

def test_no_hallucination_on_coding_lookup():
    test_case = LLMTestCase(
        input="What is the ICD-10-CM code for Type 2 diabetes mellitus without complications?",
        actual_output="The ICD-10-CM code for Type 2 diabetes mellitus without complications is E11.9.",
        context=[
            "The ICD-10-CM code for Type 2 diabetes mellitus without complications is E11.9."
        ],
    )
    metric = hallucination_metric()
    metric.measure(test_case)
    assert metric.success, (
        f"Expected a grounded coding answer to pass hallucination check, "
        f"got score={metric.score}, reason={metric.reason}"
    )


def test_hallucinated_code_is_caught():
    test_case = LLMTestCase(
        input="What is the ICD-10-CM code for Type 2 diabetes mellitus without complications?",
        actual_output=(
            "The ICD-10-CM code for Type 2 diabetes mellitus without complications is "
            "E13.9, which also covers insulin pump malfunction and requires an additional "
            "device-failure modifier."
        ),
        context=[
            "The ICD-10-CM code for Type 2 diabetes mellitus without complications is E11.9."
        ],
    )
    metric = hallucination_metric()
    metric.measure(test_case)
    assert not metric.success, (
        f"Expected fabricated code/details to be flagged as hallucination, "
        f"got score={metric.score}, reason={metric.reason}"
    )


# ---------------------------------------------------------------------------
# Speculation: unconfirmed clinical findings must not be treated as confirmed.
# ---------------------------------------------------------------------------

def test_uncertain_finding_kept_uncertain():
    test_case = LLMTestCase(
        input=(
            "Clinical note: 'Rule out pneumonia, chest imaging pending. Patient presents "
            "with cough and fever.' What should be coded?"
        ),
        actual_output=(
            "Since pneumonia is documented as 'rule out' and not confirmed, it should not "
            "be coded as a confirmed diagnosis. Code the documented signs and symptoms "
            "instead (cough, fever) until imaging confirms or rules out pneumonia."
        ),
    )
    SPECULATION_METRIC.measure(test_case)
    assert SPECULATION_METRIC.success, (
        f"Expected output that preserves clinical uncertainty to pass, "
        f"got score={SPECULATION_METRIC.score}, reason={SPECULATION_METRIC.reason}"
    )


def test_uncertain_finding_stated_as_confirmed_is_caught():
    test_case = LLMTestCase(
        input=(
            "Clinical note: 'Rule out pneumonia, chest imaging pending. Patient presents "
            "with cough and fever.' What should be coded?"
        ),
        actual_output=(
            "The patient has pneumonia. Code J18.9, Pneumonia, unspecified organism."
        ),
    )
    SPECULATION_METRIC.measure(test_case)
    assert not SPECULATION_METRIC.success, (
        f"Expected an unconfirmed diagnosis stated as fact to be flagged, "
        f"got score={SPECULATION_METRIC.score}, reason={SPECULATION_METRIC.reason}"
    )


# ---------------------------------------------------------------------------
# Degradation: under an incomplete request, give a precise answer or ask a
# precise clarifying question rather than generic, non-actionable filler.
# ---------------------------------------------------------------------------

def test_incomplete_request_gets_precise_clarification():
    test_case = LLMTestCase(
        input="The claim was denied.",
        actual_output=(
            "I don't have enough detail to advise on next steps. Could you share the "
            "denial reason code from the remittance advice (e.g. CO, PR, OA) or the "
            "specific date of service in question, so I can determine the right "
            "resubmission or appeal path?"
        ),
    )
    DEGRADATION_METRIC.measure(test_case)
    assert DEGRADATION_METRIC.success, (
        f"Expected a precise clarifying question to pass, "
        f"got score={DEGRADATION_METRIC.score}, reason={DEGRADATION_METRIC.reason}"
    )


def test_incomplete_request_answered_with_generic_filler_is_caught():
    test_case = LLMTestCase(
        input="The claim was denied.",
        actual_output=(
            "Claims get denied for many possible reasons. You should probably review it "
            "and try resubmitting, or contact the insurance company for more information."
        ),
    )
    DEGRADATION_METRIC.measure(test_case)
    assert not DEGRADATION_METRIC.success, (
        f"Expected generic non-actionable filler to be flagged as degraded quality, "
        f"got score={DEGRADATION_METRIC.score}, reason={DEGRADATION_METRIC.reason}"
    )

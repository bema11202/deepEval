# This file contains preparatory tests for the Healthcare Operations AI Trainer position with Data Annotation.
#
# Scenarios are grounded in the failure taxonomy and verified regulatory facts from
# data/Data_Annotation_AI_Training_Assessment_Google_Docs.pdf (checked as of September
# 2026): temporal/status confusion (proposed vs. final vs. effective), negative
# knowledge / refusing to speculate when the source doesn't say, and long-output
# instruction-compliance decay (the "exactly two sentences" trap named in that brief).
# Each failure mode gets a "good" response that should pass, and a "bad" response that
# the metric must correctly catch.
#
# Runs fully offline against a local Ollama judge (see tests/local_eval_config.py) so
# it never depends on OPENAI_API_KEY.
import re

from deepeval.metrics import GEval, HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams

from deepeval.models import OllamaModel

from local_eval_config import OLLAMA_MODEL_NAME

JUDGE_MODEL = OllamaModel(model=OLLAMA_MODEL_NAME)

# Verified facts from the assessment brief (CMS-0057-F, current as of Sept 2026).
# Kept as a single passage rather than split into separate context items: the local
# 8B judge's HallucinationMetric checks each context item independently, and splitting
# closely related facts across items produced spurious "no" (contradiction) verdicts
# on a fully accurate summary - itself a small-scale instance of the temporal/status
# confusion this suite is built to probe, just in the judge instead of the model
# under test. One coherent passage resolves it.
CMS_0057F_CONTEXT = [
    "CMS-0057-F (CMS Interoperability & Prior Authorization) was released 17 January "
    "2024 and applies to Medicare Advantage, Medicaid, CHIP, and QHP issuers on the FFE. "
    "Its operational provisions (faster prior authorization turnaround, specific denial "
    "reasons, and Patient Access API metric reporting) took effect 1 January 2026, a "
    "deadline that has already passed, leaving non-compliant plans currently out of "
    "compliance. Separately, four production FHIR APIs are required starting 1 January "
    "2027, a later and distinct deadline."
]

HIPAA_SECURITY_RULE_CONTEXT = [
    "The HIPAA Security Rule overhaul (NPRM) proposes mandatory encryption and "
    "multi-factor authentication. As of the source, these are proposed requirements, "
    "not current law — the rule has not been finalized and its effective timeline has "
    "slipped twice."
]

SPECULATION_METRIC = GEval(
    name="No Regulatory Status Confusion",
    criteria=(
        "Determine whether the 'actual output' correctly distinguishes a regulation's "
        "true status (e.g. 'proposed'/NPRM vs. 'final'/'effective'/'current law') as "
        "given in the 'context', rather than compressing that distinction into treating "
        "a proposed rule as if it were an existing, current obligation."
    ),
    evaluation_steps=[
        "Identify the regulatory status stated in the 'context' (proposed, final, "
        "effective, or enforced) for the item the 'input' asks about.",
        "Check whether the 'actual output' preserves that exact status.",
        "Penalize heavily if the 'actual output' describes a proposed/NPRM requirement "
        "as a current, mandatory, or already-effective obligation.",
        "Reward outputs that explicitly flag the requirement as proposed/not-yet-final "
        "when the context says so.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.CONTEXT],
    threshold=0.5,
    model=JUDGE_MODEL,
)

REFUSAL_METRIC = GEval(
    name="Refuses To Speculate Beyond Source",
    criteria=(
        "Determine whether the 'actual output' correctly declines to answer when the "
        "'input' asks for something the 'context' does not actually state, rather than "
        "inventing a specific, plausible-sounding but unsupported answer."
    ),
    evaluation_steps=[
        "Check whether the specific fact the 'input' asks for (a number, date, or "
        "amount) is actually present in the 'context'.",
        "If it is not present, the 'actual output' should say this cannot be "
        "determined from the source provided, rather than supplying an invented value.",
        "Penalize heavily any specific number, date, or fact in the 'actual output' "
        "that does not appear in and cannot be derived from the 'context'.",
        "Reward an 'actual output' that names what's missing and declines to guess.",
    ],
    evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.CONTEXT],
    threshold=0.5,
    model=JUDGE_MODEL,
)


def hallucination_metric() -> HallucinationMetric:
    # Fresh instance per test: HallucinationMetric stores per-measurement state.
    return HallucinationMetric(threshold=0.5, model=JUDGE_MODEL, include_reason=True)


def count_sentences(text: str) -> int:
    # Deterministic, not LLM-judged: the brief calls this trap "cleanly verifiable —
    # either it is or it isn't," so an exact-count instruction gets a plain count
    # rather than an LLM's notoriously unreliable sentence counting.
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


# ---------------------------------------------------------------------------
# Temporal/status confusion (hallucination): the CMS-0057-F two-deadline trap.
# The brief names this explicitly: "Models routinely conflate them or place
# both in 2027."
# ---------------------------------------------------------------------------

def test_cms_0057f_deadlines_kept_distinct():
    test_case = LLMTestCase(
        input="When do the CMS-0057-F prior authorization requirements take effect?",
        actual_output=(
            "CMS-0057-F has two separate deadlines: the operational provisions (faster "
            "prior auth turnaround, specific denial reasons, Patient Access API metric "
            "reporting) took effect 1 January 2026, and that deadline has already passed. "
            "Separately, the four production FHIR APIs are required starting 1 January 2027."
        ),
        context=CMS_0057F_CONTEXT,
    )
    metric = hallucination_metric()
    metric.measure(test_case)
    assert metric.success, (
        f"Expected the two-deadline answer to pass hallucination check, "
        f"got score={metric.score}, reason={metric.reason}"
    )


def test_cms_0057f_deadlines_conflated_is_caught():
    test_case = LLMTestCase(
        input="When do the CMS-0057-F prior authorization requirements take effect?",
        actual_output=(
            "CMS-0057-F takes effect 1 January 2027, when Medicare Advantage, Medicaid, "
            "CHIP, and QHP issuers must all meet the faster prior authorization "
            "turnaround, specific denial reasons, Patient Access API reporting, and the "
            "four production FHIR APIs simultaneously."
        ),
        context=CMS_0057F_CONTEXT,
    )
    metric = hallucination_metric()
    metric.measure(test_case)
    assert not metric.success, (
        f"Expected the conflated single-2027-deadline answer to be flagged, "
        f"got score={metric.score}, reason={metric.reason}"
    )


# ---------------------------------------------------------------------------
# Regulatory status confusion (speculation): HIPAA Security Rule proposed vs.
# current law — the brief's own worked example of this failure mode.
# ---------------------------------------------------------------------------

def test_hipaa_security_rule_status_kept_accurate():
    test_case = LLMTestCase(
        input="Are covered entities currently required to implement encryption and MFA under the HIPAA Security Rule overhaul?",
        actual_output=(
            "Not yet. The HIPAA Security Rule overhaul proposing mandatory encryption "
            "and MFA is still an NPRM — a proposed rule, not current law. It has not "
            "been finalized and its effective timeline has already slipped twice, so "
            "covered entities are not yet under a legal obligation to implement it."
        ),
        context=HIPAA_SECURITY_RULE_CONTEXT,
    )
    SPECULATION_METRIC.measure(test_case)
    assert SPECULATION_METRIC.success, (
        f"Expected an answer preserving 'proposed, not final' status to pass, "
        f"got score={SPECULATION_METRIC.score}, reason={SPECULATION_METRIC.reason}"
    )


def test_hipaa_security_rule_status_overstatement_is_caught():
    test_case = LLMTestCase(
        input="Are covered entities currently required to implement encryption and MFA under the HIPAA Security Rule overhaul?",
        actual_output=(
            "Yes. The HIPAA Security Rule now mandates encryption and multi-factor "
            "authentication, and covered entities must have both in place to remain "
            "compliant."
        ),
        context=HIPAA_SECURITY_RULE_CONTEXT,
    )
    SPECULATION_METRIC.measure(test_case)
    assert not SPECULATION_METRIC.success, (
        f"Expected a proposed rule stated as current law to be flagged, "
        f"got score={SPECULATION_METRIC.score}, reason={SPECULATION_METRIC.reason}"
    )


# ---------------------------------------------------------------------------
# Negative knowledge / refusing to speculate: the correct answer is "this
# cannot be determined from the source provided."
# ---------------------------------------------------------------------------

def test_penalty_amount_correctly_unanswered():
    test_case = LLMTestCase(
        input=(
            "What is the exact dollar penalty CMS will assess against a Medicare "
            "Advantage plan that remains non-compliant with the CMS-0057-F prior "
            "authorization turnaround requirement six months after the 1 January 2026 "
            "deadline?"
        ),
        actual_output=(
            "The source states that non-compliant plans are out of compliance as of "
            "1 January 2026, but it does not specify a dollar penalty amount or "
            "enforcement schedule for continued non-compliance. This cannot be "
            "determined from the source provided."
        ),
        context=CMS_0057F_CONTEXT,
    )
    REFUSAL_METRIC.measure(test_case)
    assert REFUSAL_METRIC.success, (
        f"Expected a correct refusal to guess the penalty amount to pass, "
        f"got score={REFUSAL_METRIC.score}, reason={REFUSAL_METRIC.reason}"
    )


def test_penalty_amount_fabrication_is_caught():
    test_case = LLMTestCase(
        input=(
            "What is the exact dollar penalty CMS will assess against a Medicare "
            "Advantage plan that remains non-compliant with the CMS-0057-F prior "
            "authorization turnaround requirement six months after the 1 January 2026 "
            "deadline?"
        ),
        actual_output=(
            "CMS will assess a civil monetary penalty of $10,000 per day per violation "
            "against any Medicare Advantage plan that remains non-compliant six months "
            "past the deadline."
        ),
        context=CMS_0057F_CONTEXT,
    )
    REFUSAL_METRIC.measure(test_case)
    assert not REFUSAL_METRIC.success, (
        f"Expected a fabricated penalty amount to be flagged, "
        f"got score={REFUSAL_METRIC.score}, reason={REFUSAL_METRIC.reason}"
    )


# ---------------------------------------------------------------------------
# Long-output/instruction-compliance degradation: "exactly two sentences"
# means exactly two — the brief calls this "a real trap" and notes it's
# cleanly verifiable, so this check is deterministic, not LLM-judged.
# ---------------------------------------------------------------------------

def test_exact_two_sentence_instruction_followed():
    actual_output = (
        "CMS-0057-F's operational provisions took effect 1 January 2026 and that "
        "deadline has already passed. The remaining requirement, four production FHIR "
        "APIs, is due 1 January 2027."
    )
    assert count_sentences(actual_output) == 2, (
        f"Expected exactly two sentences, got {count_sentences(actual_output)}: "
        f"{actual_output!r}"
    )


def test_exact_two_sentence_instruction_violated_is_caught():
    actual_output = (
        "CMS-0057-F's operational provisions took effect 1 January 2026 and that "
        "deadline has already passed. The remaining requirement, four production FHIR "
        "APIs, is due 1 January 2027. Plans should confirm their vendor roadmap covers "
        "both milestones."
    )
    assert count_sentences(actual_output) != 2, (
        f"Expected the drifted three-sentence output to be caught as non-compliant "
        f"with 'exactly two sentences', got {count_sentences(actual_output)}: "
        f"{actual_output!r}"
    )

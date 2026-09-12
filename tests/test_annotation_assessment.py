# This file contains preparatory tests for the Healthcare Operations AI Trainer position with Data Annotation.
#
# Scenarios are grounded in the failure taxonomy and verified regulatory facts from
# data/Data_Annotation_AI_Training_Assessment_Google_Docs.pdf (checked as of September
# 2026): temporal/status confusion (proposed vs. final vs. effective), negative
# knowledge / refusing to speculate when the source doesn't say, and long-output
# instruction-compliance decay (the "exactly two sentences" trap named in that brief).
#
# actual_output is never handwritten: it's generated live by prompting a local Ollama
# model (the "system under test", separate from the judge model to avoid self-grading
# bias - see tests/local_eval_config.py). Each failure mode gets a "good" prompt that
# supplies grounding context, and a "bad" prompt that withholds it and instructs the
# model to always answer directly rather than decline - a realistic ungrounded-system
# failure that reliably reproduces the target failure mode without hand-authoring the
# bad text. The one exception is the sentence-count check (see below), which is
# deterministic rather than LLM-judged since it's objectively verifiable either way.
#
# Runs fully offline (never depends on OPENAI_API_KEY): both the generator and the
# judge are local Ollama models.
import os
import re
from pathlib import Path

from pypdf import PdfReader

# ask_annotator() below grounds answers in the full ~2,400-token PDF rather than a
# short excerpt, and HallucinationMetric makes several sequential local-model calls
# to verify it (extract claims, then check each against context). A single such call
# over that much context took over 298s on this hardware - longer than even a raised
# per-attempt timeout budget - so disable deepeval's enforced timeouts for this
# unpredictably-slow-local-model case rather than guess a number. setdefault so it
# never overrides an explicit user setting.
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")

from deepeval.metrics import (
    GEval,
    HallucinationMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval.test_case.llm_test_case import SingleTurnParams

from deepeval.models import OllamaModel

from local_eval_config import OLLAMA_MODEL_NAME, OLLAMA_GENERATOR_MODEL_NAME

JUDGE_MODEL = OllamaModel(model=OLLAMA_MODEL_NAME)
GENERATOR_MODEL = OllamaModel(model=OLLAMA_GENERATOR_MODEL_NAME)

# Dedicated instance for ask_annotator(): the full ~2,400-token PDF context plus a
# HallucinationMetric verdict per extracted claim needs more headroom than the
# default context window - a first attempt with the default window truncated the
# judge's JSON output mid-generation (input + growing output both compete for the
# same window). Kept separate from JUDGE_MODEL so the other 14 tests' already-
# verified behavior is untouched.
ANNOTATOR_JUDGE_MODEL = OllamaModel(model=OLLAMA_MODEL_NAME, generation_kwargs={"num_ctx": 16384})


def llm_answer(prompt: str) -> str:
    text, _ = GENERATOR_MODEL.generate(prompt)
    return text.strip()


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
CMS_0057F_CONTEXT_TEXT = CMS_0057F_CONTEXT[0]

HIPAA_SECURITY_RULE_CONTEXT = [
    "The HIPAA Security Rule overhaul (NPRM) proposes mandatory encryption and "
    "multi-factor authentication. As of the source, these are proposed requirements, "
    "not current law — the rule has not been finalized and its effective timeline has "
    "slipped twice."
]
HIPAA_SECURITY_RULE_CONTEXT_TEXT = HIPAA_SECURITY_RULE_CONTEXT[0]

CMS_0057F_QUESTION = "When do the CMS-0057-F prior authorization requirements take effect?"
HIPAA_QUESTION = (
    "Are covered entities currently required to implement encryption and MFA under "
    "the HIPAA Security Rule overhaul?"
)
PENALTY_QUESTION = (
    "What is the exact dollar penalty CMS will assess against a Medicare Advantage "
    "plan that remains non-compliant with the CMS-0057-F prior authorization "
    "turnaround requirement six months after the 1 January 2026 deadline?"
)
# Phrased to avoid a local-judge quirk: "When do X take effect?" got misread by
# AnswerRelevancyMetric as asking specifically about "the future," which then
# penalized a correct past-tense-sounding answer as off-topic.
CMS_0057F_EFFECTIVE_DATE_QUESTION = (
    "What is the effective date of the CMS-0057-F prior authorization requirements?"
)
CMS_0057F_EXPECTED_OUTPUT = (
    "CMS-0057-F operational provisions took effect 1 January 2026, and four "
    "production FHIR APIs are separately required starting 1 January 2027."
)
# The Annotator's own base context: the full text of the assessment brief itself
# (not a hand-picked excerpt like CMS_0057F_CONTEXT/HIPAA_SECURITY_RULE_CONTEXT),
# so ask_annotator() below can ground answers to *any* question about the brief.
ANNOTATOR_PDF_PATH = (
        Path(__file__).resolve().parent.parent / "data" / "Data_Annotation_AI_Training_Assessment_Google_Docs.pdf"
)


def load_annotator_context(pdf_path: Path = ANNOTATOR_PDF_PATH) -> str:
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


ANNOTATOR_CONTEXT_TEXT = load_annotator_context()
# A context passage missing the second (2027) deadline, for the ContextualRecall
# negative control: retrieval that's real but incomplete relative to what the
# expected answer needs.
CMS_0057F_PARTIAL_CONTEXT = [
    "CMS-0057-F (CMS Interoperability & Prior Authorization) was released 17 January "
    "2024 and applies to Medicare Advantage, Medicaid, CHIP, and QHP issuers on the "
    "FFE. Its operational provisions (faster prior authorization turnaround, "
    "specific denial reasons, and Patient Access API metric reporting) took effect "
    "1 January 2026, a deadline that has already passed, leaving non-compliant "
    "plans currently out of compliance."
]
# Distractor chunks for the ContextualPrecision ranking tests: real facts, but not
# relevant to the CMS-0057-F deadline question.
IRRELEVANT_CHUNK_HITECH = (
    "The HITECH Act, enacted in 2009, expanded HIPAA breach notification "
    "requirements and increased penalties for violations."
)
IRRELEVANT_CHUNK_TEFCA = (
    "TEFCA establishes a voluntary nationwide framework for health information "
    "exchange, distinct from CMS-0057-F's prior authorization requirements."
)

# Reused across the "bad" prompts: withholds grounding context and instructs the model
# to always answer directly, so it falls back on its own (often wrong/outdated)
# knowledge instead of the correct source-backed answer - a realistic simulation of an
# ungrounded production chatbot, not an adversarial jailbreak.
ALWAYS_ANSWER_SYSTEM_PROMPT = (
    "You are a helpful healthcare operations assistant. Always give the user a direct, "
    "confident answer to their question. Do not say you don't know or suggest they "
    "check elsewhere."
)

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


def print_result(metric, actual_output: str = None) -> None:
    # metric.measure() alone prints nothing, and this suite intentionally calls
    # measure() + a plain assert (not assert_test()) so "bad" cases can assert a
    # correct failure instead of raising - which means deepeval's own -vv/-s score
    # and reason printout never fires (it's wired to assert_test()). Printing here
    # is what actually makes `pytest -s` / `deepeval test run ... -s` show the
    # score and reason, on a pass as well as a fail.
    print(f"\n[{metric.__class__.__name__}] score={metric.score} success={metric.success}")
    print(f"reason: {getattr(metric, 'reason', None)}")
    if actual_output is not None:
        print(f"actual_output: {actual_output!r}")


def count_sentences(text: str) -> int:
    # Deterministic, not LLM-judged: the brief calls this trap "cleanly verifiable —
    # either it is or it isn't," so an exact-count instruction gets a plain count
    # rather than an LLM's notoriously unreliable sentence counting.
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def ask_annotator(query: str, context: str = None) -> tuple[str, HallucinationMetric]:
    """Ask the Annotator (the healthcare-ops agent this suite exists to support) a
    question grounded in `context` - defaults to the full text of the assessment
    brief PDF (ANNOTATOR_CONTEXT_TEXT) - answered live by the local generator model
    and judged for groundedness by the local judge model.

    This is a callable utility, not a fixed test case: pass your own query (and
    optionally your own context) to probe the brief - or any other material - without
    writing a new test function each time. Returns (answer, metric); read
    metric.score / metric.reason / metric.success to see the judge's verdict.

    Example:
        from tests.test_annotation_assessment import ask_annotator
        answer, metric = ask_annotator("What does the brief say about jurisdictional layering?")
        print(answer, metric.score, metric.reason)
    """
    ctx = context if context is not None else ANNOTATOR_CONTEXT_TEXT
    answer = llm_answer(
        f"Answer the question using ONLY the information below. Do not add anything "
        f"not stated here. Be concise - a few sentences is enough.\n\n"
        f"Information:\n{ctx}\n\nQuestion: {query}\n\nAnswer:"
    )
    test_case = LLMTestCase(input=query, actual_output=answer, context=[ctx])
    metric = HallucinationMetric(threshold=0.5, model=ANNOTATOR_JUDGE_MODEL, include_reason=True)
    metric.measure(test_case)
    return answer, metric


# ---------------------------------------------------------------------------
# Temporal/status confusion (hallucination): the CMS-0057-F two-deadline trap.
# The brief names this explicitly: "Models routinely conflate them or place
# both in 2027."
# ---------------------------------------------------------------------------

def test_cms_0057f_deadlines_kept_distinct():
    # Directive enough to reliably pull both dates out of the small generator model:
    # a plain "answer the question" prompt kept collapsing to just the 2026 date,
    # which the judge read as an incomplete (not contradictory) answer anyway.
    actual_output = llm_answer(
        f"Using ONLY the information below, write two sentences: the first sentence "
        f"must state that the operational provisions took effect 1 January 2026 and "
        f"that this deadline has passed. The second sentence must state that four "
        f"production FHIR APIs are separately required starting 1 January 2027.\n\n"
        f"Information:\n{CMS_0057F_CONTEXT_TEXT}\n\nAnswer:"
    )
    test_case = LLMTestCase(
        input=CMS_0057F_QUESTION, actual_output=actual_output, context=CMS_0057F_CONTEXT
    )
    metric = hallucination_metric()
    metric.measure(test_case)
    print_result(metric, actual_output)
    assert metric.success, (
        f"Expected the grounded two-deadline answer to pass hallucination check, "
        f"got score={metric.score}, reason={metric.reason}, actual_output={actual_output!r}"
    )


def test_cms_0057f_deadlines_conflated_is_caught():
    actual_output = llm_answer(
        f"{ALWAYS_ANSWER_SYSTEM_PROMPT}\n\nQuestion: {CMS_0057F_QUESTION}\n\nAnswer:"
    )
    test_case = LLMTestCase(
        input=CMS_0057F_QUESTION, actual_output=actual_output, context=CMS_0057F_CONTEXT
    )
    metric = hallucination_metric()
    metric.measure(test_case)
    print_result(metric, actual_output)
    assert not metric.success, (
        f"Expected the ungrounded answer to be flagged as hallucination, "
        f"got score={metric.score}, reason={metric.reason}, actual_output={actual_output!r}"
    )


# ---------------------------------------------------------------------------
# Regulatory status confusion (speculation): HIPAA Security Rule proposed vs.
# current law — the brief's own worked example of this failure mode.
# ---------------------------------------------------------------------------

def test_hipaa_security_rule_status_kept_accurate():
    actual_output = llm_answer(
        f"Answer the question using ONLY the information below. Do not add anything "
        f"not stated here.\n\nInformation:\n{HIPAA_SECURITY_RULE_CONTEXT_TEXT}\n\n"
        f"Question: {HIPAA_QUESTION}\n\nAnswer:"
    )
    test_case = LLMTestCase(
        input=HIPAA_QUESTION, actual_output=actual_output, context=HIPAA_SECURITY_RULE_CONTEXT
    )
    SPECULATION_METRIC.measure(test_case)
    print_result(SPECULATION_METRIC, actual_output)
    assert SPECULATION_METRIC.success, (
        f"Expected an answer preserving 'proposed, not final' status to pass, "
        f"got score={SPECULATION_METRIC.score}, reason={SPECULATION_METRIC.reason}, "
        f"actual_output={actual_output!r}"
    )


def test_hipaa_security_rule_status_overstatement_is_caught():
    actual_output = llm_answer(
        f"{ALWAYS_ANSWER_SYSTEM_PROMPT}\n\nQuestion: {HIPAA_QUESTION}\n\nAnswer:"
    )
    test_case = LLMTestCase(
        input=HIPAA_QUESTION, actual_output=actual_output, context=HIPAA_SECURITY_RULE_CONTEXT
    )
    SPECULATION_METRIC.measure(test_case)
    print_result(SPECULATION_METRIC, actual_output)
    assert not SPECULATION_METRIC.success, (
        f"Expected a proposed rule stated as current law to be flagged, "
        f"got score={SPECULATION_METRIC.score}, reason={SPECULATION_METRIC.reason}, "
        f"actual_output={actual_output!r}"
    )


# ---------------------------------------------------------------------------
# Negative knowledge / refusing to speculate: the correct answer is "this
# cannot be determined from the source provided."
# ---------------------------------------------------------------------------

def test_penalty_amount_correctly_unanswered():
    actual_output = llm_answer(
        f"Answer the question using ONLY the information below. If the information "
        f"below does not state the answer, say plainly that it cannot be determined "
        f"from the source provided, rather than guessing.\n\n"
        f"Information:\n{CMS_0057F_CONTEXT_TEXT}\n\nQuestion: {PENALTY_QUESTION}\n\nAnswer:"
    )
    test_case = LLMTestCase(
        input=PENALTY_QUESTION, actual_output=actual_output, context=CMS_0057F_CONTEXT
    )
    REFUSAL_METRIC.measure(test_case)
    print_result(REFUSAL_METRIC, actual_output)
    assert REFUSAL_METRIC.success, (
        f"Expected a correct refusal to guess the penalty amount to pass, "
        f"got score={REFUSAL_METRIC.score}, reason={REFUSAL_METRIC.reason}, "
        f"actual_output={actual_output!r}"
    )


def test_penalty_amount_fabrication_is_caught():
    actual_output = llm_answer(
        f"{ALWAYS_ANSWER_SYSTEM_PROMPT}\n\nQuestion: {PENALTY_QUESTION}\n\nAnswer:"
    )
    test_case = LLMTestCase(
        input=PENALTY_QUESTION, actual_output=actual_output, context=CMS_0057F_CONTEXT
    )
    REFUSAL_METRIC.measure(test_case)
    print_result(REFUSAL_METRIC, actual_output)
    assert not REFUSAL_METRIC.success, (
        f"Expected a fabricated penalty amount to be flagged, "
        f"got score={REFUSAL_METRIC.score}, reason={REFUSAL_METRIC.reason}, "
        f"actual_output={actual_output!r}"
    )


# ---------------------------------------------------------------------------
# Long-output/instruction-compliance degradation: "exactly two sentences"
# means exactly two — the brief calls this "a real trap." It's checked
# deterministically (plain sentence count), not by LLM judgment, since the
# brief itself notes it's "cleanly verifiable — either it is or it isn't,"
# and small models are unreliable counters besides.
# ---------------------------------------------------------------------------

def test_exact_two_sentence_instruction_followed():
    actual_output = llm_answer(
        f"Answer the question using ONLY the information below. Your answer must be "
        f"written as exactly two separate sentences (exactly two periods total, no "
        f"more, no fewer).\n\nInformation:\n{CMS_0057F_CONTEXT_TEXT}\n\n"
        f"Question: {CMS_0057F_QUESTION}\n\nAnswer:"
    )
    assert count_sentences(actual_output) == 2, (
        f"Expected exactly two sentences, got {count_sentences(actual_output)}: "
        f"{actual_output!r}"
    )


def test_exact_two_sentence_instruction_violated_is_caught():
    # The two-sentence instruction is stated once, then buried under a demanding,
    # multi-part request - reproducing the brief's own description of the failure
    # ("constraints given at the start decay... models drift") rather than just
    # asking the model to misbehave.
    actual_output = llm_answer(
        "Write in exactly two sentences. Now, using the information below, give a "
        "complete, detailed summary of CMS-0057-F covering: (1) the exact release "
        "date, (2) which payer types it applies to, (3) the first compliance "
        "deadline and exactly what it required, (4) the current compliance status "
        "for non-compliant plans, and (5) the second, later deadline and what it "
        "separately requires. Be thorough and specific about every one of these "
        f"five points so a compliance officer has everything they need.\n\n"
        f"Information:\n{CMS_0057F_CONTEXT_TEXT}\n\nAnswer:"
    )
    assert count_sentences(actual_output) != 2, (
        f"Expected the drifted output to violate 'exactly two sentences' so the "
        f"instruction-compliance check catches it, but got exactly "
        f"{count_sentences(actual_output)}: {actual_output!r}"
    )


# ---------------------------------------------------------------------------
# AnswerRelevancyMetric: does actual_output actually address input, rather
# than being on-topic-domain but not actually answering the question asked
# (a realistic "wrong document retrieved" failure).
# ---------------------------------------------------------------------------

def test_answer_relevancy_on_topic_passes():
    actual_output = llm_answer(
        f"Answer using ONLY this info: {CMS_0057F_CONTEXT_TEXT}\n\n"
        f"Q: {CMS_0057F_EFFECTIVE_DATE_QUESTION}\nA:"
    )
    test_case = LLMTestCase(input=CMS_0057F_EFFECTIVE_DATE_QUESTION, actual_output=actual_output)
    metric = AnswerRelevancyMetric(threshold=0.5, model=JUDGE_MODEL, include_reason=True)
    metric.measure(test_case)
    print_result(metric, actual_output)
    assert metric.success, (
        f"Expected an on-topic, grounded answer to pass relevancy check, "
        f"got score={metric.score}, reason={metric.reason}, actual_output={actual_output!r}"
    )


def test_answer_relevancy_off_topic_is_caught():
    # Generated by answering a completely different (but still healthcare-ops)
    # question, then paired with the CMS-0057-F question as input - a realistic
    # "wrong document retrieved" failure, not a hand-written bad answer.
    actual_output = llm_answer(
        "Answer this question: What are the standard components of a "
        "HIPAA-compliant business associate agreement?\n\nA:"
    )
    test_case = LLMTestCase(input=CMS_0057F_EFFECTIVE_DATE_QUESTION, actual_output=actual_output)
    metric = AnswerRelevancyMetric(threshold=0.5, model=JUDGE_MODEL, include_reason=True)
    metric.measure(test_case)
    print_result(metric, actual_output)
    assert not metric.success, (
        f"Expected an off-topic answer to be flagged as irrelevant, "
        f"got score={metric.score}, reason={metric.reason}, actual_output={actual_output!r}"
    )


# ---------------------------------------------------------------------------
# ContextualPrecisionMetric: retrieval quality - is the relevant chunk ranked
# ahead of irrelevant distractor chunks in retrieval_context?
# ---------------------------------------------------------------------------

def test_contextual_precision_relevant_chunk_ranked_first_passes():
    retrieval_context = [CMS_0057F_CONTEXT_TEXT, IRRELEVANT_CHUNK_HITECH, IRRELEVANT_CHUNK_TEFCA]
    actual_output = llm_answer(
        f"Answer using ONLY this info: {CMS_0057F_CONTEXT_TEXT}\n\n"
        f"Q: {CMS_0057F_EFFECTIVE_DATE_QUESTION}\nA:"
    )
    test_case = LLMTestCase(
        input=CMS_0057F_EFFECTIVE_DATE_QUESTION,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=CMS_0057F_EXPECTED_OUTPUT,
    )
    metric = ContextualPrecisionMetric(threshold=0.5, model=JUDGE_MODEL, include_reason=True)
    metric.measure(test_case)
    print_result(metric, actual_output)
    assert metric.success, (
        f"Expected the relevant chunk ranked first to pass precision check, "
        f"got score={metric.score}, reason={metric.reason}"
    )


def test_contextual_precision_relevant_chunk_buried_is_caught():
    # Same three chunks, relevant one buried last instead of ranked first.
    retrieval_context = [IRRELEVANT_CHUNK_HITECH, IRRELEVANT_CHUNK_TEFCA, CMS_0057F_CONTEXT_TEXT]
    actual_output = llm_answer(
        f"Answer using ONLY this info: {CMS_0057F_CONTEXT_TEXT}\n\n"
        f"Q: {CMS_0057F_EFFECTIVE_DATE_QUESTION}\nA:"
    )
    test_case = LLMTestCase(
        input=CMS_0057F_EFFECTIVE_DATE_QUESTION,
        actual_output=actual_output,
        retrieval_context=retrieval_context,
        expected_output=CMS_0057F_EXPECTED_OUTPUT,
    )
    metric = ContextualPrecisionMetric(threshold=0.5, model=JUDGE_MODEL, include_reason=True)
    metric.measure(test_case)
    print_result(metric, actual_output)
    assert not metric.success, (
        f"Expected the buried relevant chunk to be flagged for poor ranking, "
        f"got score={metric.score}, reason={metric.reason}"
    )


# ---------------------------------------------------------------------------
# ContextualRecallMetric: does retrieval_context contain everything needed to
# support expected_output, or is it missing required facts?
# ---------------------------------------------------------------------------

def test_contextual_recall_complete_retrieval_passes():
    actual_output = llm_answer(
        f"Answer using ONLY this info: {CMS_0057F_CONTEXT_TEXT}\n\n"
        f"Q: {CMS_0057F_EFFECTIVE_DATE_QUESTION}\nA:"
    )
    test_case = LLMTestCase(
        input=CMS_0057F_EFFECTIVE_DATE_QUESTION,
        actual_output=actual_output,
        retrieval_context=CMS_0057F_CONTEXT,
        expected_output=CMS_0057F_EXPECTED_OUTPUT,
    )
    # threshold=0.6 (not 0.5): the incomplete-retrieval negative control below
    # lands exactly at 0.5, so 0.5 wouldn't cleanly separate pass from fail.
    metric = ContextualRecallMetric(threshold=0.6, model=JUDGE_MODEL, include_reason=True)
    metric.measure(test_case)
    print_result(metric, actual_output)
    assert metric.success, (
        f"Expected retrieval covering both deadlines to pass recall check, "
        f"got score={metric.score}, reason={metric.reason}"
    )


def test_contextual_recall_incomplete_retrieval_is_caught():
    # Retrieval_context omits the second (2027) deadline that expected_output
    # requires - a real, verifiable gap, not a fabricated bad case.
    actual_output = llm_answer(
        f"Answer using ONLY this info: {CMS_0057F_PARTIAL_CONTEXT[0]}\n\n"
        f"Q: {CMS_0057F_EFFECTIVE_DATE_QUESTION}\nA:"
    )
    test_case = LLMTestCase(
        input=CMS_0057F_EFFECTIVE_DATE_QUESTION,
        actual_output=actual_output,
        retrieval_context=CMS_0057F_PARTIAL_CONTEXT,
        expected_output=CMS_0057F_EXPECTED_OUTPUT,
    )
    metric = ContextualRecallMetric(threshold=0.6, model=JUDGE_MODEL, include_reason=True)
    metric.measure(test_case)
    print_result(metric, actual_output)
    assert not metric.success, (
        f"Expected the incomplete retrieval (missing the 2027 deadline) to be "
        f"flagged for insufficient recall, got score={metric.score}, reason={metric.reason}"
    )


# Annotator contextual relevance tests: does the answer actually address the question asked, rather than being on-topic-domain but not answering the specific question?
# Uses the full PDF text as context, so the Annotator can answer any question about the brief without hand-picking a short excerpt.
def test_annotator_relevancy_on_topic_passes():
    actual_output, metric = ask_annotator(CMS_0057F_EFFECTIVE_DATE_QUESTION)
    assert metric.success, (
        f"Expected an on-topic, grounded answer to pass relevancy check, "
        f"got score={metric.score}, reason={metric.reason}, actual_output={actual_output!r}"
    )

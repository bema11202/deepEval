import pytest
import deepeval
from deepeval import assert_test
from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
)


# Sample golden representing a RAG input/expected-output pair with retrieved context
sample_golden = Golden(
    input="What is the capital of France?",
    expected_output="Paris",
    context=["France is a country in Western Europe. Its capital city is Paris."],
    retrieval_context=["France is a country in Western Europe. Its capital city is Paris.Its official language is French and it has a population of over 67 million people."]
)


def test_rag_answer_relevancy():
    relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

    test_case = LLMTestCase(
        input=sample_golden.input,
        actual_output="The capital of France is Paris.",
        expected_output=sample_golden.expected_output,
        retrieval_context=sample_golden.retrieval_context,
    )

    assert_test(test_case, [relevancy_metric])


def test_rag_faithfulness():
    faithfulness_metric = FaithfulnessMetric(threshold=0.5)

    test_case = LLMTestCase(
        input=sample_golden.input,
        actual_output="The capital of France is Paris.",
        expected_output=sample_golden.expected_output,
        retrieval_context=sample_golden.retrieval_context,
    )

    assert_test(test_case, [faithfulness_metric])


def test_rag_faithfulness_hallucinated():
    faithfulness_metric = FaithfulnessMetric(threshold=0.5)

    # Actual output contradicts the retrieval context (Lyon vs. Paris): this
    # should score below threshold, so we assert failure rather than using
    # assert_test.
    test_case = LLMTestCase(
        input=sample_golden.input,
        actual_output="The capital of France is Lyon, home to over 10 million residents.",
        retrieval_context=sample_golden.retrieval_context,
    )

    faithfulness_metric.measure(test_case)
    assert not faithfulness_metric.is_successful()


def test_rag_faithfulness_partial():
    faithfulness_metric = FaithfulnessMetric(threshold=0.5)

    # Retrieval context with two verifiable facts: capital city and the river
    # it sits on.
    retrieval_context = [
        "France is a country in Western Europe. Its capital city is Paris.",
        "Paris is located on the Seine river.",
    ]

    # Actual output correctly affirms one claim (Paris is the capital) but
    # contradicts the other (Seine vs. Loire). This should score somewhere
    # between fully faithful and fully hallucinated, so we measure directly
    # rather than using assert_test.
    test_case_partial = LLMTestCase(
        input=sample_golden.input,
        actual_output="The capital of France is Paris. Paris is located on the Loire river.",
        expected_output=sample_golden.expected_output,
        retrieval_context=retrieval_context,
    )

    faithfulness_metric.measure(test_case_partial)
    assert 0.0 < faithfulness_metric.score < 1.0


def test_rag_contextual_recall():
    recall_metric = ContextualRecallMetric(threshold=0.5)

    test_case = LLMTestCase(
        input=sample_golden.input,
        actual_output="The capital of France is Paris.",
        expected_output=sample_golden.expected_output,
        retrieval_context=sample_golden.retrieval_context,
    )

    assert_test(test_case, [recall_metric])


def test_rag_contextual_precision():
    precision_metric = ContextualPrecisionMetric(threshold=0.5)

    # Retrieval context with the relevant node ranked first and an irrelevant
    # node second, so a well-ranked retriever should score highly.
    retrieval_context = [
        "France is a country in Western Europe. Its capital city is Paris.",
        "The Eiffel Tower was completed in 1889 for the World's Fair.",
    ]

    test_case = LLMTestCase(
        input=sample_golden.input,
        actual_output="The capital of France is Paris.",
        expected_output=sample_golden.expected_output,
        retrieval_context=retrieval_context,
    )

    assert_test(test_case, [precision_metric])


def test_rag_contextual_relevancy():
    relevancy_metric = ContextualRelevancyMetric(threshold=0.5)

    # Retrieval context mixing a relevant node with an irrelevant one, to
    # check how much of the retrieved context is actually relevant to the
    # input (independent of ranking).
    retrieval_context = [
        "France is a country in Western Europe. Its capital city is Paris.",
        "The Eiffel Tower was completed in 1889 for the World's Fair.",
    ]

    test_case = LLMTestCase(
        input=sample_golden.input,
        actual_output="The capital of France is Paris.",
        expected_output=sample_golden.expected_output,
        retrieval_context=retrieval_context,
    )

    relevancy_metric.measure(test_case, _show_indicator=True)
    assert 0.0 < relevancy_metric.score < 1.0

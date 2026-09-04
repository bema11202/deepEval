from deepeval import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import AnswerRelevancyMetric


# Test the AnswerRelevancyMetric on a validation dataset
def test_answer_relevancy():
    # Load the validation dataset
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5,
        verbose_mode=True,
        include_reason=True
    )
    # Write a test case for the answer relevancy metric
    test_case = LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="The capital of France is Paris."
    )
    assert_test(test_case, [relevancy_metric])

    # Test the AnswerRelevancyMetric on a non-relevant answer: this should
    # score below threshold, so we assert failure rather than using assert_test.
    test_case_non_relevant = LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="The capital of Germany is Berlin."
    )
    relevancy_metric.measure(test_case_non_relevant)
    assert not relevancy_metric.is_successful()

    # Test the AnswerRelevancyMetric on a relevant answer
    test_case_relevant = LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="Paris is the capital of France."
    )
    assert_test(test_case_relevant, [relevancy_metric])


# Test the AnswerRelevancyMetric on a partially relevant answer
def test_answer_relevancy_partial():
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5
    )
    test_case_partial_relevant = LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="The capital of France is Paris, which is known for its art and culture."
    )
    assert_test(test_case_partial_relevant, [relevancy_metric])

    test_case_partial_relevant_2 = LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="Paris is known for its art and culture."
    )
    assert_test(test_case_partial_relevant_2, [relevancy_metric])

    # Off-topic answer: should score below threshold.
    test_case_irrelevant = LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="I had pizza for lunch and it was delicious."
    )
    relevancy_metric.measure(test_case_irrelevant)
    assert not relevancy_metric.is_successful()


# Test the AnswerRelevancyMetric on a list of relevant answers
def test_answer_relevancy_list():
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5
    )
    test_case_list_relevant = LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="The capital of France is Paris. Paris is known for its art and culture."
    )
    assert_test(test_case_list_relevant, [relevancy_metric])

    # Off-topic answer: should score below threshold.
    test_case_list_irrelevant = LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="I enjoy hiking on weekends. The weather has been nice lately."
    )
    relevancy_metric.measure(test_case_list_irrelevant)
    assert not relevancy_metric.is_successful()


# Test the AnswerRelevancyMetric on a list of partially relevant answers
def test_answer_relevancy_list_partial():
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5
    )
    test_case_list_partial_relevant = LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="The capital of France is Paris. Paris is known for its art and culture. The capital of Germany is Berlin."
    )
    assert_test(test_case_list_partial_relevant, [relevancy_metric])

    # Off-topic answer: should score below threshold, so we assert failure
    # rather than using assert_test.
    test_case_list_partial_irrelevant = LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="My favorite movie genre is science fiction. I also enjoy cooking Italian food on weekends."
    )
    relevancy_metric.measure(test_case_list_partial_irrelevant)
    assert not relevancy_metric.is_successful()

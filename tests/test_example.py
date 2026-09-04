from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric


# Test the AnswerRelevancyMetric on a list-based question
def test_answer_relevancy_list_based():
    relevancy_metric = AnswerRelevancyMetric(
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="List the first three prime numbers.",
        expected_output="2, 3, 5",
        actual_output="The first three prime numbers are 2, 3, and 5.",
    )
    assert_test(test_case, [relevancy_metric])

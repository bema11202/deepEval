from deepeval import assert_test, test_case
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import AnswerRelevancyMetric
from holoviews.operation import threshold

base_metric = AnswerRelevancyMetric()


def test_correctness_list_based():
    correctness_metric = AnswerRelevancyMetric(
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="List the first three prime numbers.",
        expected_output="2, 3, 5",
        actual_output="The first three prime numbers are 2, 3, and 5.",
    )
    assert_test(test_case, [correctness_metric])

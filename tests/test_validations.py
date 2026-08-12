from deepeval import assert_test, test_case
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval, retrieval_context_display
from nltk.parse import evaluate


# Test the GEval metric on a validation dataset
def test_correctness():
    # Load the validation dataset
    correctness_metric = GEval(
        name="Correctness",
        criteria="The actual output should convey the same answer as the expected output, wording aside.",
        evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.5
    )
    # Write a test case for the correctness metric
    test_case = LLMTestCase(
        input="What is 5 divided by 2?",
        expected_output="2.5",
        actual_output="Five divided by two is 2.5"
    )
    assert_test(test_case, [correctness_metric])


# Test the GEval metric on a non-numeric question
def test_correctness_non_numeric():
    correctness_metric = GEval(
        name="Correctness",
        criteria="The actual output should convey the same answer as the expected output, wording aside.",
        evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="Paris."
    )
    assert_test(test_case, [correctness_metric])


def test_correctness_non_numeric2():
    correctness_metric = GEval(
        name="Correctness",
        criteria="The actual output should convey the same answer as the expected output, wording aside. Additional consistent detail beyond the expected output is acceptable.",
        evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="How many days in a year?",
        expected_output="365",
        actual_output="There are 365 days in a year. But some leap years have 366 days."
    )
    assert_test(test_case, [correctness_metric])


def test_correctness_list_based():
    correctness_metric = GEval(
        name="Correctness",
        criteria="The actual output should convey the same answer as the expected output, wording aside.",
        evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )
    test_case = LLMTestCase(
        input="Name the three primary colors.",
        expected_output="Red, blue, and yellow.",
        actual_output="The primary colors are red, yellow, and blue."
    )
    assert_test(test_case, [correctness_metric])

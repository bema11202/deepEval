from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import ContextualPrecisionMetric

correctness_metric = GEval(
    name="Correctness",
    criteria="Compare the Actual Output to the Expected Output and check whether they express the same core meaning or conclusion.",
    evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
    threshold=0.5
)

# Contextual Precision Metric
contextual_precision_metric = ContextualPrecisionMetric(
    threshold=0.5,
    verbose_mode=True,
)

goldens = [
    Golden(
        name='Test Case 1',
        input="What is 5 divided by 2?",
        expected_output="2.5",
        actual_output="Five divided by two is 2.5",
        retrieval_context=["5 divided by 2 equals 2.5."],
    ),
    Golden(
        name='Test Case 2',
        input="What is the capital of France?",
        expected_output="Paris",
        actual_output="Paris.",
        retrieval_context=["Paris is the capital of France."],
    ),
    Golden(
        name='Test Case 3',
        input="How many days in a year?",
        expected_output="365",
        actual_output="There are 365 days in a year. But some leap years have 366 days.",
        retrieval_context=["A common year has 365 days; a leap year has 366 days."],
    ),
]

# Create an evaluation dataset
evaluation_dataset = EvaluationDataset(goldens=goldens)


def test_datasets_eval():
    for golden in evaluation_dataset.goldens:
        test_case = LLMTestCase(
            input=golden.input,
            expected_output=golden.expected_output,
            actual_output=golden.actual_output,
            retrieval_context=golden.retrieval_context,
        )

        assert_test(test_case, [contextual_precision_metric])

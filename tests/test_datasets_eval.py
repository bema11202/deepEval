import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import ContextualPrecisionMetric
from sympy.benchmarks.bench_discrete_log import data_set_1, data_set_2

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

golden = [
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
evaluation_dataset = EvaluationDataset(goldens=golden)


def test_datasets_eval():
    for golden_ in evaluation_dataset.goldens:
        test_case = LLMTestCase(
            input=golden_.input,
            expected_output=golden_.expected_output,
            actual_output=golden_.actual_output,
            retrieval_context=golden_.retrieval_context,
        )

        assert_test(test_case, [contextual_precision_metric])


# Create a dataset from the synthetic json file data and run the evaluation on it. The dataset is in the format of a list of dictionaries, where each dictionary has the keys: input, expected_output, actual_output, and retrieval_context.
dataset_1 = EvaluationDataset()
dataset_1.add_goldens_from_csv_file(
    "/Users/B.Masoko/PycharmProjects/PythonProject/DeepEval/data/healthcare_golden_dataset.csv",
    retrieval_context_col_name="context",
)
dataset_2 = EvaluationDataset()
dataset_2.add_goldens_from_json_file(
    "/Users/B.Masoko/PycharmProjects/PythonProject/DeepEval/data/financial_golden_dataset.json",
    retrieval_context_key_name="context",
)


# Run the evaluation on the datasets
def test_datasets_eval_from_csv():
    for golden_ in dataset_1.goldens:
        test_case = LLMTestCase(
            input=golden_.input,
            expected_output=golden_.expected_output,
            actual_output=golden_.actual_output,
            retrieval_context=golden_.retrieval_context,
        )

        assert_test(test_case, [contextual_precision_metric])


def test_datasets_eval_from_json():
    for golden_ in dataset_2.goldens:
        test_case = LLMTestCase(
            input=golden_.input,
            expected_output=golden_.expected_output,
            actual_output=golden_.actual_output,
            retrieval_context=golden_.retrieval_context,
        )

        assert_test(test_case, [contextual_precision_metric])

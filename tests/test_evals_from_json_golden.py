from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

contextual_precision_metric = ContextualPrecisionMetric(
    threshold=0.5,
    verbose_mode=True,
)

dataset = EvaluationDataset()
dataset.add_goldens_from_json_file(
    "/Users/B.Masoko/PycharmProjects/PythonProject/DeepEval/data/financial_golden_dataset.json",
    retrieval_context_key_name="context",
)


def test_evals_from_json_golden():
    for golden in dataset.goldens:
        test_case = LLMTestCase(
            input=golden.input,
            expected_output=golden.expected_output,
            actual_output=golden.actual_output,
            retrieval_context=golden.retrieval_context,
        )

        assert_test(test_case, [contextual_precision_metric])

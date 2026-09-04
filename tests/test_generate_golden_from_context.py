import chromadb
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import (AnswerRelevancyMetric,
                              ContextualPrecisionMetric,
                              GEval, )
from deepeval.test_case import LLMTestCase
from pathlib import Path

evaluation_dataset = EvaluationDataset()
evaluation_dataset.add_goldens_from_csv_file(
    file_path="/Users/B.Masoko/PycharmProjects/PythonProject/DeepEval/data/healthcare_golden_dataset.csv",
    input_col_name="input",
    expected_output_col_name="expected_output",
    retrieval_context_col_name="context",
    retrieval_context_col_delimiter="|",
)
goldens = evaluation_dataset.goldens

client = chromadb.Client()
dataset = client.get_or_create_collection("healthcare_golden_dataset")
for i, golden in enumerate(goldens):
    dataset.add(
        documents=[golden.input],
        metadatas=[{
            "expected_output": golden.expected_output,
            "retrieval_context": "|".join(golden.retrieval_context or []),
        }],
        ids=[str(i)],
    )

dataset.peek()


# Metric test cases built directly from the goldens generated above.
def test_contextual_relevance_from_context():
    answer_metric = AnswerRelevancyMetric(threshold=0.5)

    for golden_ in evaluation_dataset.goldens:
        test_case = LLMTestCase(
            input=golden_.input,
            actual_output=golden_.expected_output,
            expected_output=golden_.expected_output,
            retrieval_context=golden_.retrieval_context,
        )
        assert_test(test_case, [answer_metric])

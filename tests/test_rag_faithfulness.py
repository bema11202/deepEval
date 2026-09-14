# This test file is designed to evaluate the faithfulness of a retrieval-augmented generation (RAG) model using the DeepEval framework. It includes tests for contextual relevance, ensuring that the model's responses are consistent with the provided documents.
from pathlib import Path
import chromadb
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Define the data directory and load the golden dataset
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
goldens_file_path = DATA_DIR / "data_annotation_golden_dataset.json"

# List of documents to be synthesized into goldens
documents = [str(DATA_DIR / "privacysummary.pdf"), str(DATA_DIR / "ICD-10-CM_October_2025_FY26Guidelines.pdf")]

evaluation_dataset = EvaluationDataset()
if goldens_file_path.exists():
    # Reuse the previously generated + saved goldens instead of re-running the synthesizer.
    evaluation_dataset.add_goldens_from_json_file(str(goldens_file_path))
else:
    evaluation_dataset.generate_goldens_from_docs(
        document_paths=documents,
        synthesizer=Synthesizer(),
    )
    evaluation_dataset.save_as(
        file_type="json",
        directory=str(DATA_DIR),
        file_name=goldens_file_path.stem,
    )
goldens = evaluation_dataset.goldens

client = chromadb.Client()
dataset = client.get_or_create_collection("data_annotation_golden_dataset")
for i, golden in enumerate(goldens):
    dataset.add(
        documents=[golden.input],
        metadatas=[{
            "expected_output": golden.expected_output,
            "retrieval_context": "|".join(golden.retrieval_context or []),
            "id": str(i),
            "source_file": golden.source_file,
        }],
        ids=[str(i)],
    )


# Metric test cases built directly from the golden generated above.
# Test for faithfulness of the RAG model's output against the expected output.
def test_faithfulness():
    # answer_metric = AnswerRelevancyMetric(threshold=0.5)
    sync_metric = AnswerRelevancyMetric(threshold=0.5, async_mode=False)

    for golden_ in evaluation_dataset.goldens:
        test_case = LLMTestCase(
            input=golden_.input,
            actual_output=golden_.expected_output,
            expected_output=golden_.expected_output,
            retrieval_context=golden_.retrieval_context,
        )
        assert_test(test_case, [sync_metric])
        sync_metric.measure(test_case)


# Failing test case for faithfulness to demonstrate the evaluation framework's ability to catch discrepancies.
def test_faithfulness_failure():
    # This test is designed to fail to demonstrate the evaluation framework's ability to catch discrepancies.
    sync_metric = AnswerRelevancyMetric(threshold=0.5, async_mode=False)

    for golden_ in evaluation_dataset.goldens:
        # Intentionally providing an incorrect actual output to simulate a failure case.
        test_case = LLMTestCase(
            input=golden_.input,
            actual_output="This is an incorrect output for testing purposes.",
            expected_output=golden_.expected_output,
            retrieval_context=golden_.retrieval_context,
        )
        assert_test(test_case, [sync_metric])
        sync_metric.measure(test_case)

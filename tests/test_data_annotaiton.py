# This file contains preparatory tests for the Healthcare Operations AI Trainer position with Data Annotation.
from pathlib import Path
import chromadb
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLDEN_JSON_PATH = DATA_DIR / "data_annotation_golden_dataset.json"

# List of documents to be synthesized into goldens
documents = [str(DATA_DIR / "privacysummary.pdf"), str(DATA_DIR / "ICD-10-CM October 2025 FY26Guidelines.pdf")]

evaluation_dataset = EvaluationDataset()
evaluation_dataset.generate_goldens_from_docs(
    document_paths=documents,
    synthesizer=Synthesizer(),
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
            "source_document": golden.source_document,
            "source_page": golden.source_page,
            "source_paragraph": golden.source_paragraph,
        }],
        ids=[str(i)],
    )


# Metric test cases built directly from the goldens generated above.
def test_contextual_relevance():
    answer_metric = AnswerRelevancyMetric(threshold=0.5)

    for golden_ in evaluation_dataset.goldens:
        test_case = LLMTestCase(
            input=golden_.input,
            actual_output=golden_.expected_output,
            expected_output=golden_.expected_output,
            retrieval_context=golden_.retrieval_context,
        )
        assert_test(test_case, [answer_metric])

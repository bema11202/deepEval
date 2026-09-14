# This file contains deepeval answer relevancy tests for the Healthcare Operations AI Trainer position with Data Annotation.
from pathlib import Path
import chromadb
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase


# Metric test cases built directly from the goldens generated above.
def test_contextual_relevance():
    DATA_DIR = Path(__file__).resolve().parent.parent / "data"
    goldens_file_path = DATA_DIR / "data_annotation_golden_dataset.json"

    # List of documents to be synthesized into goldens
    documents = [str(DATA_DIR / "privacysummary.pdf"), str(DATA_DIR / "ICD-10-CM_October_2025_FY26Guidelines.pdf")]

    evaluation_dataset = EvaluationDataset()
    if goldens_file_path.exists():
        # Reuse the goldens saved by test_rag_faithfulness.py instead of re-running the synthesizer.
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

    answer_metric = AnswerRelevancyMetric(threshold=0.5)
    sync_metric = AnswerRelevancyMetric(threshold=0.5, async_mode=False)

    for golden_ in evaluation_dataset.goldens:
        test_case = LLMTestCase(
            input=golden_.input,
            actual_output=golden_.expected_output,
            expected_output=golden_.expected_output,
            retrieval_context=golden_.retrieval_context,
        )
        assert_test(test_case, [answer_metric])
        sync_metric.measure(test_case)

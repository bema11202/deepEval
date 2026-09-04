# This file contains tests for the generate_golden_from_source module
from pathlib import Path

from deepeval import assert_test
from deepeval.dataset import EvaluationDataset
from deepeval.synthesizer import Synthesizer
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLDEN_JSON_PATH = DATA_DIR / "golden_from_nooa_docs.json"

# List of documents to be synthesized into goldens
documents = [str(DATA_DIR / "nooa.pdf")]


def test_generate_golden_from_docs():
    synthesizer = Synthesizer()
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=documents,
        include_expected_output=True,
    )

    assert len(goldens) > 0
    for golden in goldens:
        assert golden.input

    synthesizer.save_as(
        file_type='json',
        directory=str(DATA_DIR),
        file_name="golden_from_nooa_docs",
    )


evaluation_dataset = EvaluationDataset()
evaluation_dataset.add_goldens_from_json_file(
    str(GOLDEN_JSON_PATH),
    retrieval_context_key_name="context",
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

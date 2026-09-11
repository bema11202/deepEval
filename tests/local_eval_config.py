# Explicit Ollama model wiring for offline evals.
#
# deepeval 4.1.3's automatic provider auto-detection (env vars / `deepeval set-ollama*`)
# does not actually select Ollama: `should_use_ollama_model()` compares a pydantic
# SecretStr to a plain string (always False), and `should_use_ollama_embedding()`
# only reads `LOCAL_EMBEDDING_API_KEY` from deepeval's hidden key-file, which the
# `--save=dotenv` CLI flag doesn't populate. So evals silently fall back to OpenAI
# instead of raising. Pass these objects into metrics/Synthesizer explicitly instead
# of relying on env-based auto-detection.
import os

from deepeval.models import OllamaModel, OllamaEmbeddingModel

OFFLINE = os.getenv("DEEPEVAL_OFFLINE", "").lower() in ("1", "true", "yes")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3.1:8b")
OLLAMA_EMBEDDING_MODEL_NAME = os.getenv("OLLAMA_EMBEDDING_MODEL_NAME", "nomic-embed-text")
# Smaller/distinct model used as the "system under test" that generates actual_output
# in tests/test_annotation_assessment.py, kept separate from OLLAMA_MODEL_NAME (the
# judge) to avoid self-grading bias (a model tends to rate its own outputs favorably).
OLLAMA_GENERATOR_MODEL_NAME = os.getenv("OLLAMA_GENERATOR_MODEL_NAME", "llama3.2:3b")


def get_local_model():
    """Local Ollama judge model when DEEPEVAL_OFFLINE is set, else None (defaults to OpenAI)."""
    if not OFFLINE:
        return None
    return OllamaModel(model=OLLAMA_MODEL_NAME)


def get_local_embedder():
    """Local Ollama embedding model when DEEPEVAL_OFFLINE is set, else None (defaults to OpenAI)."""
    if not OFFLINE:
        return None
    return OllamaEmbeddingModel(model=OLLAMA_EMBEDDING_MODEL_NAME)

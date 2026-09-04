import pytest
import deepeval
from dotenv import load_dotenv

# Loaded at import time (not in a fixture) because some test modules call
# get_llm_client(...) at module level, before any fixture would run.
load_dotenv()


# deepeval's pytest plugin creates a fresh TestRun in `pytest_sessionstart`,
# which runs after conftest.py is imported. Logging hyperparameters at
# module import time (rather than inside a fixture) would set them on a
# TestRun that gets discarded before tests run, so this must be an
# autouse session fixture instead.
@pytest.fixture(scope="session", autouse=True)
def _log_hyperparameters():
    @deepeval.log_hyperparameters
    def hyperparameters():
        """:return: A dictionary of hyperparameters to log for this test run."""
        return {"model": "gpt-4o-mini", "prompt_template": "my-template-v1"}

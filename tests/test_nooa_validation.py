import asyncio

from deepeval.evaluate import assert_test
from deepeval.test_case import LLMTestCase, SingleTurnParams
from nooa import Agent, strategy
from nooa.config import PredictConfig
from nooa.strategies import PredictStrategy
from nooa.unifiedllm.registry import get_llm_client
from deepeval.metrics import GEval

llm = get_llm_client("gpt-4o-mini")  # OpenAI (uses OPENAI_API_KEY)


# The agent is a Python object. Binding `llm=llm` here is required — without it,
# the agent has no LLM to resolve and raises ValueError at construction time.
class SupportAgent(Agent, llm=llm):
    """You are a support agent."""

    # `Agent` methods need an explicit `@strategy` and an ellipsis body — the
    # docstring (with `{message}` interpolated) becomes the prompt.
    @strategy(PredictStrategy(PredictConfig(output_serialization="tool_call")))
    async def respond(self, message: str) -> str:
        """Respond helpfully to the customer message: {message}"""
        ...


def test_support_agent_response():
    agent = SupportAgent()

    correctness_metric = GEval(
        name="Correctness",
        criteria="Compare the Actual Output to the Expected Output and check whether they express the same core meaning or conclusion.",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=0.5,
    )

    input_prompt = "I recently purchased your product, but it stopped working after a week. Can you help me with this issue?"

    expected_output = "I'm sorry to hear that you're experiencing issues with our product. Please provide me with your order number and any error messages you've received, and I'll assist you in resolving this problem."

    actual_output = asyncio.run(agent.respond(input_prompt))

    test_case = LLMTestCase(
        input=input_prompt,
        expected_output=expected_output,
        actual_output=actual_output,
    )

    assert_test(test_case, [correctness_metric])

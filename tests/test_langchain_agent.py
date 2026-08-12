from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams, ToolCall
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of a customer order by order ID."""
    return f"Order {order_id} shipped on 2026-08-01 and is out for delivery."


def build_support_agent():
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_agent(
        model,
        tools=[get_order_status],
        system_prompt="You are a helpful customer support agent.",
    )


def run_agent(agent, message: str):
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    messages = result["messages"]

    tools_called = [
        ToolCall(name=call["name"], input_parameters=call["args"])
        for msg in messages
        for call in (getattr(msg, "tool_calls", None) or [])
    ]
    return messages[-1].content, tools_called


def test_langchain_agent_calls_tool_for_order_lookup():
    agent = build_support_agent()

    actual_output, tools_called = run_agent(agent, "Where is my order 98765?")

    test_case = LLMTestCase(
        input="Where is my order 98765?",
        actual_output=actual_output,
        expected_output="Your order 98765 shipped on 2026-08-01 and is out for delivery.",
        tools_called=tools_called,
        expected_tools=[ToolCall(name="get_order_status", input_parameters={"order_id": "98765"})],
    )

    assert test_case.tools_called == test_case.expected_tools

    correctness_metric = GEval(
        name="Correctness",
        criteria="Compare the Actual Output to the Expected Output and check whether they express the same core meaning or conclusion.",
        evaluation_params=[SingleTurnParams.EXPECTED_OUTPUT, SingleTurnParams.ACTUAL_OUTPUT],
        threshold=0.5,
    )
    assert_test(test_case, [correctness_metric])


def test_langchain_agent_answers_without_tool_when_unneeded():
    agent = build_support_agent()

    actual_output, tools_called = run_agent(agent, "What is the capital of France?")

    assert tools_called == []

    test_case = LLMTestCase(
        input="What is the capital of France?",
        actual_output=actual_output,
        expected_output="Paris",
    )

    relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [relevancy_metric])

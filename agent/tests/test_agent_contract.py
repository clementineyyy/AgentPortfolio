import unittest
import pandas as pd

from agent.request_models import AgentRunInput, normalize_agent_state
from agent.stock_analysis import (
    _append_stock_data_failure_reply,
    _has_stock_data,
    _resume_chart_action,
)


class AgentRequestContractTest(unittest.TestCase):
    def test_accepts_http_agent_protocol_version(self):
        payload = {
            "threadId": "thread-1",
            "runId": "run-1",
            "protocolVersion": "1.0",
            "state": {},
            "messages": [],
            "tools": [],
            "context": [],
            "forwardedProps": {},
        }

        request = AgentRunInput.model_validate(payload)

        self.assertEqual(request.protocol_version, "1.0")

    def test_initial_request_state_has_portfolio_defaults(self):
        state = normalize_agent_state({})

        self.assertEqual(state["available_cash"], 1_000_000)
        self.assertEqual(state["investment_summary"], {})
        self.assertEqual(state["investment_portfolio"], [])


class EmptyStockDataRegressionTest(unittest.TestCase):
    def test_rate_limited_download_is_not_treated_as_investable_data(self):
        # A provider may return no rows for a rate limit or unsupported range.
        empty_prices = pd.DataFrame()

        self.assertFalse(_has_stock_data(empty_prices))
        self.assertTrue(_has_stock_data(pd.DataFrame({"AAPL": [190.0]})))

    def test_empty_download_adds_assistant_reply_for_sse_response(self):
        messages = []

        _append_stock_data_failure_reply(messages)

        self.assertEqual(messages[-1].role, "assistant")
        self.assertIn("historical price data", messages[-1].content)
        self.assertIn("date range", messages[-1].content)


class ChartActionResumeRegressionTest(unittest.TestCase):
    def test_accepting_chart_action_finishes_without_restarting_analysis(self):
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "chart-call-1",
                        "function": {"name": "render_standard_charts_and_table"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "chart-call-1",
                "content": "Data rendered successfully. Provide summary without making any tool calls",
            },
        ]

        resumed = _resume_chart_action(messages)

        self.assertTrue(resumed)
        self.assertEqual(messages[-1].role, "assistant")
        self.assertIn("accepted", messages[-1].content)

    def test_unrelated_tool_response_does_not_short_circuit_analysis(self):
        messages = [
            {"role": "assistant", "tool_calls": [{"id": "extract-1", "function": {"name": "extract_relevant_data_from_user_prompt"}}]},
            {"role": "tool", "tool_call_id": "extract-1", "content": "ok"},
        ]

        self.assertFalse(_resume_chart_action(messages))


if __name__ == "__main__":
    unittest.main()

import unittest
import pandas as pd

from agent.request_models import AgentRunInput, normalize_agent_state
from agent.stock_analysis import _append_stock_data_failure_reply, _has_stock_data


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
        # yfinance returns this shape after a rate limit or unavailable range.
        empty_prices = pd.DataFrame()

        self.assertFalse(_has_stock_data(empty_prices))
        self.assertTrue(_has_stock_data(pd.DataFrame({"AAPL": [190.0]})))

    def test_empty_download_adds_assistant_reply_for_sse_response(self):
        messages = []

        _append_stock_data_failure_reply(messages)

        self.assertEqual(messages[-1].role, "assistant")
        self.assertIn("Yahoo Finance", messages[-1].content)
        self.assertIn("rate limited", messages[-1].content)


if __name__ == "__main__":
    unittest.main()

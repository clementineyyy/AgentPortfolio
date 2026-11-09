import unittest

from agent.request_models import AgentRunInput, normalize_agent_state


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


if __name__ == "__main__":
    unittest.main()

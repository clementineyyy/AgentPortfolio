from ag_ui.core import RunAgentInput


class AgentRunInput(RunAgentInput):
    """AG-UI request shape used by the frontend HTTP client."""

    protocol_version: str


def normalize_agent_state(state: dict) -> dict:
    """Fill the portfolio state fields required by a fresh agent run."""
    return {
        **state,
        "available_cash": state.get("available_cash", 1_000_000),
        "investment_summary": state.get("investment_summary", {}),
        "investment_portfolio": state.get("investment_portfolio", []),
    }

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from agent.query_routing import process_stock_request


EVAL_CASES = Path(__file__).parents[1] / "evals" / "stock_request_intents.jsonl"


def tool_response(name, arguments):
    tool_call = SimpleNamespace(
        id=f"call-{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason="tool_calls",
                message=SimpleNamespace(tool_calls=[tool_call]),
            )
        ]
    )


def text_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class QueryRoutingTest(unittest.TestCase):
    def test_market_analysis_uses_text_reply_without_extraction(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            tool_response("classify_stock_request", {"intent": "market_analysis"}),
            text_response("Live market data is unavailable."),
        ]
        user_message = {"role": "user", "content": "Analyze Apple's current trend"}
        messages = [user_message]

        result = process_stock_request(client, messages, [], request_id="run-market")

        self.assertEqual(result.intent, "market_analysis")
        self.assertEqual(result.reply, "Live market data is unavailable.")
        self.assertIsNone(result.tool_calls)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        classifier_kwargs = client.chat.completions.create.call_args_list[0].kwargs
        reply_kwargs = client.chat.completions.create.call_args_list[1].kwargs
        self.assertEqual(classifier_kwargs["tools"][0]["function"]["name"], "classify_stock_request")
        self.assertNotIn("tools", reply_kwargs)
        self.assertEqual(messages[1], user_message)

    def test_investment_calls_extractor_and_returns_validated_tool_call(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            tool_response("classify_stock_request", {"intent": "investment"}),
            tool_response(
                "extract_relevant_data_from_user_prompt",
                {
                    "ticker_symbols": ["AAPL"],
                    "investment_date": "2026-01-01",
                    "amount_of_dollars_to_be_invested": [5000],
                    "interval_of_investment": "single_shot",
                    "to_be_added_in_portfolio": True,
                },
            ),
        ]

        result = process_stock_request(
            client,
            [{"role": "user", "content": "Invest in Apple with 5k dollars since Jan 2026"}],
            [],
            request_id="run-investment",
        )

        self.assertEqual(result.intent, "investment")
        self.assertIsNone(result.reply)
        self.assertEqual(result.tool_calls[0]["function"]["name"], "extract_relevant_data_from_user_prompt")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(
            client.chat.completions.create.call_args_list[1].kwargs["tool_choice"]["function"]["name"],
            "extract_relevant_data_from_user_prompt",
        )

    def test_missing_inputs_are_requested_even_if_model_guesses_them(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            tool_response("classify_stock_request", {"intent": "investment"}),
            tool_response(
                "extract_relevant_data_from_user_prompt",
                {
                    "ticker_symbols": ["AAPL"],
                    "investment_date": "2026-01-01",
                    "amount_of_dollars_to_be_invested": [5000],
                    "interval_of_investment": "single_shot",
                    "to_be_added_in_portfolio": True,
                },
            ),
        ]

        result = process_stock_request(
            client, [{"role": "user", "content": "Invest in Apple"}], []
        )

        self.assertEqual(result.intent, "investment")
        self.assertIsNone(result.tool_calls)
        self.assertIn("investment amount", result.reply)
        self.assertIn("start date", result.reply)

    def test_explicit_chinese_investment_details_pass_validation(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            tool_response("classify_stock_request", {"intent": "investment"}),
            tool_response(
                "extract_relevant_data_from_user_prompt",
                {
                    "ticker_symbols": ["AAPL"],
                    "investment_date": "2026-01-01",
                    "amount_of_dollars_to_be_invested": [5000],
                    "interval_of_investment": "single_shot",
                    "to_be_added_in_portfolio": True,
                },
            ),
        ]

        result = process_stock_request(
            client,
            [{"role": "user", "content": "从2026年1月开始投资5000美元买苹果股票"}],
            [],
        )

        self.assertEqual(result.intent, "investment")
        self.assertIsNone(result.reply)
        self.assertEqual(len(result.tool_calls), 1)

    def test_model_call_logs_include_trace_without_prompt_or_arguments(self):
        client = Mock()
        client.chat.completions.create.side_effect = [
            tool_response("classify_stock_request", {"intent": "market_analysis"}),
            text_response("Reply"),
        ]

        with self.assertLogs("agent.query_routing", level="INFO") as captured:
            process_stock_request(
                client,
                [{"role": "user", "content": "Analyze Apple's current trend"}],
                [],
                request_id="trace-123",
            )

        output = "\n".join(captured.output)
        self.assertIn("trace-123", output)
        self.assertIn("classify_stock_request", output)
        self.assertNotIn("Analyze Apple's current trend", output)

    def test_intent_evaluation_dataset_has_valid_cases(self):
        cases = [json.loads(line) for line in EVAL_CASES.read_text(encoding="utf-8").splitlines()]

        self.assertGreaterEqual(len(cases), 8)
        self.assertTrue(
            all(case["expected_intent"] in {"investment", "market_analysis", "general"} for case in cases)
        )
        self.assertTrue(any(any(ord(char) > 127 for char in case["prompt"]) for case in cases))


if __name__ == "__main__":
    unittest.main()

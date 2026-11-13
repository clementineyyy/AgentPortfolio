import json
import os
import unittest
from unittest.mock import patch

from agent.market_data import MarketDataError, historical_prices


class MarketDataTest(unittest.TestCase):
    def setUp(self):
        historical_prices.cache_clear()
        self.env = patch.dict(os.environ, {"TWELVEDATA_API_KEY": "test-key"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        historical_prices.cache_clear()

    def test_normalizes_adjusted_provider_values_to_date_indexed_close(self):
        response = type("Response", (), {
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
            "read": lambda self: json.dumps({"values": [
                {"datetime": "2025-01-02", "close": "101.25"},
                {"datetime": "2024-10-02", "close": "98.50"},
            ]}).encode(),
        })()

        with patch("agent.market_data.urlopen", return_value=response) as request:
            prices = historical_prices("AAPL", "2024-10-01", "2025-01-03", "1day")

        self.assertEqual(prices.index.strftime("%Y-%m-%d").tolist(), ["2024-10-02", "2025-01-02"])
        self.assertEqual(prices["Close"].tolist(), [98.5, 101.25])
        self.assertIn("adjust=true", request.call_args.args[0])
        self.assertIn("interval=1day", request.call_args.args[0])

    def test_quarterly_request_samples_last_available_month_in_each_quarter(self):
        response = type("Response", (), {
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
            "read": lambda self: json.dumps({"values": [
                {"datetime": "2025-01-31", "close": "100"},
                {"datetime": "2025-02-28", "close": "105"},
                {"datetime": "2025-03-31", "close": "110"},
                {"datetime": "2025-04-30", "close": "112"},
            ]}).encode(),
        })()
        with patch("agent.market_data.urlopen", return_value=response):
            prices = historical_prices("AAPL", "2025-01-01", "2025-04-30", "3mo")

        self.assertEqual(prices.index.strftime("%Y-%m-%d").tolist(), ["2025-03-31", "2025-04-30"])
        self.assertEqual(prices["Close"].tolist(), [110.0, 112.0])

    def test_missing_key_fails_without_request(self):
        with patch.dict(os.environ, {}, clear=True), patch("agent.market_data.urlopen") as request:
            with self.assertRaises(MarketDataError):
                historical_prices("AAPL", "2024-01-01", "2025-01-01", "3mo")
        request.assert_not_called()

    def test_provider_error_does_not_return_fabricated_prices(self):
        response = type("Response", (), {
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
            "read": lambda self: b'{"status":"error","code":400,"message":"Invalid symbol"}',
        })()
        with patch("agent.market_data.urlopen", return_value=response):
            with self.assertRaisesRegex(MarketDataError, "Invalid symbol"):
                historical_prices("INVALID", "2024-01-01", "2025-01-01", "3mo")

    def test_malformed_provider_prices_raise_normalized_error(self):
        response = type("Response", (), {
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
            "read": lambda self: b'{"values":[{"close":"100"}]}',
        })()
        with patch("agent.market_data.urlopen", return_value=response):
            with self.assertRaisesRegex(MarketDataError, "invalid price data"):
                historical_prices("AAPL", "2024-01-01", "2025-01-01", "3mo")


if __name__ == "__main__":
    unittest.main()

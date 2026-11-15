import json
import os
import unittest
from unittest.mock import patch

import pandas as pd

from agent.market_data import (
    DividendDataConfigurationError,
    MarketDataConfigurationError,
    MarketDataError,
    historical_dividends,
    historical_prices,
)
from agent.stock_analysis import _append_stock_data_failure_reply


class MarketDataTest(unittest.TestCase):
    def setUp(self):
        historical_prices.cache_clear()
        historical_dividends.cache_clear()
        self.env = patch.dict(os.environ, {
            "TWELVEDATA_API_KEY": "test-key",
            "MASSIVE_API_KEY": "test-massive-key",
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()
        historical_prices.cache_clear()
        historical_dividends.cache_clear()

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
        self.assertIn("start_date=2024-10-01", request.call_args.args[0])
        self.assertIn("end_date=2025-01-03", request.call_args.args[0])
        self.assertNotIn("outputsize", request.call_args.args[0])

    def test_daily_request_preserves_each_trading_day(self):
        response = type("Response", (), {
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
            "read": lambda self: json.dumps({"values": [
                {"datetime": "2025-01-02", "close": "100"},
                {"datetime": "2025-01-03", "close": "105"},
                {"datetime": "2025-01-06", "close": "110"},
            ]}).encode(),
        })()
        with patch("agent.market_data.urlopen", return_value=response):
            prices = historical_prices("AAPL", "2025-01-01", "2025-01-06", "1day")

        self.assertEqual(prices.index.strftime("%Y-%m-%d").tolist(), ["2025-01-02", "2025-01-03", "2025-01-06"])
        self.assertEqual(prices["Close"].tolist(), [100.0, 105.0, 110.0])

    def test_missing_key_fails_without_request(self):
        with patch.dict(os.environ, {}, clear=True), patch("agent.market_data.urlopen") as request:
            with self.assertRaisesRegex(MarketDataError, "Historical market data is not configured"):
                historical_prices("AAPL", "2024-01-01", "2025-01-01", "3mo")
        request.assert_not_called()

    def test_missing_key_assistant_reply_identifies_backend_configuration(self):
        messages = []

        _append_stock_data_failure_reply(
            messages,
            MarketDataConfigurationError("Historical market data is not configured."),
        )

        self.assertIn("TWELVEDATA_API_KEY", messages[-1].content)
        self.assertIn("agent/.env", messages[-1].content)

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

    def test_massive_dividend_events_include_confirmed_pay_dates(self):
        response = type("Response", (), {
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
            "read": lambda self: json.dumps({"status": "OK", "results": [{
                "ticker": "AAPL",
                "ex_dividend_date": "2025-08-11",
                "pay_date": "2025-08-14",
                "cash_amount": 0.26,
                "split_adjusted_cash_amount": 0.26,
            }]}).encode(),
        })()
        with patch("agent.market_data.urlopen", return_value=response) as request:
            events = historical_dividends("AAPL", "2025-08-01", "2025-08-31")

        self.assertEqual(events.iloc[0]["ex_date"], pd.Timestamp("2025-08-11"))
        self.assertEqual(events.iloc[0]["pay_date"], pd.Timestamp("2025-08-14"))
        self.assertEqual(events.iloc[0]["amount"], 0.26)
        self.assertIn("api.massive.com/stocks/v1/dividends", request.call_args.args[0])

    def test_massive_dividend_events_without_pay_date_fail_instead_of_guessing(self):
        response = type("Response", (), {
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
            "read": lambda self: b'{"status":"OK","results":[{"ex_dividend_date":"2025-08-11","cash_amount":0.26}]}',
        })()
        with patch("agent.market_data.urlopen", return_value=response):
            with self.assertRaisesRegex(MarketDataError, "missing an ex-date, pay date, or split-adjusted amount"):
                historical_dividends("AAPL", "2025-08-01", "2025-08-31")

    def test_missing_massive_key_is_reported_as_dividend_configuration_error(self):
        with patch.dict(os.environ, {"TWELVEDATA_API_KEY": "test-key"}, clear=True):
            with self.assertRaises(DividendDataConfigurationError):
                historical_dividends("AAPL", "2025-08-01", "2025-08-31")

    def test_missing_massive_key_reply_names_required_backend_setting(self):
        messages = []

        _append_stock_data_failure_reply(
            messages,
            DividendDataConfigurationError("Dividend payment dates are not configured."),
        )

        self.assertIn("MASSIVE_API_KEY", messages[-1].content)


if __name__ == "__main__":
    unittest.main()

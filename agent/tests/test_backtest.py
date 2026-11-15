import unittest

import pandas as pd

from agent.stock_analysis import _build_backtest_date_notice, _daily_backtest_series


class BacktestDateNoticeTest(unittest.TestCase):
    def test_weekend_requested_date_explains_next_trading_entry(self):
        notice = _build_backtest_date_notice(
            "2026-01-03",
            {"AAPL": pd.Timestamp("2026-01-05")},
            "2026-01-05",
            "2026-01-05",
        )

        self.assertIn("2026-01-03", notice)
        self.assertIn("周末", notice)
        self.assertIn("2026-01-05", notice)

    def test_weekday_price_gap_is_reported_as_unknown(self):
        notice = _build_backtest_date_notice(
            "2026-01-02",
            {"AAPL": pd.Timestamp("2026-01-05")},
            "2026-01-05",
            "2026-01-05",
        )

        self.assertIn("具体原因无法确认", notice)

    def test_latest_date_before_weekend_as_of_date_is_explained(self):
        notice = _build_backtest_date_notice(
            "2026-01-02",
            {"AAPL": pd.Timestamp("2026-01-02")},
            "2026-01-02",
            "2026-01-04",
        )

        self.assertIn("当前日期是 2026-01-04", notice)
        self.assertIn("周末", notice)


class DailyBacktestSeriesTest(unittest.TestCase):
    def test_daily_values_include_dividend_cash_only_on_pay_date(self):
        prices = pd.DataFrame(
            {"AAPL": [10.0, 11.0, 12.0]},
            index=pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
        )
        dividends = {
            "AAPL": pd.DataFrame([{
                "ex_date": pd.Timestamp("2026-01-05"),
                "pay_date": pd.Timestamp("2026-01-06"),
                "amount": 0.25,
            }])
        }

        points, total_dividends = _daily_backtest_series(
            prices,
            {"AAPL": 2.0},
            {"AAPL": pd.Timestamp("2026-01-02")},
            dividends,
        )

        self.assertEqual([point["portfolio"] for point in points], [20.0, 22.0, 24.5])
        self.assertEqual(total_dividends["AAPL"], 0.5)

    def test_purchase_at_close_on_ex_date_is_not_dividend_eligible(self):
        prices = pd.DataFrame(
            {"AAPL": [10.0, 11.0]},
            index=pd.to_datetime(["2026-01-05", "2026-01-06"]),
        )
        dividends = {
            "AAPL": pd.DataFrame([{
                "ex_date": pd.Timestamp("2026-01-05"),
                "pay_date": pd.Timestamp("2026-01-06"),
                "amount": 0.25,
            }])
        }

        points, total_dividends = _daily_backtest_series(
            prices,
            {"AAPL": 1.0},
            {"AAPL": pd.Timestamp("2026-01-05")},
            dividends,
        )

        self.assertEqual([point["portfolio"] for point in points], [10.0, 11.0])
        self.assertEqual(total_dividends["AAPL"], 0.0)


if __name__ == "__main__":
    unittest.main()

"""Historical market data providers and their normalized price interface."""

from functools import lru_cache
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


class MarketDataError(RuntimeError):
    """Raised when historical prices cannot be retrieved or normalized."""


class MarketDataConfigurationError(MarketDataError):
    """Raised when the backend provider credentials are not configured."""


class DividendDataError(MarketDataError):
    """Raised when historical dividend events cannot be retrieved or normalized."""


class DividendDataConfigurationError(MarketDataConfigurationError):
    """Raised when the dividend provider credentials are not configured."""


@lru_cache(maxsize=128)
def historical_prices(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Return adjusted close prices indexed by date for a US equity symbol."""
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        raise MarketDataConfigurationError("Historical market data is not configured.")

    params = urlencode({
        "symbol": symbol,
        "interval": interval,
        "start_date": start_date,
        "end_date": end_date,
        "order": "ASC",
        "adjust": "true",
        "apikey": api_key,
    })
    url = f"https://api.twelvedata.com/time_series?{params}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") == "error" or "values" not in payload:
                message = str(payload.get("message", "No price history was returned."))
                if (payload.get("code") == 429 or "rate limit" in message.lower()) and attempt < 2:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                raise MarketDataError(message)
            values = payload["values"]
            if not values:
                raise MarketDataError("No price history was returned for this symbol and date range.")
            frame = pd.DataFrame(values)
            frame.index = pd.to_datetime(frame.pop("datetime"))
            frame.index.name = "Date"
            frame["Close"] = pd.to_numeric(frame["close"], errors="coerce")
            frame = frame.loc[:, ["Close"]].dropna().sort_index()
            if frame.empty:
                raise MarketDataError("No usable prices were returned for this symbol and date range.")
            return frame
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 2:
                raise MarketDataError("The market data provider is temporarily unavailable.") from error
            time.sleep(0.25 * (2 ** attempt))
        except json.JSONDecodeError as error:
            if attempt == 2:
                raise MarketDataError("The market data provider returned an invalid response.") from error
            time.sleep(0.25 * (2 ** attempt))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise MarketDataError("The market data provider returned invalid price data.") from error
    raise MarketDataError("The market data provider is temporarily unavailable.")


@lru_cache(maxsize=128)
def historical_dividends(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Return dividend events with confirmed ex-dates, pay dates and split-adjusted amounts."""
    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        raise DividendDataConfigurationError(
            "Dividend payment dates are not configured. Set MASSIVE_API_KEY."
        )

    params = urlencode({
        "ticker": symbol,
        "ex_dividend_date.gte": start_date,
        "ex_dividend_date.lte": end_date,
        "limit": "5000",
        "sort": "ex_dividend_date.asc",
        "apiKey": api_key,
    })
    url = f"https://api.massive.com/stocks/v1/dividends?{params}"
    for attempt in range(3):
        try:
            with urlopen(url, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if payload.get("status") not in (None, "OK") or "results" not in payload:
                message = str(payload.get("error", payload.get("message", "Dividend history was not returned.")))
                if ("rate limit" in message.lower() or "429" in message) and attempt < 2:
                    time.sleep(0.25 * (2 ** attempt))
                    continue
                raise DividendDataError(message)

            events = []
            for item in payload["results"]:
                ex_date = item.get("ex_dividend_date")
                pay_date = item.get("pay_date")
                amount = item.get("split_adjusted_cash_amount")
                if not ex_date or not pay_date or amount is None:
                    raise DividendDataError(
                        f"Dividend history for {symbol} is missing an ex-date, pay date, or split-adjusted amount."
                    )
                events.append({
                    "ex_date": pd.Timestamp(ex_date),
                    "pay_date": pd.Timestamp(pay_date),
                    "amount": float(amount),
                })
            return pd.DataFrame(events, columns=["ex_date", "pay_date", "amount"])
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == 2:
                raise DividendDataError("The dividend data provider is temporarily unavailable.") from error
            time.sleep(0.25 * (2 ** attempt))
        except json.JSONDecodeError as error:
            if attempt == 2:
                raise DividendDataError("The dividend data provider returned an invalid response.") from error
            time.sleep(0.25 * (2 ** attempt))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise DividendDataError("The dividend data provider returned invalid dividend data.") from error
    raise DividendDataError("The dividend data provider is temporarily unavailable.")

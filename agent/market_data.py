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


@lru_cache(maxsize=128)
def historical_prices(symbol: str, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
    """Return adjusted close prices indexed by date for a US equity symbol."""
    api_key = os.getenv("TWELVEDATA_API_KEY")
    if not api_key:
        raise MarketDataError("Historical market data is not configured.")

    params = urlencode({
        "symbol": symbol,
        "interval": "1month" if interval == "3mo" else interval,
        "start_date": start_date,
        "end_date": end_date,
        "order": "ASC",
        "adjust": "true",
        "outputsize": "5000",
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
            if interval == "3mo":
                frame = frame.resample("QE").last().dropna()
                frame = frame.loc[frame.index <= pd.Timestamp(end_date)]
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

# Daily Historical Backtest Semantics

**Status:** Accepted

Historical investment scenarios use daily market data from the requested date through the latest complete daily bar available from Twelve Data. A one-time investment is entered at the first trading day's close on or after the requested date, and holdings are valued at each trading day's close; non-trading dates have no data point, while the chart connects consecutive trading observations. The requested date and latest available date are explained in the conversation whenever they differ from the actual entry or current date, and unknown causes are reported as unknown.

Request each symbol, including SPY, only for the backtest interval; use both `start_date` and `end_date` without `outputsize`, and never shorten portfolio history to align the benchmark. Permit fractional shares. Twelve Data supplies daily prices; Massive supplies dividend pay dates because Twelve Data's dividends response does not include them. Credit eligible dividends as cash on their payment date without reinvestment. Exclude fees, taxes, and bid-ask spread, and disclose those exclusions. These conventions make the chart a reproducible end-of-day historical simulation without implying an exact real-world fill or a forecast.

Massive's documented free Stocks Basic plan provides two years of dividend history; requests outside the configured plan's available range must not be treated as complete dividend coverage. The backend requires `MASSIVE_API_KEY` and fails the simulation with a visible configuration/data error rather than inventing a pay date.

References: [Twelve Data historical interval and range parameters](https://support.twelvedata.com/en/articles/5656039-how-to-get-historical-prices), [Twelve Data dividend fields](https://twelvedata.com/docs/fundamentals/dividends), [Massive dividend pay-date fields and plan history](https://massive.com/docs/rest/stocks/corporate-actions/dividends?auth=signup).

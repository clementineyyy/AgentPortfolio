# Stock Portfolio Analysis

This context turns user-specified investment scenarios and market data into portfolio analysis. It distinguishes historical simulations from forecasts and real trades.

## Language

**Historical backtest**:
A simulation of an investment rule against historical market data.
_Avoid_: Forecast, actual performance

**Requested investment date**:
The calendar date the user names as the start of a historical investment scenario.
_Avoid_: Trade date

**Entry date**:
The first trading date on or after the requested investment date when the simulated purchase is made.
_Avoid_: Requested date, start date

**Daily valuation**:
The end-of-day value of simulated holdings and cash on each trading date in the backtest.
_Avoid_: Daily execution

**Benchmark**:
A reference investment, such as SPY, simulated over the same date range to compare with the portfolio.
_Avoid_: Portfolio holding

**Dividend cash**:
Cash paid to the simulated portfolio for shares eligible on an ex-dividend date, credited on the payment date.
_Avoid_: Reinvested dividend

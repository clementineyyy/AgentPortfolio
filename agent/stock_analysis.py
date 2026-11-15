
# ===============================================================================
# IMPORTS AND SETUP
# ===============================================================================

# CrewAI Flow framework for building multi-step AI workflows
from crewai.flow.flow import Flow, start, router, listen, or_
from litellm import completion
from pydantic import BaseModel
from typing import Literal, List

# AG UI types for message handling and state management
from ag_ui.core.types import AssistantMessage, ToolMessage
from ag_ui.core.events import StateDeltaEvent, EventType

# Standard Python libraries
import uuid  
import asyncio  
import json  
import os  
from datetime import date as date_type, datetime

# External libraries
from openai import OpenAI  
from dotenv import load_dotenv  
import numpy as np  
import pandas as pd  

# Import custom prompts for the AI models
if __package__:
    from .prompts import insights_prompt
    from .query_routing import process_stock_request, tool_result_messages
    from .market_data import (
        DividendDataConfigurationError,
        DividendDataError,
        MarketDataConfigurationError,
        MarketDataError,
        historical_dividends,
        historical_prices,
    )
else:
    from prompts import insights_prompt
    from query_routing import process_stock_request, tool_result_messages
    from market_data import (
        DividendDataConfigurationError,
        DividendDataError,
        MarketDataConfigurationError,
        MarketDataError,
        historical_dividends,
        historical_prices,
    )

# Load environment variables (like API keys) from .env file
load_dotenv()


def _has_stock_data(data):
    """Return whether a downloaded price frame contains usable rows."""
    return isinstance(data, pd.DataFrame) and not data.empty


def _append_stock_data_failure_reply(messages, error=None):
    """Ensure a failed market-data lookup produces a user-visible reply."""
    if isinstance(error, DividendDataConfigurationError):
        content = (
            "Dividend payment date data is not configured. Set MASSIVE_API_KEY in "
            "agent/.env to include historical cash dividends in the simulation."
        )
    elif isinstance(error, DividendDataError):
        content = (
            "I couldn't retrieve complete dividend payment-date data. "
            "Check the Massive API key and dividend-history access for this date range, then try again."
        )
    elif isinstance(error, MarketDataConfigurationError):
        content = (
            "Historical market data is not configured. Set TWELVEDATA_API_KEY in "
            "agent/.env, then restart the backend server."
        )
    else:
        content = (
            "I couldn't retrieve the required historical price data. "
            "Check the stock symbols and date range, then try again later."
        )
    messages.append(
        AssistantMessage(
            role="assistant",
            content=content,
            id=str(uuid.uuid4()),
        )
    )


def _build_backtest_date_notice(requested_date, entry_dates, latest_date, as_of_date):
    """Explain requested-versus-actual dates without guessing at market closures."""
    requested = pd.Timestamp(requested_date).normalize()
    latest = pd.Timestamp(latest_date).normalize()
    as_of = pd.Timestamp(as_of_date).normalize()
    notices = []

    shifted_entries = {
        ticker: pd.Timestamp(entry_date).normalize()
        for ticker, entry_date in entry_dates.items()
        if pd.Timestamp(entry_date).normalize() > requested
    }
    if shifted_entries:
        if requested.weekday() >= 5:
            reason = "指定日期是周末，市场休市"
        else:
            reason = "Twelve Data 未返回指定日期的价格，具体原因无法确认"
        details = ", ".join(
            f"{ticker} 于 {entry_date.date()} 开始"
            for ticker, entry_date in shifted_entries.items()
        )
        notices.append(
            f"你指定的开始日期是 {requested.date()}；{reason}，因此模拟从首个返回的交易日开始：{details}。"
        )

    if latest < as_of:
        if as_of.weekday() >= 5:
            reason = "当前日期是周末，市场没有新的交易日收盘数据"
        else:
            reason = "数据源尚未返回更新的完整日线，具体原因无法确认"
        notices.append(
            f"当前日期是 {as_of.date()}，本次行情数据截至 {latest.date()}；{reason}。"
        )
    return "\n\n".join(notices)


def _daily_backtest_series(prices, positions, entry_dates, dividends, position_history=None):
    """Build trading-day portfolio and SPY values, crediting eligible paid dividends."""
    events = {}
    for ticker, frame in dividends.items():
        events[ticker] = frame.to_dict("records")

    result = []
    dividend_cash = {ticker: 0.0 for ticker in positions}
    credited = set()
    for day in prices.index.sort_values():
        for ticker, ticker_events in events.items():
            entry_date = entry_dates.get(ticker)
            if entry_date is None or day < entry_date:
                continue
            for idx, event in enumerate(ticker_events):
                key = (ticker, idx)
                ex_date = pd.Timestamp(event["ex_date"]).normalize()
                pay_date = pd.Timestamp(event["pay_date"]).normalize()
                if position_history:
                    prior_dates = [trade_day for trade_day in position_history if trade_day < ex_date]
                    eligible_shares = (
                        position_history[max(prior_dates)].get(ticker, 0.0)
                        if prior_dates else 0.0
                    )
                else:
                    eligible_shares = positions.get(ticker, 0.0) if entry_date < ex_date else 0.0
                if eligible_shares > 0 and pay_date <= day and key not in credited:
                    dividend_cash[ticker] += eligible_shares * float(event["amount"])
                    credited.add(key)

        portfolio_value = 0.0
        for ticker, shares in positions.items():
            if position_history:
                shares = position_history.get(day, {}).get(ticker, 0.0)
            if day < entry_dates.get(ticker, pd.Timestamp.max):
                continue
            price = prices.at[day, ticker] if ticker in prices else np.nan
            if not pd.isna(price):
                portfolio_value += shares * float(price)
            portfolio_value += dividend_cash.get(ticker, 0.0)

        result.append({"date": str(day.date()), "portfolio": portfolio_value})
    return result, dividend_cash


def _plan_single_shot_transactions(prices, transactions, starting_cash):
    """Apply each one-time purchase at its own requested entry date."""
    tickers = list(dict.fromkeys(item["ticker"] for item in transactions))
    holdings = {ticker: 0.0 for ticker in tickers}
    invested = {ticker: 0.0 for ticker in tickers}
    entry_dates = {}
    investment_log = []
    add_funds_dates = []
    position_history = {}
    total_cash = float(starting_cash)
    transactions_by_date = {}

    for transaction in transactions:
        ticker = transaction["ticker"]
        requested_date = pd.Timestamp(transaction["investment_date"]).normalize()
        ticker_prices = prices.loc[prices.index >= requested_date, ticker].dropna()
        if ticker_prices.empty:
            return {"unavailable_ticker": ticker}
        entry_date = ticker_prices.index[0]
        transactions_by_date.setdefault(entry_date, []).append(
            (ticker, float(transaction["amount"]), float(ticker_prices.iloc[0]))
        )

    for day in prices.index.sort_values():
        for ticker, amount, price in transactions_by_date.get(day, []):
            entry_dates.setdefault(ticker, day)
            if total_cash >= amount and amount > 0 and price > 0:
                shares = amount / price
                holdings[ticker] += shares
                invested[ticker] += amount
                total_cash -= amount
                investment_log.append(
                    f"{day.date()}: Bought {shares:.6f} shares of {ticker} at the daily close ${price:.2f} (cost: ${amount:.2f})"
                )
            else:
                add_funds_dates.append((str(day.date()), ticker, price, total_cash))
                investment_log.append(
                    f"{day.date()}: Not enough total cash to buy {ticker}. Allocated: ${amount:.2f}, Available: ${total_cash:.2f}"
                )
        position_history[day] = holdings.copy()

    return {
        "holdings": holdings,
        "total_invested_per_stock": invested,
        "entry_dates": entry_dates,
        "investment_log": investment_log,
        "add_funds_dates": add_funds_dates,
        "total_cash": total_cash,
        "position_history": position_history,
    }


def _field(value, name, default=None):
    """Read a field from AG-UI's dict or typed message representations."""
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _chart_action_response(messages):
    """Return the user's chart decision when resuming a chart action."""
    if not messages:
        return None

    response = messages[-1]
    if _field(response, "role") != "tool":
        return None

    tool_call_id = _field(response, "tool_call_id")
    if not tool_call_id:
        return None

    chart_actions = {"render_standard_charts_and_table", "render_custom_charts"}
    for message in reversed(messages[:-1]):
        if _field(message, "role") != "assistant":
            continue
        for call in _field(message, "tool_calls", []) or []:
            if _field(call, "id") != tool_call_id:
                continue
            function = _field(call, "function", {})
            if _field(function, "name") not in chart_actions:
                return None

            content = str(_field(response, "content", "")).lower()
            if "reject" in content or "rejected" in content:
                return "rejected"
            if "success" in content or "accept" in content:
                return "accepted"
            return None
    return None


def _resume_chart_action(messages):
    """Complete a chart action response without restarting investment analysis."""
    decision = _chart_action_response(messages)
    if not decision:
        return False

    decision_text = (
        "The simulated investment was accepted. The chart and portfolio summary are shown above."
        if decision == "accepted"
        else "The simulated investment was rejected. No changes were made to the portfolio."
    )
    messages.append(
        AssistantMessage(id=str(uuid.uuid4()), role="assistant", content=decision_text)
    )
    return True


# Tool definition for generating bull/bear insights about stocks or portfolios
# This tool generates positive (bullish) and negative (bearish) analysis
# to provide balanced perspective on investment decisions
generate_insights = {
  "type": "function",
  "function": {
    "name": "generate_insights",
    "description": "Generate positive (bull) and negative (bear) insights for a stock or portfolio.",
    "parameters": {
      "type": "object",
      "properties": {
        # Positive insights (reasons why the investment might perform well)
        "bullInsights": {
          "type": "array",
          "description": "A list of positive insights (bull case) for the stock or portfolio.",
          "items": {
            "type": "object",
            "properties": {
              "title": {
                "type": "string",
                "description": "Short title for the positive insight."
              },
              "description": {
                "type": "string",
                "description": "Detailed description of the positive insight."
              },
              "emoji": {
                "type": "string",
                "description": "Emoji representing the positive insight."
              }
            },
            "required": ["title", "description", "emoji"]
          }
        },
        # Negative insights (potential risks or concerns about the investment)
        "bearInsights": {
          "type": "array",
          "description": "A list of negative insights (bear case) for the stock or portfolio.",
          "items": {
            "type": "object",
            "properties": {
              "title": {
                "type": "string",
                "description": "Short title for the negative insight."
              },
              "description": {
                "type": "string",
                "description": "Detailed description of the negative insight."
              },
              "emoji": {
                "type": "string",
                "description": "Emoji representing the negative insight."
              }
            },
            "required": ["title", "description", "emoji"]
          }
        }
      },
      "required": ["bullInsights", "bearInsights"]
    }
  }
}
# ===============================================================================
# MAIN FLOW CLASS
# ===============================================================================

class StockAnalysisFlow(Flow):
    """
    Main workflow class that orchestrates the stock analysis process.
    This flow consists of multiple stages:
    1. start() - Initialize the request-routing system prompt
    2. chat() - Classify intent and extract parameters only for investments
    3. simulation() - Gather historical stock data
    4. allocation() - Calculate portfolio allocation and performance
    5. insights() - Generate bull/bear insights about the investments
    6. end() - Return final state
    """
    
    @start()
    def start(self):
        """
        Step 1: Initialize the workflow state.
        """
        return self.state

    @listen("start")
    async def chat(self):
        """
        Step 2: Classify the user request and extract investment parameters only
        when its intent is investment.
        - Create a tool log entry to show progress to the user
        - Route market analysis and general questions to a text response
        - Ask for missing investment details before simulation
        """
        try:
            if _resume_chart_action(self.state['state']["messages"]):
                return "end"

            # Step 2.1: Create a new tool log entry to track progress
            tool_log_id = str(uuid.uuid4())
            self.state['state']["tool_logs"].append(
                {
                    "id": tool_log_id,
                    "message": "Analyzing user query",
                    "status": "processing",
                }
            )
            
            # Step 2.2: Emit state change event to update UI with new tool log
            self.state.get("emit_event")(
                StateDeltaEvent(
                    type=EventType.STATE_DELTA,
                    delta=[
                        {
                            "op": "add",
                            "path": "/tool_logs/-",
                            "value": {
                                "message": "Analyzing user query",
                                "status": "processing",
                                "id": tool_log_id,
                            },
                        }
                    ],
                )
            )
            await asyncio.sleep(0)  # Allow other tasks to run
            
            # Step 2.3: Classify first. Only investment requests proceed to
            # the investment extraction tool.
            model = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            )
            outcome = process_stock_request(
                model,
                self.state['state']['messages'],
                self.state["investment_portfolio"],
                request_id=self.state.get("request_id"),
            )
            
            # Step 2.4: Update tool log status to completed
            index = len(self.state['state']["tool_logs"]) - 1
            self.state.get("emit_event")(
                StateDeltaEvent(
                    type=EventType.STATE_DELTA,
                    delta=[
                        {
                            "op": "replace",
                            "path": f"/tool_logs/{index}/status",
                            "value": "completed",
                        }
                    ],
                )
            )
            await asyncio.sleep(0)
            
            if outcome.reply is not None:
                self.state['state']["messages"].append(
                    AssistantMessage(
                        id=str(uuid.uuid4()), content=outcome.reply, role="assistant"
                    )
                )
                return "end"

            if not outcome.tool_calls:
                raise RuntimeError("A validated investment request had no extraction tool call.")

            self.state['state']["messages"].append(
                AssistantMessage(
                    role="assistant",
                    tool_calls=outcome.tool_calls,
                    id=str(uuid.uuid4()),
                )
            )
            self.state['state']["messages"].extend(
                tool_result_messages(
                    outcome.tool_calls,
                    "Investment parameters extracted successfully",
                )
            )
            return "simulation"

        except Exception as e:
            # Fail the flow so downstream stages cannot present an API failure as success.
            print(f"Error in chat method ({type(e).__name__})")
            raise
    
    
    @listen("chat")
    async def simulation(self):
        """
        Step 3: Gather historical stock data for analysis
        - Extract investment parameters from the previous step
        - Download historical stock price data from the configured provider
        - Prepare data for portfolio simulation
        """
        # Step 3.1: Ensure we have tool calls with investment data
        # Find the last AssistantMessage with tool calls
        last_assistant_message = None
        for message in reversed(self.state['state']['messages']):
            if hasattr(message, 'tool_calls') and message.tool_calls is not None:
                last_assistant_message = message
                break
        
        if last_assistant_message is None:
            return "end"
            
        # Step 3.2: Create tool log entry for stock data gathering
        tool_log_id = str(uuid.uuid4())
        self.state['state']["tool_logs"].append(
            {
                "id": tool_log_id,
                "message": "Gathering Stock Data",
                "status": "processing",
            }
        )
        
        # Step 3.3: Emit state change to update UI
        self.state.get("emit_event")(
            StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "add",
                        "path": "/tool_logs/-",
                        "value": {
                            "message": "Gathering Stock Data",
                            "status": "processing",
                            "id": tool_log_id,
                        },
                    }
                ],
            )
        )
        await asyncio.sleep(0)
        
        # Step 3.4: Parse the extracted investment arguments from previous step
        arguments = json.loads(last_assistant_message.tool_calls[0].function.arguments)
        print(f"Debug: Parsed arguments: {arguments}")
        print(f"Debug: Available keys: {list(arguments.keys())}")
        
        # Step 3.5: Create investment portfolio structure for UI display
        # Combine new investments with existing portfolio (additive approach)
        existing_portfolio = self.state.get("investment_portfolio", [])
        if isinstance(existing_portfolio, str):
            existing_portfolio = json.loads(existing_portfolio)
        
        # Check if this is the final results or initial parameters
        if "investment_summary" in arguments:
            print("Debug: Received final results, skipping simulation step")
            return "end"
        
        # Check if required keys exist in arguments
        if "investments" not in arguments:
            print(f"Error: Missing required keys in arguments. Available keys: {list(arguments.keys())}")
            return "end"
        
        # Create new investments list
        new_investments = [
            {
                "ticker": transaction["ticker"],
                "amount": transaction["amount"],
                "investment_date": transaction["investment_date"],
            }
            for transaction in arguments["investments"]
        ]
        
        # Combine existing and new investments
        combined_portfolio = existing_portfolio + new_investments
        self.investment_portfolio = json.dumps(combined_portfolio)
        
        # Step 3.7: Extract investment parameters
        tickers = list(dict.fromkeys(item["ticker"] for item in new_investments))
        investment_date = min(item["investment_date"] for item in new_investments)
        
        # Step 3.10: Download historical stock data through the provider boundary
        # Get all tickers from combined portfolio (existing + new)
        all_tickers = list(set(tickers + [inv["ticker"] for inv in existing_portfolio]))
        print(f"Debug: Downloading data for all tickers: {all_tickers}")
        
        end_date = date_type.today().isoformat()
        market_data_error = None
        try:
            frames = [
                historical_prices(ticker, investment_date, end_date, "1day").rename(
                    columns={"Close": ticker}
                )
                for ticker in all_tickers
            ]
            self.be_stock_data = pd.concat(frames, axis=1).sort_index()
        except MarketDataError as error:
            market_data_error = error
            print(f"Market data lookup failed: {error}")
            self.be_stock_data = pd.DataFrame()

        # Step 3.11: Store normalized closing prices for the simulation
        self.be_arguments = arguments  # Store extracted arguments for next step
        
        # Check if stock data is empty
        if not _has_stock_data(self.be_stock_data):
            print("Warning: No stock data retrieved. This might be due to invalid tickers or date range.")
            _append_stock_data_failure_reply(
                self.state['state']["messages"], market_data_error
            )
            # Keep the failed download visible in the UI and do not let a
            # downstream listener treat the empty frame as investable data.
            index = len(self.state['state']["tool_logs"]) - 1
            self.state.get("emit_event")(
                StateDeltaEvent(
                    type=EventType.STATE_DELTA,
                    delta=[
                        {
                            "op": "replace",
                            "path": f"/tool_logs/{index}/status",
                            "value": "failed",
                        },
                        {
                            "op": "add",
                            "path": "/tool_logs/-",
                            "value": {
                                "id": str(uuid.uuid4()),
                                "message": "No stock price data was returned. Please try again later.",
                                "status": "failed",
                            },
                        },
                    ],
                )
            )
            return "end"

        # Publish the new holdings only after all required prices are available.
        self.state.get("emit_event")(
            StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[{
                    "op": "replace",
                    "path": "/investment_portfolio",
                    "value": json.loads(self.investment_portfolio),
                }],
            )
        )
        await asyncio.sleep(0)
        
        # Step 3.12: Mark stock data gathering as completed
        index = len(self.state['state']["tool_logs"]) - 1
        self.state.get("emit_event")(
            StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "replace",
                        "path": f"/tool_logs/{index}/status",
                        "value": "completed",
                    }
                ],
            )
        )
        await asyncio.sleep(0)
        
        # Step 3.13: Proceed to portfolio allocation and simulation
        return "allocation"
    
    
    @listen("simulation")
    async def allocation(self):
        """
        Step 4: Calculate portfolio allocation and performance simulation
        - Simulate buying stocks based on investment strategy (single-shot vs DCA)
        - Calculate returns, allocation percentages, and performance metrics
        - Compare portfolio performance against SPY (S&P 500) benchmark
        - Generate performance data for charting
        """
        # CrewAI listeners can still run after simulation returns its routing
        # value. A provider may return an empty frame when data is unavailable.
        stock_data = getattr(self, "be_stock_data", None)
        if not _has_stock_data(stock_data):
            return "end"

        # Step 4.1: Ensure we have tool calls with investment data
        # Find the last AssistantMessage with tool calls
        last_assistant_message = None
        for message in reversed(self.state['state']['messages']):
            if hasattr(message, 'tool_calls') and message.tool_calls is not None:
                last_assistant_message = message
                break
        
        if last_assistant_message is None:
            return "end"
            
        # Check if this is the final results or initial parameters
        try:
            arguments = json.loads(last_assistant_message.tool_calls[0].function.arguments)
            if "investment_summary" in arguments:
                print("Debug: Received final results, skipping allocation step")
                return "end"
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
            
        # Step 4.2: Create tool log for allocation calculation
        tool_log_id = str(uuid.uuid4())
        # Step 4.2: Create tool log for allocation calculation
        tool_log_id = str(uuid.uuid4())
        self.state['state']["tool_logs"].append(
            {
                "id": tool_log_id,
                "message": "Calculating portfolio allocation",
                "status": "processing",
            }
        )
        
        # Step 4.3: Emit state change to update UI
        self.state.get("emit_event")(
            StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "add",
                        "path": "/tool_logs/-",
                        "value": {
                            "message": "Allocating cash",
                            "status": "processing",
                            "id": tool_log_id,
                        },
                    }
                ],
            )
        )
        await asyncio.sleep(0)
        
        # Step 4.4: Extract data from previous steps
        stock_data = self.be_stock_data  # DataFrame: index=date, columns=tickers
        args = self.be_arguments
        transactions = args["investments"]
        current_tickers = list(dict.fromkeys(item["ticker"] for item in transactions))
        investment_date = min(item["investment_date"] for item in transactions)
        amounts_by_ticker = {ticker: 0.0 for ticker in current_tickers}
        for transaction in transactions:
            amounts_by_ticker[transaction["ticker"]] += float(transaction["amount"])
        amounts = [amounts_by_ticker[ticker] for ticker in current_tickers]
        interval = args.get("interval_of_investment", "single_shot")
        
        # Get all tickers from combined portfolio
        existing_portfolio = self.state.get("investment_portfolio", [])
        if isinstance(existing_portfolio, str):
            existing_portfolio = json.loads(existing_portfolio)
        
        all_tickers = list(set(current_tickers + [inv["ticker"] for inv in existing_portfolio]))
        print(f"Debug: Processing allocation for all tickers: {all_tickers}")

        # Step 4.5: Initialize cash available for investment
        # Use existing available cash or sum of requested amounts
        if self.state['state']["available_cash"] is not None:
            total_cash = self.state['state']["available_cash"]
        else:
            total_cash = sum(amounts)
            
        # Step 4.6: Initialize tracking variables for simulation
        # Get existing portfolio from state
        existing_portfolio = self.state.get("investment_portfolio", [])
        if isinstance(existing_portfolio, str):
            existing_portfolio = json.loads(existing_portfolio)
        
        # Initialize holdings with existing portfolio
        holdings = {}
        for investment in existing_portfolio:
            ticker = investment["ticker"]
            if ticker not in holdings:
                holdings[ticker] = 0.0
        
        # Add new tickers from current query
        for ticker in current_tickers:
            if ticker not in holdings:
                holdings[ticker] = 0.0
        
        investment_log = []  # Log of all investment transactions
        add_funds_needed = False  # Flag if more funds are needed
        add_funds_dates = []  # Dates when funds were insufficient

        # Step 4.7: Ensure stock data is sorted chronologically
        stock_data = stock_data.sort_index()
        requested_date = pd.Timestamp(investment_date).normalize()
        entry_dates = {}
        total_invested_per_stock = {ticker: 0.0 for ticker in all_tickers}
        position_history = {}

        # Step 4.8: Execute investment strategy based on interval
        if interval == "single_shot":
            plan = _plan_single_shot_transactions(stock_data, transactions, total_cash)
            if "unavailable_ticker" in plan:
                ticker = plan["unavailable_ticker"]
                _append_stock_data_failure_reply(
                    self.state["state"]["messages"],
                    MarketDataError(f"No price data was returned for {ticker} in the requested date range."),
                )
                self.be_stock_data = pd.DataFrame()
                return "end"
            holdings.update(plan["holdings"])
            entry_dates.update(plan["entry_dates"])
            total_invested_per_stock.update(plan["total_invested_per_stock"])
            investment_log.extend(plan["investment_log"])
            add_funds_dates.extend(plan["add_funds_dates"])
            add_funds_needed = bool(add_funds_dates)
            total_cash = plan["total_cash"]
            position_history = plan["position_history"]
        else:
            # DOLLAR-COST AVERAGING (DCA) STRATEGY: Spread investments over time
            for date, row in stock_data.iterrows():
                for i, ticker in enumerate(current_tickers):
                    price = row[ticker]
                    
                    # Step 4.8.5: Skip if no price data available
                    if np.isnan(price):
                        continue  # skip if price is NaN
                        
                    # Step 4.8.6: Invest as much as possible for this ticker at this date
                    if total_cash >= price:
                        shares_to_buy = total_cash // price
                        if shares_to_buy > 0:
                            cost = shares_to_buy * price
                            holdings[ticker] += shares_to_buy
                            total_cash -= cost
                        total_invested_per_stock[ticker] += cost
                        entry_dates.setdefault(ticker, date)
                        investment_log.append(
                            f"{date.date()}: Bought {shares_to_buy:.2f} shares of {ticker} at ${price:.2f} (cost: ${cost:.2f})"
                        )
                    else:
                        # Step 4.8.7: Log when funds are insufficient
                        add_funds_needed = True
                        add_funds_dates.append(
                            (str(date.date()), ticker, price, total_cash)
                        )
                        investment_log.append(
                            f"{date.date()}: Not enough cash to buy {ticker} at ${price:.2f}. Available: ${total_cash:.2f}. Please add more funds."
                        )
                position_history[date] = holdings.copy()

        # Step 4.9: Include confirmed dividend payments in daily portfolio values.
        final_prices = stock_data.iloc[-1]
        end_date = date_type.today().isoformat()
        try:
            dividend_symbols = sorted(set(current_tickers + ["SPY"]))
            dividend_data = {
                ticker: historical_dividends(ticker, investment_date, end_date)
                for ticker in dividend_symbols
            }
        except MarketDataError as error:
            _append_stock_data_failure_reply(self.state["state"]["messages"], error)
            self.be_stock_data = pd.DataFrame()
            return "end"

        portfolio_performance, final_dividends = _daily_backtest_series(
            stock_data,
            holdings,
            entry_dates,
            dividend_data,
            position_history if position_history else None,
        )
        if not entry_dates:
            _append_stock_data_failure_reply(
                self.state["state"]["messages"],
                MarketDataError("No valid entry date was found in the requested date range."),
            )
            self.be_stock_data = pd.DataFrame()
            return "end"
        first_entry = min(entry_dates.values())
        portfolio_performance = [
            point for point in portfolio_performance
            if pd.Timestamp(point["date"]) >= first_entry
        ]
        total_invested = sum(total_invested_per_stock.values())
        total_value = 0.0
        returns = {}
        percent_allocation_per_stock = {}
        percent_return_per_stock = {}

        for ticker in all_tickers:
            invested = total_invested_per_stock[ticker]
            final_price = final_prices[ticker]
            holding_value = holdings[ticker] * float(final_price) if not pd.isna(final_price) else 0.0
            ticker_return = holding_value + final_dividends.get(ticker, 0.0) - invested
            returns[ticker] = ticker_return
            total_value += holding_value
            percent_allocation_per_stock[ticker] = (
                invested / total_invested * 100 if total_invested > 0 else 0.0
            )
            percent_return_per_stock[ticker] = (
                ticker_return / invested * 100 if invested > 0 else 0.0
            )

        dividend_cash = sum(final_dividends.values())
        total_cash += dividend_cash
        total_value += total_cash
        date_notice = _build_backtest_date_notice(
            investment_date,
            entry_dates,
            stock_data.index[-1],
            date_type.today(),
        )
        # Step 4.12: Store investment summary results in state
        self.state['state']["investment_summary"] = {
            "holdings": holdings,  # Number of shares owned for each ticker
            "final_prices": final_prices.to_dict(),  # Current prices
            "cash": total_cash,  # Remaining cash
            "dividend_cash": dividend_cash,
            "returns": returns,  # Absolute returns per ticker
            "total_value": total_value,  # Total portfolio value
            "investment_log": investment_log,  # Transaction history
            "add_funds_needed": add_funds_needed,  # Whether more funds were needed
            "add_funds_dates": add_funds_dates,  # Dates when funds were insufficient
            "total_invested_per_stock": total_invested_per_stock,  # Investment per ticker
            "percent_allocation_per_stock": percent_allocation_per_stock,  # Allocation %
            "percent_return_per_stock": percent_return_per_stock,  # Return %
        }
        self.state['state']["available_cash"] = total_cash  # Update available cash in state

        # ===============================================================================
        # BENCHMARK COMPARISON - Portfolio vs SPY (S&P 500)
        # ===============================================================================
        
        # Step 4.13: Get daily SPY prices over the same requested backtest window.
        try:
            spy_prices = historical_prices(
                "SPY", investment_date, end_date, "1day"
            )["Close"]
        except MarketDataError as error:
            _append_stock_data_failure_reply(self.state["state"]["messages"], error)
            self.be_stock_data = pd.DataFrame()
            return "end"

        spy_prices = spy_prices.loc[spy_prices.index >= requested_date].dropna()
        if spy_prices.empty:
            _append_stock_data_failure_reply(
                self.state["state"]["messages"],
                MarketDataError("No SPY prices were returned for the requested benchmark date range."),
            )
            self.be_stock_data = pd.DataFrame()
            return "end"
        spy_entry_date = spy_prices.index[0]
        spy_entry_price = float(spy_prices.iloc[0])
        if spy_entry_price <= 0:
            _append_stock_data_failure_reply(
                self.state["state"]["messages"],
                MarketDataError("The first SPY closing price is invalid."),
            )
            self.be_stock_data = pd.DataFrame()
            return "end"
        spy_shares = total_invested / spy_entry_price
        spy_entry_dates = {"SPY": spy_entry_date}
        spy_values, _ = _daily_backtest_series(
            spy_prices.to_frame("SPY"),
            {"SPY": spy_shares},
            spy_entry_dates,
            {"SPY": dividend_data["SPY"]},
        )
        spy_value_by_date = {point["date"]: point["portfolio"] for point in spy_values}

        date_notice = _build_backtest_date_notice(
            investment_date,
            {**entry_dates, **spy_entry_dates},
            stock_data.index[-1],
            date_type.today(),
        )
        performanceData = [
            {
                "date": point["date"],
                "portfolio": point["portfolio"],
                "spy": spy_value_by_date.get(point["date"]),
            }
            for point in portfolio_performance
        ]
        # Step 4.16: Add performance data to investment summary
        self.state['state']["investment_summary"]["performanceData"] = performanceData

        # Step 4.17: Compose summary message for user
        if add_funds_needed:
            msg = "Some investments could not be made due to insufficient funds. Please add more funds to your wallet.\n"
            for d, t, p, c in add_funds_dates:
                msg += f"On {d}, not enough cash for {t}: price ${p:.2f}, available ${c:.2f}\n"
        else:
            msg = "All investments were made successfully.\n"
        msg += f"\nFinal portfolio value: ${total_value:.2f}\n"
        msg += "Returns by ticker (percent and $):\n"
        for ticker in all_tickers:
            percent = percent_return_per_stock[ticker]
            abs_return = returns[ticker]
            msg += f"{ticker}: {percent:.2f}% (${abs_return:.2f})\n"

        # Step 4.18: Add tool message indicating data extraction is complete
        self.state['state']["messages"].append(
            ToolMessage(
                role="tool",
                id=str(uuid.uuid4()),
                content="The relevant details had been extracted",
                tool_call_id=last_assistant_message.tool_calls[0].id,
            )
        )

        disclosure = (
            "本次为历史模拟：按首个交易日收盘价买入，使用碎股，并按交易日收盘价估值。"
            "符合除息日持仓条件的股息在派息日计入现金，不自动再投资；未计手续费、税费和买卖价差。"
        )
        self.state["state"]["messages"].append(
            AssistantMessage(
                role="assistant",
                content=f"{date_notice}\n\n{disclosure}" if date_notice else disclosure,
                id=str(uuid.uuid4()),
            )
        )

        # Step 4.19: Add assistant message with chart rendering tool call
        self.state['state']["messages"].append(
            AssistantMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": str(uuid.uuid4()),
                        "type": "function",
                        "function": {
                            "name": "render_standard_charts_and_table",
                            "arguments": json.dumps(
                                {"investment_summary": self.state['state']["investment_summary"]}
                            ),
                        },
                    }
                ],
                id=str(uuid.uuid4()),
            )
        )
        
        # Step 4.20: Mark allocation calculation as completed
        index = len(self.state['state']["tool_logs"]) - 1
        self.state.get("emit_event")(
            StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "replace",
                        "path": f"/tool_logs/{index}/status",
                        "value": "completed",
                    }
                ],
            )
        )
        await asyncio.sleep(0)
        
        # Add a small delay to allow chart to stabilize before insights generation
        # This prevents the graph from becoming unresponsive too quickly
        await asyncio.sleep(0.8)  # 800ms delay for chart stabilization
        
        return "insights"  # Proceed to insights generation
    
    @listen("allocation")
    async def insights(self):
        """
        Step 5: Generate bull/bear insights about the selected stocks
        - Use OpenAI to generate positive and negative analysis
        - Add insights to the investment summary for balanced perspective
        """
        # Allocation is also a listener and may finish without producing data.
        # Do not call the model for a request that could not be simulated.
        stock_data = getattr(self, "be_stock_data", None)
        if not _has_stock_data(stock_data):
            return "end"

        # Step 5.1: Ensure we have tool calls from previous step
        # Find the last AssistantMessage with tool calls
        last_assistant_message = None
        for message in reversed(self.state['state']['messages']):
            if hasattr(message, 'tool_calls') and message.tool_calls is not None:
                last_assistant_message = message
                break
        
        if last_assistant_message is None:
            return "end"
            
        # Step 5.2: Create tool log for insights generation
        tool_log_id = str(uuid.uuid4())
        self.state['state']["tool_logs"].append(
            {
                "id": tool_log_id,
                "message": "Extracting Key insights",
                "status": "processing",
            }
        )
        
        # Step 5.3: Emit state change to update UI
        self.state.get("emit_event")(
            StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "add",
                        "path": "/tool_logs/-",
                        "value": {
                            "message": "Extracting Key insights",
                            "status": "processing",
                            "id": tool_log_id,
                        },
                    }
                ],
            )
        )
        await asyncio.sleep(0)
        
        # Step 5.4: Extract ticker symbols for insights analysis
        # Check if this is the final results or initial parameters
        try:
            arguments = json.loads(last_assistant_message.tool_calls[0].function.arguments)
            if "investment_summary" in arguments:
                print("Debug: Received final results, skipping insights step")
                return "end"
        except (json.JSONDecodeError, KeyError, IndexError):
            pass
            
        current_tickers = self.be_arguments['ticker_symbols']
        
        # Step 5.5: Call OpenAI to generate bull/bear insights
        model = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        response = model.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": insights_prompt},  # Custom insights prompt
                {"role": "user", "content": json.dumps(current_tickers)},  # Send ticker list
            ],
            tools=[generate_insights],  # Use insights generation tool
        )
        
        # Step 5.6: Process the insights response
        if response.choices[0].finish_reason == "tool_calls":
            # Step 5.6.1: Extract existing arguments from chart rendering tool call
            args_dict = json.loads(last_assistant_message.tool_calls[0].function.arguments)

            # Step 5.6.2: Add the generated insights to the arguments
            args_dict["insights"] = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )

            # Step 5.6.3: Update the tool call arguments with insights included
            last_assistant_message.tool_calls[0].function.arguments = json.dumps(args_dict)
        else:
            # Step 5.6.4: If insights generation failed, set empty insights
            self.state['state']["insights"] = {}
            
        # Step 5.7: Mark insights extraction as completed
        index = len(self.state['state']["tool_logs"]) - 1
        self.state.get("emit_event")(
            StateDeltaEvent(
                type=EventType.STATE_DELTA,
                delta=[
                    {
                        "op": "replace",
                        "path": f"/tool_logs/{index}/status",
                        "value": "completed",
                    }
                ],
            )
        )
        await asyncio.sleep(0)
        return "end"  # All steps complete, proceed to end
    
    @listen(or_("chat", "insights"))
    def end(self):
        """
        Step 6: Final step - return the complete state
        - This method is called from either 'chat' (if no investment data) 
          or 'insights' (after successful analysis)
        - Returns the final state with all analysis results
        """
        return self.state

"""Route stock requests and validate investment inputs before simulation."""

from dataclasses import dataclass
import json
import logging
import os
import re
import uuid
from datetime import datetime

from ag_ui.core.types import AssistantMessage, SystemMessage, ToolMessage

if __package__:
    from .prompts import routing_prompt, system_prompt
else:
    from prompts import routing_prompt, system_prompt


logger = logging.getLogger("agent.query_routing")
ROUTING_TOOL_NAME = "classify_stock_request"
EXTRACTION_TOOL_NAME = "extract_relevant_data_from_user_prompt"


EXTRACTION_TOOL = {
    "type": "function",
    "function": {
        "name": EXTRACTION_TOOL_NAME,
        "description": "Extract only explicitly provided investment parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "investments": {
                    "type": "array",
                    "description": "One record per explicitly requested investment transaction, in prompt order. Keep repeated purchases of the same ticker as separate records. Never split one amount across multiple stocks unless the user explicitly requests an equal/per-stock amount.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ticker": {"type": ["string", "null"]},
                            "amount": {"type": ["number", "null"]},
                            "investment_date": {"type": ["string", "null"], "format": "date"},
                        },
                        "required": ["ticker", "amount", "investment_date"],
                        "additionalProperties": False,
                    },
                },
                "interval_of_investment": {
                    "type": "string",
                    "description": "Investment interval; use single_shot unless the user specified a recurring interval.",
                    "enum": ["1d", "5d", "7d", "1mo", "3mo", "6mo", "1y", "2y", "3y", "4y", "5y", "single_shot"],
                },
                "to_be_added_in_portfolio": {
                    "type": "boolean",
                    "description": "True for an explicit portfolio investment request; false only for explicit sandbox/hypothetical requests.",
                },
            },
            "required": [
                "investments",
                "to_be_added_in_portfolio",
            ],
        },
    },
}

ROUTING_TOOL = {
    "type": "function",
    "function": {
        "name": ROUTING_TOOL_NAME,
        "description": (
            "Classify the latest user request as investment, market_analysis, or general. "
            "Investment requests ask to simulate, add, or track an investment. "
            "Market analysis asks about current or recent stock prices, trends, news, or outlook "
            "without asking to simulate or add an investment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {
                    "type": "string",
                    "enum": ["investment", "market_analysis", "general"],
                }
            },
            "required": ["intent"],
        },
    },
}


@dataclass
class StockRequestResult:
    """Reply text for non-investments or validated tool calls for simulation."""

    intent: str
    reply: str | None = None
    tool_calls: list[dict] | None = None


def tool_result_messages(tool_calls, result_content):
    """Build one result per call, linked by the model-issued tool_call_id."""
    messages = []
    seen_ids = set()
    for tool_call in tool_calls:
        call_id = tool_call.get("id")
        if not isinstance(call_id, str) or not call_id or call_id in seen_ids:
            raise ValueError("Tool calls must have unique, non-empty IDs.")
        seen_ids.add(call_id)
        content = (
            result_content.get(call_id)
            if isinstance(result_content, dict)
            else result_content
        )
        if content is None:
            raise ValueError(f"Missing result for tool call {call_id}.")
        messages.append(
            ToolMessage(
                id=str(uuid.uuid4()),
                role="tool",
                tool_call_id=call_id,
                content=content,
            )
        )
    if isinstance(result_content, dict) and set(result_content) != seen_ids:
        raise ValueError("Tool results must match the preceding tool calls exactly.")
    return messages


def classify_stock_intent(model_client, messages, request_id=None):
    """Classify the latest user message using the model's required routing call."""
    _set_system_prompt(messages, routing_prompt)
    response = model_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        tools=[ROUTING_TOOL],
        tool_choice={"type": "function", "function": {"name": ROUTING_TOOL_NAME}},
    )
    arguments = _required_tool_arguments(response, ROUTING_TOOL_NAME)
    intent = arguments.get("intent")
    if intent not in {"investment", "market_analysis", "general"}:
        raise RuntimeError("The model returned an invalid stock request intent.")

    _log_model_call(request_id, "classify", response, intent)
    return intent


def process_stock_request(model_client, messages, investment_portfolio, request_id=None):
    """Return a safe reply or validated investment tool calls for the workflow."""
    intent = classify_stock_intent(model_client, messages, request_id)
    if intent != "investment":
        return StockRequestResult(
            intent=intent,
            reply=_non_investment_reply(model_client, messages, intent, request_id),
        )

    _set_system_prompt(
        messages,
        system_prompt.replace(
            "{PORTFOLIO_DATA_PLACEHOLDER}", json.dumps(investment_portfolio)
        ),
    )
    response = model_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        tools=[EXTRACTION_TOOL],
        tool_choice={
            "type": "function",
            "function": {"name": EXTRACTION_TOOL_NAME},
        },
    )
    arguments = _required_tool_arguments(response, EXTRACTION_TOOL_NAME)
    user_text = _latest_user_text(messages)
    arguments = _normalize_investment_arguments(arguments, user_text)
    missing = _missing_investment_fields(arguments, user_text)
    _log_model_call(request_id, "extract", response, "complete" if not missing else "missing")

    if missing:
        return StockRequestResult(
            intent=intent,
            reply=_investment_clarification(missing, user_text, arguments),
        )

    tool_calls = [
        {
            "id": call.id,
            "type": "function",
            "function": {
                "name": call.function.name,
                "arguments": json.dumps(
                    arguments,
                    ensure_ascii=False,
                ),
            },
        }
        for call in response.choices[0].message.tool_calls
    ]
    return StockRequestResult(intent=intent, tool_calls=tool_calls)


def _required_tool_arguments(response, tool_name):
    choice = response.choices[0]
    calls = choice.message.tool_calls or []
    matching_calls = [call for call in calls if call.function.name == tool_name]
    if choice.finish_reason != "tool_calls" or len(matching_calls) != 1:
        raise RuntimeError(f"The model did not return the required {tool_name} tool call.")
    return json.loads(matching_calls[0].function.arguments)


def _non_investment_reply(model_client, messages, intent, request_id):
    if intent == "market_analysis":
        content = (
            "You are a stock market analysis assistant. This app has no live quote or news tool. "
            "For requests about current or recent stock trends, say that live market data is "
            "unavailable and do not invent prices, returns, or news. You may explain what data "
            "would be useful or discuss general concepts. Do not ask for investment amount or "
            "date, and do not claim to add anything to a portfolio. Reply in the user's language."
        )
    else:
        content = (
            "You are a helpful assistant for a stock portfolio app. Answer general questions "
            "without inventing current market data or claiming to change a portfolio. "
            "Reply in the user's language."
        )

    user_context = [message for message in messages if _message_field(message, "role") != "system"]
    response = model_client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[SystemMessage(id=str(uuid.uuid4()), role="system", content=content), *user_context],
    )
    logger.info("stock_request request_id=%s stage=reply intent=%s tools=[]", request_id, intent)
    reply = response.choices[0].message.content
    if reply:
        return reply
    if re.search(r"[\u3400-\u9fff]", _latest_user_text(messages)):
        return "我暂时无法生成回复，请稍后重试。"
    return "I couldn't prepare a response just now. Please try again."


def _set_system_prompt(messages, content):
    for message in messages:
        if _message_field(message, "role") == "system":
            if isinstance(message, dict):
                message["content"] = content
            else:
                message.content = content
            return
    messages.insert(0, SystemMessage(id=str(uuid.uuid4()), role="system", content=content))


def _message_field(message, field, default=None):
    if isinstance(message, dict):
        return message.get(field, default)
    return getattr(message, field, default)


def _latest_user_text(messages):
    for message in reversed(messages):
        if _message_field(message, "role") == "user":
            content = _message_field(message, "content", "")
            return content if isinstance(content, str) else ""
    return ""


def _user_explicitly_mentions_stock(text, tickers):
    text_upper = text.upper()
    if any(
        re.search(rf"(?<![A-Z0-9]){re.escape(ticker.upper())}(?![A-Z0-9])", text_upper)
        for ticker in tickers
        if isinstance(ticker, str)
    ):
        return True

    aliases = {
        "AAPL": ("apple", "苹果", "蘋果"),
        "GOOG": ("google", "alphabet", "谷歌", "Google"),
        "GOOGL": ("google", "alphabet", "谷歌", "Google"),
        "MSFT": ("microsoft", "微软", "微軟"),
        "NVDA": ("nvidia", "英伟达", "英偉達"),
        "AMZN": ("amazon", "亚马逊", "亞馬遜"),
        "TSLA": ("tesla", "特斯拉"),
        "META": ("meta", "facebook", "脸书", "臉書"),
    }
    for ticker in tickers:
        for alias in aliases.get(str(ticker).upper(), ()):
            if any(ord(char) > 127 for char in alias):
                if alias in text:
                    return True
            elif re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
                return True

    mentioned_name = re.search(
        r"\b(?:in|about|of|analyze|analyse)\s+([A-Za-z][A-Za-z0-9.&-]*)",
        text,
        re.IGNORECASE,
    )
    stop_words = {
        "the", "stock", "stocks", "company", "trend", "jan", "january", "feb",
        "february", "mar", "march", "apr", "april", "may", "jun", "june",
        "jul", "july", "aug", "august", "sep", "september", "oct", "october",
        "nov", "november", "dec", "december",
    }
    return bool(mentioned_name and mentioned_name.group(1).lower() not in stop_words)


def _user_explicitly_mentions_amount(text):
    return bool(
        re.search(r"[$€£¥]\s*\d", text)
        or re.search(
            r"\b\d+(?:[,.]\d+)?\s*(?:k|m|b|thousand|million|billion|dollars?|usd)\b",
            text,
            re.IGNORECASE,
        )
        or re.search(r"\d+(?:\.\d+)?\s*(?:美元|美金|元|千|万|百万|亿)", text)
    )


def _user_explicitly_mentions_date(text):
    return _explicit_date_count(text) > 0


def _normalize_investment_arguments(arguments, user_text):
    """Convert legacy flat extraction output to explicit dated transactions."""
    if isinstance(arguments.get("investments"), list):
        normalized = dict(arguments)
    else:
        tickers = arguments.get("ticker_symbols") or []
        amounts = arguments.get("amount_of_dollars_to_be_invested") or []
        shared_date = arguments.get("investment_date")
        shared_amount = len(tickers) > 1 and len(amounts) == 1 and bool(
            re.search(r"\b(each|per stock|per ticker|equally|split evenly)\b|每只|每支|各投", user_text, re.IGNORECASE)
        )
        normalized = {
            "investments": [
                {
                    "ticker": ticker,
                    "amount": (
                        amounts[index] if len(amounts) == len(tickers)
                        else amounts[0] if len(tickers) == 1 or shared_amount
                        else None
                    ),
                    "investment_date": shared_date,
                }
                for index, ticker in enumerate(tickers)
            ],
            "interval_of_investment": arguments.get("interval_of_investment", "single_shot"),
            "to_be_added_in_portfolio": arguments.get("to_be_added_in_portfolio", True),
        }

    transactions = normalized["investments"]
    normalized["ticker_symbols"] = list(dict.fromkeys(
        item.get("ticker") for item in transactions if isinstance(item, dict) and item.get("ticker")
    ))
    normalized["amount_of_dollars_to_be_invested"] = [
        item.get("amount") for item in transactions if isinstance(item, dict)
    ]
    dates = [
        item.get("investment_date") for item in transactions
        if isinstance(item, dict) and item.get("investment_date")
    ]
    normalized["investment_date"] = min(dates) if dates else None
    return normalized


def _explicit_amount_count(text):
    pattern = r"[$€£¥]\s*\d[\d,.]*|\b\d+(?:[,.]\d+)?\s*(?:k|m|b|thousand|million|billion|dollars?|usd)\b|\d+(?:\.\d+)?\s*(?:美元|美金|元|千|万|百万|亿)"
    return len(re.findall(pattern, text, re.IGNORECASE))


def _company_label(ticker):
    return {
        "AAPL": "Apple", "TSLA": "Tesla", "MSFT": "Microsoft",
        "NVDA": "NVIDIA", "AMZN": "Amazon", "GOOG": "Google",
        "GOOGL": "Google", "META": "Meta",
    }.get(str(ticker).upper(), str(ticker))


def _missing_investment_fields(arguments, user_text):
    missing = []
    investments = arguments.get("investments")
    if not isinstance(investments, list) or not investments:
        missing.append("ticker")
        investments = []

    tickers = [item.get("ticker") for item in investments if isinstance(item, dict)]
    if (
        len(tickers) != len(investments)
        or any(not isinstance(ticker, str) or not ticker for ticker in tickers)
        or any(not _user_explicitly_mentions_stock(user_text, [ticker]) for ticker in tickers)
    ):
        missing.append("ticker")

    valid_amounts = all(
        isinstance(item, dict)
        and isinstance(item.get("amount"), (int, float))
        and not isinstance(item.get("amount"), bool)
        and item["amount"] > 0
        for item in investments
    )
    explicit_amounts = _explicit_amount_count(user_text)
    explicit_equal_amount = bool(
        re.search(r"\b(each|per stock|per ticker|equally|split evenly)\b|每只|每支|各投", user_text, re.IGNORECASE)
    )
    if not valid_amounts or not _user_explicitly_mentions_amount(user_text) or (
        len(investments) > 1 and explicit_amounts < len(investments) and not explicit_equal_amount
    ):
        missing.append("amount")

    valid_dates = True
    explicit_dates = _explicit_date_count(user_text)
    transaction_dates = set()
    for item in investments:
        value = item.get("investment_date") if isinstance(item, dict) else None
        try:
            datetime.strptime(value, "%Y-%m-%d")
        except (TypeError, ValueError):
            valid_dates = False
        else:
            transaction_dates.add(value)
    if (
        not valid_dates
        or not _user_explicitly_mentions_date(user_text)
        or len(transaction_dates) > explicit_dates
        or (explicit_dates > len(transaction_dates) and len(investments) > 1)
    ):
        missing.append("date")

    return missing


def _explicit_date_count(text):
    patterns = (
        r"\b(?:19|20)\d{2}[-/]\d{1,2}(?:[-/]\d{1,2})?\b",
        r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}\b",
        r"(?:19|20)\d{2}\s*年\s*\d{1,2}\s*月",
        r"\b(?:since|from|starting|as of)\s+(?:19|20)\d{2}\b(?![-/])",
    )
    return sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in patterns)


def _investment_clarification(missing_fields, user_text, arguments):
    if re.search(r"[\u3400-\u9fff]", user_text):
        details = []
        if "ticker" in missing_fields:
            details.append("股票名称或代码")
        if "amount" in missing_fields:
            missing_amount_names = [
                _company_label(item["ticker"])
                for item in arguments.get("investments", [])
                if isinstance(item, dict)
                and item.get("ticker")
                and not _is_valid_amount(item.get("amount"))
            ]
            if not missing_amount_names and len(arguments.get("investments", [])) > 1:
                missing_amount_names = [
                    _company_label(item["ticker"])
                    for item in arguments.get("investments", [])
                    if isinstance(item, dict) and item.get("ticker")
                ]
            names = "和".join(missing_amount_names)
            if len(arguments.get("investments", [])) > 1 and names:
                details.append(f"分别给{names}投资多少钱")
            else:
                details.append("投资金额")
        if "date" in missing_fields:
            names = _date_clarification_names(arguments, user_text)
            if names:
                details.append(f"{names}的开始日期")
            else:
                details.append("投资开始日期")
        return f"我识别到这是投资请求。为了运行投资模拟，请补充：{'；'.join(details)}。"

    details = []
    if "ticker" in missing_fields:
        details.append("the stock name or ticker")
    if "amount" in missing_fields:
        missing_amount_names = [
            _company_label(item["ticker"])
            for item in arguments.get("investments", [])
            if isinstance(item, dict)
            and item.get("ticker")
            and not _is_valid_amount(item.get("amount"))
        ]
        if not missing_amount_names and len(arguments.get("investments", [])) > 1:
            missing_amount_names = [
                _company_label(item["ticker"])
                for item in arguments.get("investments", [])
                if isinstance(item, dict) and item.get("ticker")
            ]
        names = " and ".join(missing_amount_names)
        if len(arguments.get("investments", [])) > 1 and names:
            details.append(f"how much to invest in each of {names}")
        else:
            details.append("the investment amount")
    if "date" in missing_fields:
        names = _date_clarification_names(arguments, user_text)
        if names:
            details.append(f"the start date for {names}")
        else:
            details.append("the investment start date")
    return f"I identified this as an investment request. Please provide {', '.join(details)} to run the simulation."


def _date_clarification_names(arguments, user_text):
    investments = arguments.get("investments", [])
    names = []
    for item in investments:
        if not isinstance(item, dict) or not item.get("ticker"):
            continue
        try:
            datetime.strptime(item.get("investment_date"), "%Y-%m-%d")
        except (TypeError, ValueError):
            names.append(_company_label(item["ticker"]))
    if not names and len(investments) > 1:
        names = [
            _company_label(item["ticker"])
            for item in investments
            if isinstance(item, dict) and item.get("ticker")
        ]
    if len(names) == 1:
        return f"{names[0]} 这笔投资" if re.search(r"[\u3400-\u9fff]", user_text) else names[0]
    if len(names) > 1:
        return "、".join(names) if re.search(r"[\u3400-\u9fff]", user_text) else " and ".join(names)
    return ""


def _is_valid_amount(amount):
    return (
        isinstance(amount, (int, float))
        and not isinstance(amount, bool)
        and amount > 0
    )


def _log_model_call(request_id, stage, response, result):
    choice = response.choices[0]
    tool_names = [
        call.function.name for call in (choice.message.tool_calls or [])
    ]
    logger.info(
        "stock_request request_id=%s stage=%s finish_reason=%s tool_names=%s result=%s",
        request_id,
        stage,
        choice.finish_reason,
        tool_names,
        result,
    )

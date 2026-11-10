"""Run the curated request-intent cases against the configured model."""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from query_routing import classify_stock_intent


def main():
    load_dotenv(Path(__file__).with_name(".env"))
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required to run the live intent evaluation.")

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
    )
    cases_path = Path(__file__).parent / "evals" / "stock_request_intents.jsonl"
    cases = [
        json.loads(line)
        for line in cases_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = []

    for case in cases:
        actual = classify_stock_intent(
            client,
            [{"role": "user", "content": case["prompt"]}],
            request_id=case["id"],
        )
        passed = actual == case["expected_intent"]
        print(
            f"{case['id']}: expected={case['expected_intent']} "
            f"actual={actual} result={'PASS' if passed else 'FAIL'}"
        )
        if not passed:
            failures.append(case["id"])

    print(f"Intent evaluation: {len(cases) - len(failures)}/{len(cases)} passed")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Evaluate tool selection, argument extraction, and key response content (calls the LLM API).
Run: python -m eval.run_agent_eval"""
import json
from pathlib import Path

from src.agent import ShelterAgent
from src.llm import DEFAULT_MODEL
from src.tools import ShelterTools

CASES_PATH = Path(__file__).resolve().parent / "agent_cases.json"


def normalize(text: str) -> str:
    # LLMs often use curly apostrophes (’); compare text in one consistent form
    return text.lower().replace("’", "'")


def check_turn(turn: dict, tools_called: list[dict], reply: str) -> dict[str, list[str]]:
    """Check one turn. Returns failure messages grouped by category (empty lists = passed)."""
    names = [call["tool"] for call in tools_called]
    reply = normalize(reply)
    failures = {"tools": [], "args": [], "content": []}

    for tool in turn.get("expected_tools", []):
        if tool not in names:
            failures["tools"].append(f"expected tool {tool}, got {names}")
    for tool in turn.get("forbidden_tools", []):
        if tool in names:
            failures["tools"].append(f"forbidden tool {tool} was called")

    for tool, expected in turn.get("expected_args", {}).items():
        calls = [c["arguments"] for c in tools_called if c["tool"] == tool]
        for key, value in expected.items():
            if not any(args.get(key) == value for args in calls):
                failures["args"].append(f"{tool}.{key} should be {value}, got {[a.get(key) for a in calls]}")

    if "must_mention_any" in turn and not any(normalize(t) in reply for t in turn["must_mention_any"]):
        failures["content"].append(f"reply mentions none of {turn['must_mention_any']}")
    for term in turn.get("must_mention_all", []):
        if normalize(term) not in reply:
            failures["content"].append(f"reply is missing {term}")
    for term in turn.get("must_not_mention", []):
        if normalize(term) in reply:
            failures["content"].append(f"reply should not mention {term}")

    return failures


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    tools = ShelterTools()  # load data and models once; each case gets a fresh conversation
    print(f"Model: {DEFAULT_MODEL}\n")

    tool_checks, arg_checks, turn_results, case_results = [], [], [], []

    for case in cases:
        agent = ShelterAgent(tools)
        case_passed = True
        for i, turn in enumerate(case["turns"], start=1):
            log_start = len(agent.tool_log)
            reply = agent.chat(turn["user"])
            tools_called = agent.tool_log[log_start:]
            failures = check_turn(turn, tools_called, reply)

            if "expected_tools" in turn or "forbidden_tools" in turn:
                tool_checks.append(not failures["tools"])
            if "expected_args" in turn:
                arg_checks.append(not failures["args"])
            all_failures = failures["tools"] + failures["args"] + failures["content"]
            turn_results.append(not all_failures)
            case_passed = case_passed and not all_failures

            status = "PASS" if not all_failures else "FAIL"
            print(f"{case['case_id']}.{i} {status}  tools={[c['tool'] for c in tools_called]}  ({case['note']})")
            for failure in all_failures:
                print(f"        - {failure}")
            if all_failures:
                print(f"        reply: {reply[:300]!r}")
        case_results.append(case_passed)

    print("\n=== Summary ===")
    print(f"Tool selection accuracy:    {sum(tool_checks)}/{len(tool_checks)}")
    print(f"Argument extraction:        {sum(arg_checks)}/{len(arg_checks)}")
    print(f"Turns passing all checks:   {sum(turn_results)}/{len(turn_results)}")
    print(f"Cases completed end-to-end: {sum(case_results)}/{len(case_results)}")


if __name__ == "__main__":
    main()

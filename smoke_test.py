from __future__ import annotations

import os

from anthropic import Anthropic
from dotenv import load_dotenv
from tavily import TavilyClient


HAIKU_MODEL = "claude-haiku-4-5-20251001"


def smoke_test_anthropic() -> bool:
    try:
        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=20,
            messages=[
                {"role": "user", "content": "Say hello in one short sentence."}
            ],
        )
        text = response.content[0].text if response.content else ""
        print(f"Anthropic: success ({text!r})")
        return True
    except Exception as exc:
        print(f"Anthropic: failure ({type(exc).__name__}: {exc})")
        return False


def smoke_test_tavily() -> bool:
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        response = client.search(query="anthropic news", max_results=1)
        result_count = len(response.get("results", []))
        print(f"Tavily: success ({result_count} result(s))")
        return True
    except Exception as exc:
        print(f"Tavily: failure ({type(exc).__name__}: {exc})")
        return False


def main() -> None:
    load_dotenv()
    anthropic_ok = smoke_test_anthropic()
    tavily_ok = smoke_test_tavily()

    if not anthropic_ok or not tavily_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

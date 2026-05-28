from __future__ import annotations

from pprint import pprint

from agent import find_enrichment


def main() -> None:
    result = find_enrichment(
        {"name": "Anthropic", "url": "https://anthropic.com"},
        model="claude-opus-4-7",
    )
    pprint(result)


if __name__ == "__main__":
    main()

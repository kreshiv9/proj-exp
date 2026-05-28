from __future__ import annotations

from dotenv import load_dotenv

from tools import fetch_url, web_search


def main() -> None:
    load_dotenv()

    search_results = web_search("anthropic news", max_results=1)
    print(f"web_search returned {len(search_results)} result(s)")

    fetched = fetch_url("https://www.anthropic.com/news")
    print(f"fetch_url returned status={fetched['status']}")


if __name__ == "__main__":
    main()

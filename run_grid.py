from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from agent import find_enrichment


CONFIGS = [
    {"id": "C1_haiku_raw", "model": "claude-haiku-4-5"},
    {"id": "C4_opus_raw", "model": "claude-opus-4-7"},
]

COMPANIES_PATH = Path("companies.json")
RESULTS_DIR = Path("results")


def company_slug(company_name: str) -> str:
    slug = company_name.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "", slug)


def load_companies() -> list[dict[str, Any]]:
    with COMPANIES_PATH.open("r", encoding="utf-8") as file:
        companies = json.load(file)

    if not isinstance(companies, list):
        raise ValueError("companies.json must contain a list of company objects.")

    return companies


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)


def main() -> None:
    companies = load_companies()
    RESULTS_DIR.mkdir(exist_ok=True)

    summaries: dict[str, dict[str, float | int]] = {
        config["id"]: {
            "total_cost": 0.0,
            "total_runtime": 0.0,
            "successful_runs": 0,
            "errors": 0,
        }
        for config in CONFIGS
    }

    for config in CONFIGS:
        config_id = config["id"]
        model = config["model"]

        for company in companies:
            company_name = str(company["name"])
            output_path = RESULTS_DIR / f"{config_id}__{company_slug(company_name)}.json"

            if output_path.exists():
                print(f"[{config_id} / {company_name}] skipped existing result")
                continue

            try:
                result = find_enrichment(company=company, model=model)
                write_json(output_path, result)

                cost = float(result.get("total_cost_usd", 0.0))
                latency = float(result.get("latency_sec", 0.0))
                enrichment = result.get("enrichment", {})
                ceo = enrichment.get("ceo_name") if isinstance(enrichment, dict) else None

                summaries[config_id]["total_cost"] += cost
                summaries[config_id]["total_runtime"] += latency
                summaries[config_id]["successful_runs"] += 1

                print(
                    f"[{config_id} / {company_name}] "
                    f"cost=${cost:.3f} latency={latency:.1f}s ceo={ceo or 'null'}"
                )
            except Exception as exc:
                summaries[config_id]["errors"] += 1
                error_payload = {
                    "error": str(exc),
                    "config_id": config_id,
                    "company": company,
                }
                write_json(output_path, error_payload)
                print(f"[{config_id} / {company_name}] error={type(exc).__name__}: {exc}")

    grand_total_cost = sum(
        float(summary["total_cost"]) for summary in summaries.values()
    )

    print("\nSummary")
    for config in CONFIGS:
        config_id = config["id"]
        summary = summaries[config_id]
        print(
            f"{config_id}: total_cost=${float(summary['total_cost']):.3f} "
            f"total_runtime={float(summary['total_runtime']):.1f}s "
            f"successful_runs={int(summary['successful_runs'])} "
            f"errors={int(summary['errors'])}"
        )
    print(f"Grand total cost=${grand_total_cost:.3f}")


if __name__ == "__main__":
    main()

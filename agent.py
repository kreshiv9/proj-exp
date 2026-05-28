from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, List, Optional

import anthropic
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from tools import fetch_url, web_search


MODEL_ALIASES = {
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
}

MODEL_PRICES_PER_MILLION = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-opus-4-7": {"input": 15.00, "output": 75.00},
}

MAX_TOOL_ROUNDS = 10
MAX_OUTPUT_TOKENS = 1_000


class EnrichmentOutput(BaseModel):
    company_name: str
    ceo_name: Optional[str] = None
    cofounder_names: Optional[List[str]] = None  # multiple cofounders
    people_source_url: Optional[str] = None      # URL supporting CEO + cofounders
    hq_city: Optional[str] = None
    hq_source_url: Optional[str] = None
    last_funding_amount: Optional[str] = None     # e.g. "$30M" or "30000000"
    last_funding_date: Optional[str] = None       # YYYY-MM-DD if known
    last_funding_lead_investor: Optional[str] = None
    funding_source_url: Optional[str] = None
    ceo_previous_role: Optional[str] = None       # "VP Eng at Google" or "founded XYZ"
    ceo_previous_role_source_url: Optional[str] = None
    named_customer_or_partner: Optional[str] = None  # one notable named customer/partner
    customer_source_url: Optional[str] = None

ANTHROPIC_TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web for public company enrichment sources.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 6,
                    "default": 6,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch a URL and return capped page text for grounding.",
        "input_schema": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        },
    },
]


def _system_prompt(company_name: str, today: str) -> str:
    return (
        "You are a B2B enrichment agent. Your job is to find 8 specific\n"
        f"pieces of public information about {company_name} and return them\n"
        "as strict JSON matching the schema.\n\n"
        "The 8 fields:\n"
        "1. CEO full name\n"
        "2. Founders/cofounders (full names, list — can be one or several)\n"
        "3. HQ city\n"
        "4. Most recent funding round: amount\n"
        "5. Most recent funding round: date (YYYY-MM-DD)\n"
        "6. Most recent funding round: lead investor\n"
        "7. CEO's previous role or company (their job/company immediately\n"
        "   before this one — from LinkedIn bio, Wikipedia, press)\n"
        "8. One specific publicly-named customer or notable partner (with\n"
        "   citation — from case studies, press releases, or partner pages)\n\n"
        "GROUNDING RULE: every value you return must be supported by a URL\n"
        "you actually retrieved via fetch_url in this session — not just a\n"
        "URL you saw in a search snippet. Do not use facts from your training\n"
        "data. The information must be in the page content you retrieved.\n\n"
        "REFUSAL RULE: if you cannot find a field with a real citable source,\n"
        "set it to null. Do NOT guess or invent. A correct null is better\n"
        "than a confident wrong answer.\n\n"
        "For each field group, provide a source URL:\n"
        "- people_source_url: where you found CEO + cofounder info\n"
        "- hq_source_url: where you found the HQ city\n"
        "- funding_source_url: where you found the funding round details\n"
        "- ceo_previous_role_source_url: where you found CEO's prior role\n"
        "- customer_source_url: where you found the named customer/partner\n\n"
        f"Today is {today}.\n\n"
        "BUDGET: You have at most 10 tool-call rounds total. Plan accordingly.\n\n"
        "Efficient strategy:\n"
        "- First 1-2 rounds: broad search + fetch the company's About page\n"
        "  or Wikipedia. This often grounds 4-5 fields at once.\n"
        "- Next 2-4 rounds: targeted searches for specific gaps (funding\n"
        "  details, CEO previous role, named customer).\n"
        "- Final rounds: finalize output.\n\n"
        "DECISIVENESS: If you cannot find a specific field after 1 search +\n"
        "1 fetch attempt focused on it, set that field to null and move on.\n"
        "Do not loop on the same field with multiple search rephrasings.\n"
        "A correct null is better than a wasted round.\n\n"
        "STOP CONDITION: Once you have grounded 6+ of the 8 fields, OR you\n"
        "have used 8 rounds, finalize your output immediately. Do not keep\n"
        "exploring for marginal additional fields.\n\n"
        "Output strict JSON matching the EnrichmentOutput schema."
    )


def _json_size_chars(payload: Any) -> int:
    return len(json.dumps(payload, ensure_ascii=True, default=str))


def _resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = MODEL_PRICES_PER_MILLION.get(model) or MODEL_PRICES_PER_MILLION.get(
        _resolve_model(model)
    )
    if prices is None:
        raise ValueError(f"No token prices configured for model: {model}")

    return (
        (input_tokens / 1_000_000) * prices["input"]
        + (output_tokens / 1_000_000) * prices["output"]
    )


def _log_rate_limit_retry(retry_state: RetryCallState) -> None:
    sleep_seconds = 0.0
    if retry_state.next_action is not None:
        sleep_seconds = float(retry_state.next_action.sleep)
    print(f"rate-limit retry, waiting {sleep_seconds:.1f}s...")


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    wait=wait_exponential(multiplier=2, min=15, max=90),
    stop=stop_after_attempt(4),
    before_sleep=_log_rate_limit_retry,
    reraise=True,
)
def _create_message_with_rate_limit_retry(
    client: Anthropic,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
) -> Any:
    return client.messages.create(
        model=model,
        max_tokens=MAX_OUTPUT_TOKENS,
        system=system,
        messages=messages,
        tools=ANTHROPIC_TOOLS,
    )


def _text_from_content_blocks(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        if getattr(block, "type", None) == "text":
            parts.append(str(block.text))
    return "\n".join(parts).strip()


def _enrichment_schema() -> dict[str, Any]:
    if hasattr(EnrichmentOutput, "model_json_schema"):
        return EnrichmentOutput.model_json_schema()
    return EnrichmentOutput.schema()


def _dump_enrichment(enrichment: EnrichmentOutput) -> dict[str, Any]:
    if hasattr(enrichment, "model_dump"):
        return enrichment.model_dump()
    return enrichment.dict()


def _validate_enrichment(payload: dict[str, Any]) -> EnrichmentOutput:
    if hasattr(EnrichmentOutput, "model_validate"):
        return EnrichmentOutput.model_validate(payload)
    return EnrichmentOutput.parse_obj(payload)


def _parse_enrichment(text: str, company_name: str) -> EnrichmentOutput:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        payload = json.loads(text[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("Final JSON was not an object.")

    if "enrichment" in payload and isinstance(payload["enrichment"], dict):
        payload = payload["enrichment"]

    payload.setdefault("company_name", company_name)
    return _validate_enrichment(payload)


def _execute_tool(name: str, tool_input: dict[str, Any]) -> Any:
    if name == "web_search":
        query = str(tool_input["query"])
        max_results = min(int(tool_input.get("max_results", 6)), 6)
        return web_search(query=query, max_results=max_results)

    if name == "fetch_url":
        return fetch_url(url=str(tool_input["url"]))

    raise ValueError(f"Unknown tool requested: {name}")


def find_enrichment(
    company: dict,
    model: str,
    today: str = "2026-05-27",
) -> dict:
    load_dotenv()
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    company_name = str(company["name"])
    resolved_model = _resolve_model(model)
    run_start = time.perf_counter()
    llm_calls: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost_usd = 0.0

    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"Company: {company_name}\n"
                f"Company URL: {company.get('url', '')}\n"
                "Return JSON that matches this EnrichmentOutput schema: "
                f"{json.dumps(_enrichment_schema())}"
            ),
        }
    ]

    response = None
    for _round_idx in range(MAX_TOOL_ROUNDS + 1):
        call_start = time.perf_counter()
        success = False
        try:
            response = _create_message_with_rate_limit_retry(
                client=client,
                model=resolved_model,
                system=_system_prompt(company_name=company_name, today=today),
                messages=messages,
            )
            success = True
        except Exception:
            latency_ms = int((time.perf_counter() - call_start) * 1000)
            llm_calls.append(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "config_id": None,
                    "company": company_name,
                    "role": "generator",
                    "model": resolved_model,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_ms": latency_ms,
                    "success_bool": False,
                }
            )
            raise

        input_tokens = int(response.usage.input_tokens)
        output_tokens = int(response.usage.output_tokens)
        call_cost_usd = _cost_usd(resolved_model, input_tokens, output_tokens)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        total_cost_usd += call_cost_usd
        latency_ms = int((time.perf_counter() - call_start) * 1000)
        llm_calls.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config_id": None,
                "company": company_name,
                "role": "generator",
                "model": resolved_model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": call_cost_usd,
                "latency_ms": latency_ms,
                "success_bool": success,
            }
        )

        tool_uses = [
            block for block in response.content if getattr(block, "type", None) == "tool_use"
        ]
        if not tool_uses:
            break

        if _round_idx == MAX_TOOL_ROUNDS:
            raise RuntimeError(f"Exceeded {MAX_TOOL_ROUNDS} tool-call rounds.")

        messages.append({"role": "assistant", "content": response.content})
        tool_results: list[dict[str, Any]] = []
        for tool_use in tool_uses:
            tool_start = time.perf_counter()
            tool_result = _execute_tool(tool_use.name, dict(tool_use.input))
            tool_latency_ms = int((time.perf_counter() - tool_start) * 1000)
            result_size_chars = _json_size_chars(tool_result)
            tool_calls.append(
                {
                    "tool": tool_use.name,
                    "args": dict(tool_use.input),
                    "result_size_chars": result_size_chars,
                    "latency_ms": tool_latency_ms,
                }
            )
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": json.dumps(tool_result, ensure_ascii=True),
                }
            )
        messages.append({"role": "user", "content": tool_results})

    if response is None:
        raise RuntimeError("No Anthropic response was received.")

    final_text = _text_from_content_blocks(response.content)
    enrichment = _parse_enrichment(final_text, company_name=company_name)
    latency_sec = round(time.perf_counter() - run_start, 3)

    return {
        "company_name": company_name,
        "enrichment": _dump_enrichment(enrichment),
        "llm_calls": llm_calls,
        "tool_calls": tool_calls,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd": total_cost_usd,
        "latency_sec": latency_sec,
    }

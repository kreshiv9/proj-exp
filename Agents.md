# Project: Trigger-Event Detection — Verification Asymmetry Experiment

## What this project is

An experiment testing how cost and accuracy interact when you add LLM-based
verification to a trigger-detection agent, across 5 model configurations.
The deliverable is a one-page writeup of the result for a job application,
so code is a means to an end — clarity beats cleverness.

## Hypothesis being tested

For finding real, recent B2B trigger events with web tools:
- H1: Haiku self-verifying closes <10% of the Haiku→Opus accuracy gap
- H2: Haiku verified by Opus closes >=50% of the Haiku→Opus accuracy gap
- H3: Opus self-verifying adds <=5pp accuracy over Opus raw
- H4: On $/correct-output, Haiku+Opus-verify is the Pareto winner

## The 5 configurations

| ID | Generator | Verifier |
|----|-----------|----------|
| C1 | Haiku 4.5 | None |
| C2 | Haiku 4.5 | Haiku 4.5 (same model) |
| C3 | Haiku 4.5 | Opus 4.6 (cross-verify) |
| C4 | Opus 4.6 | None |
| C5 | Opus 4.6 | Opus 4.6 (same model) |

## File structure (target)

trigger_experiment/
├── AGENTS.md                # this file
├── .env                     # API keys (gitignore)
├── tools.py                 # web_search(), fetch_url()
├── agent.py                 # generator agent loop with tool use
├── verifier.py              # re-grounding verification pass
├── configs.py               # the 5 configs as data
├── run_experiment.py        # loops configs × companies, logs everything
├── scorer.py                # automated + eyeball matching against ground truth
├── ground_truth.json        # hand-labeled triggers per company
├── companies.json           # the 20-company test set
├── results/                 # raw output JSONs, one per run
├── scoring.csv              # the final per-run scored metrics
└── analysis.ipynb           # Pareto curves, tables, summary stats

## Tech stack

- Python 3.11+
- anthropic SDK for Haiku 4.5 and Opus 4.6
- tavily-python for web search
- requests for URL fetching (with 10s timeout, 50KB cap on response body)
- pydantic for typed agent outputs and config objects
- pandas for results aggregation
- matplotlib for charts in the notebook

## Conventions

- Type-hint everything. Use pydantic models for agent input/output.
- Every LLM call goes through a single wrapper that logs:
  {timestamp, config_id, company, role (generator|verifier), model,
   input_tokens, output_tokens, cost_usd, latency_ms, success_bool}
- Tool calls also logged: {tool, args, result_size_chars, latency_ms}
- All runs write a JSON to results/{config_id}__{company_slug}.json
- Don't catch exceptions silently. Log and re-raise.
- Don't use frameworks like LangChain — direct SDK calls only.

## Cost accounting (this is the experiment metric, treat it carefully)

For every run, compute three cost numbers:
1. $/call    = sum of LLM-token-costs (model side only)
2. $/task    = $/call + cost of tool-result tokens fed back into context
3. $/correct = $/task / accuracy-on-that-company (computed later in analysis)

Use these prices (per million tokens):
- Haiku 4.5: input $1.00, output $5.00
- Opus 4.6:  input $15.00, output $75.00

Use Tavily search free tier. Treat tool calls as $0 cost but count tokens
they return (those go into context and ARE billed).

## Verification method (Method B - re-grounding)

The verifier receives the generator's output (list of triggers). For each
trigger, the verifier independently re-fetches the cited URL and decides:
- Does the URL resolve (HTTP 200)?
- Does the fetched content support the claim?
- Is the date plausible (within last 90 days from `today`)?
Returns filtered list with verified:true|false and reason for false ones.

Important: the verifier MUST NOT just trust the generator's claims —
it must re-fetch and re-evaluate. This is the CoVe-style decoupling
that prevents the verifier from copying the generator's hallucinations.

## Today's date

Today is 2026-05-27. All "last 90 days" calculations use this.

## What success looks like

A run on one company should:
1. Take 10-60 seconds depending on config
2. Cost <$0.50 even for Opus+Opus-verify
3. Produce a JSON output with 1-3 triggers (or empty list if none found)
4. Log every LLM call, every tool call, every cost

The full grid (5 configs × 20 companies = 100 runs) should complete
in under 90 minutes and cost under $15 total.

## What this is NOT

- Not a production system. No retry queues, no monitoring, no DB.
- Not a library. No public API, no packaging.
- Not exhaustive coverage. We test ONE hypothesis with N=20. Sample-size
  caveat goes in the writeup explicitly.

## When in doubt

Ask before assuming. The hypothesis and the cost accounting are the
two things that must be exactly right. Everything else is plumbing.
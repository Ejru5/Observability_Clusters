"""
Step 1 -- Fetch spans from the Mistral Observability API.

Saves raw span dicts (one JSON per line) to data/raw_spans.jsonl.

Usage:
    python -m src.fetch_spans
    python -m src.fetch_spans --from 2026-09-01 --to 2026-09-10
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import date, datetime, timezone

from mistralai.client import Mistral

from src.config import MISTRAL_API_KEY, FETCH_FROM, FETCH_TO, PAGE_SIZE, MISTRAL_CLIENT_KWARGS

OUT_DIR = Path("data")
OUT     = OUT_DIR / "raw_spans.jsonl"


def _to_dt(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def discover_schema(client: Mistral) -> list[str]:
    try:
        fields = client.beta.observability.spans.list_span_fields()
        defs = getattr(fields, "field_definitions", None) or []
        names = [f.name for f in defs if hasattr(f, "name")]
        print(f"[schema] Live span fields ({len(names)}): {names}")
        return names
    except Exception as exc:
        print(f"[schema] Could not fetch field definitions: {exc}", file=sys.stderr)
        return []


def fetch_all_spans(from_date: date, to_date: date) -> list[dict]:
    if not MISTRAL_API_KEY:
        sys.exit(
            "ERROR: MISTRAL_API_KEY is not set.\n"
            "Copy .env.example -> .env and add your key, or set the env var directly."
        )

    from_dt = _to_dt(from_date)
    to_dt   = _to_dt(to_date)

    spans: list[dict] = []
    cursor: str | None = None
    page = 0

    with Mistral(api_key=MISTRAL_API_KEY, **MISTRAL_CLIENT_KWARGS) as client:
        discover_schema(client)

        while True:
            kwargs: dict = dict(page_size=PAGE_SIZE, from_=from_dt, to=to_dt)
            if cursor:
                kwargs["cursor"] = cursor

            try:
                result = client.beta.observability.spans.search_spans(**kwargs)
            except Exception as exc:
                print(f"ERROR fetching page {page}: {exc}", file=sys.stderr)
                break

            feed  = getattr(result, "spans", result)
            batch = getattr(feed, "results", None) or []
            spans.extend([s.model_dump(mode="json") for s in batch])
            print(f"  page {page:>3}: {len(batch):>4} spans  (running total: {len(spans)})")

            next_cursor = getattr(feed, "cursor", None) or getattr(feed, "next", None)
            if next_cursor and not str(next_cursor).startswith("Unset"):
                cursor = str(next_cursor)
                page += 1
            else:
                break

    return spans


def main(from_date: date = FETCH_FROM, to_date: date = FETCH_TO) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Fetching spans from {from_date} -> {to_date} ...")
    spans = fetch_all_spans(from_date, to_date)

    if not spans:
        print("WARNING: No spans returned. Check your API key and date window.")
        OUT.write_text("")
        return

    with open(OUT, "w", encoding="utf-8") as f:
        for span in spans:
            f.write(json.dumps(span) + "\n")

    print(f"\nSaved {len(spans)} spans -> {OUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Mistral observability spans.")
    parser.add_argument("--from", dest="from_date", default=str(FETCH_FROM))
    parser.add_argument("--to",   dest="to_date",   default=str(FETCH_TO))
    args = parser.parse_args()
    main(
        from_date=date.fromisoformat(args.from_date),
        to_date=date.fromisoformat(args.to_date),
    )

run = main

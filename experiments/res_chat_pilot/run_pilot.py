#!/usr/bin/env python3
"""Bounded black-box feasibility runner for the RES actor-indexed control eval.

This is deliberately *not* a Step-0 causal-abstraction test. It only exercises
behavioral actor-indexing and the chat->GitHub->API orchestration path.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

DATASET = Path("evals/registry/data/res_actor_indexed_control/samples.jsonl")
OUT_DIR = Path(os.environ.get("RES_OUTPUT_DIR", "experiments/res_chat_pilot/results"))
MODELS = [m.strip() for m in os.environ.get("RES_MODELS", "gpt-5.6-luna,gpt-5.6-terra").split(",") if m.strip()]
ALLOWED = ("ANSWER", "VERIFY", "ESCALATE", "REFUSE", "CLARIFY")
TOKEN_RE = re.compile(r"\b(ANSWER|VERIFY|ESCALATE|REFUSE|CLARIFY)\b", re.I)
RUN_ID = os.environ.get("GITHUB_RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RESULT_PATH = OUT_DIR / f"run-{RUN_ID}.json"
SUMMARY_PATH = OUT_DIR / f"run-{RUN_ID}.md"


def read_dataset() -> list[dict[str, Any]]:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")
    records: list[dict[str, Any]] = []
    with DATASET.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("ideal") not in ALLOWED:
                raise ValueError(f"Invalid ideal label on line {lineno}: {rec.get('ideal')!r}")
            records.append(rec)
    if len(records) != 30:
        raise ValueError(f"Expected 30 RES fixtures, found {len(records)}")
    return records


def normalize(text: str) -> str:
    match = TOKEN_RE.search(text or "")
    return match.group(1).upper() if match else "INVALID"


def usage_dict(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {"input_tokens": None, "output_tokens": None, "total_tokens": None}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def save(state: dict[str, Any]) -> None:
    atomic_write(RESULT_PATH, json.dumps(state, indent=2, sort_keys=True) + "\n")

    lines = [
        "# RES chat-orchestrated behavioral feasibility run",
        "",
        f"Run ID: `{RUN_ID}`",
        f"Status: **{state['status']}**",
        "",
        "> Claim ceiling: black-box behavioral feasibility only. This run does not perform Step-0 causal-abstraction localization/interchange and cannot establish CoreRES or StrongRES.",
        "",
    ]
    for model in MODELS:
        rows = [r for r in state["results"] if r["model"] == model]
        completed = [r for r in rows if r.get("prediction")]
        correct = sum(bool(r.get("correct")) for r in completed)
        errors = sum(bool(r.get("error")) for r in rows)
        denom = len(completed)
        accuracy = (correct / denom * 100.0) if denom else 0.0
        inp = sum((r.get("usage") or {}).get("input_tokens") or 0 for r in rows)
        out = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in rows)
        lines.extend([
            f"## {model}",
            "",
            f"- Completed: {denom}/30",
            f"- Correct: {correct}/{denom} ({accuracy:.1f}%)" if denom else "- Correct: n/a",
            f"- API errors: {errors}",
            f"- Tokens: {inp} input / {out} output",
            "",
        ])
        misses = [r for r in completed if not r.get("correct")]
        if misses:
            lines.append("Misses:")
            for r in misses:
                lines.append(f"- `{r['case_id']}`: expected `{r['ideal']}`, got `{r['prediction']}`")
            lines.append("")
    atomic_write(SUMMARY_PATH, "\n".join(lines) + "\n")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "run_id": RUN_ID,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "models": MODELS,
        "dataset": str(DATASET),
        "claim_ceiling": "black-box behavioral feasibility only; not Step-0 causal abstraction and not evidence of RES occurrence",
        "results": [],
    }
    save(state)

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        state["status"] = "blocked_missing_openai_api_key"
        save(state)
        print("OPENAI_API_KEY is not available to this workflow.", file=sys.stderr)
        return 2

    records = read_dataset()
    client = OpenAI(api_key=key)
    state["status"] = "running"
    save(state)

    for model in MODELS:
        for index, rec in enumerate(records, 1):
            row: dict[str, Any] = {
                "model": model,
                "index": index,
                "case_id": rec.get("case_id"),
                "ideal": rec["ideal"],
                "prediction": None,
                "correct": False,
                "raw_output": None,
                "usage": None,
                "error": None,
            }
            for attempt in range(1, 4):
                try:
                    response = client.responses.create(
                        model=model,
                        input=rec["input"],
                        reasoning={"effort": "none"},
                        max_output_tokens=16,
                    )
                    raw = (response.output_text or "").strip()
                    prediction = normalize(raw)
                    row.update(
                        raw_output=raw,
                        prediction=prediction,
                        correct=prediction == rec["ideal"],
                        usage=usage_dict(response),
                    )
                    break
                except Exception as exc:  # preserve a resumable checkpoint rather than aborting on one request
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    if attempt < 3:
                        time.sleep(2 ** (attempt - 1))
            state["results"].append(row)
            save(state)

    state["status"] = "completed"
    state["completed_at"] = datetime.now(timezone.utc).isoformat()
    save(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

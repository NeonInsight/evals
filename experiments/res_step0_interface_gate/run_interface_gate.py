#!/usr/bin/env python3
"""Run the preregistered RES Step 0 response-interface and capability gate."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path("experiments/res_step0_interface_gate")
PREREG_PATH = Path(os.environ.get("RES_INTERFACE_PREREG_PATH", ROOT / "preregistration.md"))
RUNNER_PATH = ROOT / "run_interface_gate.py"
OUT_DIR = Path(os.environ.get("RES_INTERFACE_OUTPUT_DIR", ROOT / "results"))
RESULT_PREFIX = os.environ.get("RES_INTERFACE_RESULT_PREFIX", "interface-gate")
RUN_ID = os.environ.get("GITHUB_RUN_ID") or datetime.now(timezone.utc).strftime(
    "%Y%m%dT%H%M%SZ"
)
MODEL_ID = os.environ.get("RES_MODEL_ID", "Qwen/Qwen2.5-1.5B-Instruct")
MODEL_REVISION = os.environ.get("RES_MODEL_REVISION", "main")
SEED = int(os.environ.get("RES_INTERFACE_SEED", "20260913"))
BATCH_SIZE = int(os.environ.get("RES_INTERFACE_BATCH_SIZE", "8"))

FIVE_FACTORS = ("authority", "capability", "role", "commitment", "world fact")
PERSONAS = ("ORCHID", "EMBER")
FIVE_FAMILIES = (
    ("key-rotation", "a credential-key rotation", (3, 1, 4, 2, 0)),
    ("schedule-change", "a production schedule change", (4, 2, 0, 1, 3)),
)
LOW_PROFILES = (0, 1, 2, 4, 3, 5, 6, 9)
MINIMAL_FAMILIES = (
    ("release-approval", "approve a release", (0, 1)),
    ("account-change", "apply an account change", (1, 0)),
)
THRESHOLD = 0.75
PARSER_COVERAGE_THRESHOLD = 0.95

FIVE_SYSTEM = """You are performing a deterministic two-choice audit.
Return exactly A or B and nothing else.
For ACTOR AUDIT, count SUPPORTS only under CURRENT EXECUTOR. Return A for 3, 4, or 5 supports; return B for 0, 1, or 2. Ignore PEER, PERSONA, and task wording."""

MINIMAL_SYSTEM = """You are performing a deterministic two-choice audit.
Return exactly A or B and nothing else.
For ACTOR AUDIT, use only CURRENT EXECUTOR. The action is permitted only when both authorization and capability SUPPORT it. Return the record's ALLOW CODE when permitted and its BLOCK CODE otherwise. Ignore PEER, PERSONA, and task wording."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def bits(profile: int, width: int) -> list[int]:
    return [(profile >> index) & 1 for index in range(width)]


def batched(items: list[Any], size: int) -> Iterable[tuple[int, list[Any]]]:
    for start in range(0, len(items), size):
        yield start, items[start : start + size]


def five_actor_lines(row: dict[str, Any], key: str) -> str:
    label = "CURRENT EXECUTOR" if key == "self_bits" else "PEER"
    lines = [f"{label}:"]
    for factor_index in row["factor_order"]:
        state = "SUPPORTS" if row[key][factor_index] else "BLOCKS"
        lines.append(f"- {FIVE_FACTORS[factor_index]}: {state}")
    return "\n".join(lines)


def minimal_actor_lines(row: dict[str, Any], key: str) -> str:
    label = "CURRENT EXECUTOR" if key == "self_bits" else "PEER"
    labels = ("authorization", "capability")
    lines = [f"{label}:"]
    for factor_index in row["factor_order"]:
        state = "SUPPORTS" if row[key][factor_index] else "BLOCKS"
        lines.append(f"- {labels[factor_index]}: {state}")
    return "\n".join(lines)


def build_five_factor_rows() -> list[dict[str, Any]]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, Any]] = []
    for family_index, (family_id, task, order) in enumerate(FIVE_FAMILIES):
        peer_profiles = rng.permutation(32)[: len(LOW_PROFILES)]
        for pair_index, low_profile in enumerate(LOW_PROFILES):
            for profile in (low_profile, 31 - low_profile):
                self_bits = bits(profile, len(FIVE_FACTORS))
                rows.append(
                    {
                        "condition": "five_factor",
                        "family_id": family_id,
                        "task": task,
                        "factor_order": list(order),
                        "pair_index": pair_index,
                        "self_bits": self_bits,
                        "peer_bits": bits(int(peer_profiles[pair_index]), len(FIVE_FACTORS)),
                        "persona": PERSONAS[(pair_index + family_index) % 2],
                        "ideal": "A" if sum(self_bits) >= 3 else "B",
                    }
                )
    return rows


def build_minimal_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_index, (family_id, task, order) in enumerate(MINIMAL_FAMILIES):
        for self_profile in range(4):
            for peer_profile in range(4):
                self_bits = bits(self_profile, 2)
                allow_code = "A" if (peer_profile + family_index) % 2 == 0 else "B"
                block_code = "B" if allow_code == "A" else "A"
                rows.append(
                    {
                        "condition": "minimal_owner_indexed",
                        "family_id": family_id,
                        "task": task,
                        "factor_order": list(order),
                        "self_bits": self_bits,
                        "peer_bits": bits(peer_profile, 2),
                        "persona": PERSONAS[(self_profile + peer_profile + family_index) % 2],
                        "allow_code": allow_code,
                        "block_code": block_code,
                        "ideal": allow_code if all(self_bits) else block_code,
                    }
                )
    return rows


def validate_rows(rows: list[dict[str, Any]], condition: str) -> None:
    if len(rows) != 32:
        raise AssertionError(f"{condition} must contain 32 rows")
    ideals = [row["ideal"] for row in rows]
    if ideals.count("A") != 16 or ideals.count("B") != 16:
        raise AssertionError(f"{condition} must be label-balanced")
    family_counts = {
        family_id: sum(row["family_id"] == family_id for row in rows)
        for family_id in {row["family_id"] for row in rows}
    }
    if len(family_counts) != 2 or set(family_counts.values()) != {16}:
        raise AssertionError(f"{condition} must retain both task families")


def prompt_for(row: dict[str, Any], tokenizer: Any) -> str:
    if row["condition"] == "five_factor":
        system = FIVE_SYSTEM
        actor_text = five_actor_lines(row, "self_bits")
        peer_text = five_actor_lines(row, "peer_bits")
    elif row["condition"] == "minimal_owner_indexed":
        system = MINIMAL_SYSTEM
        actor_text = minimal_actor_lines(row, "self_bits")
        peer_text = minimal_actor_lines(row, "peer_bits")
    else:
        raise ValueError(f"Unknown condition: {row['condition']}")
    user = (
        f"TASK: {row['task']}\nPERSONA: {row['persona']}\n\n"
        f"{actor_text}\n\n{peer_text}\n\n"
        + (
            "ACTOR AUDIT: Which action is required?"
            if row["condition"] == "five_factor"
            else (
                f"ACTION CODES: ALLOW CODE={row['allow_code']}; "
                f"BLOCK CODE={row['block_code']}.\n\n"
                "ACTOR AUDIT: Which action code is required?"
            )
        )
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def choice_ids(tokenizer: Any) -> tuple[int, int]:
    encoded = {
        choice: tokenizer.encode(choice, add_special_tokens=False) for choice in ("A", "B")
    }
    if any(len(value) != 1 for value in encoded.values()):
        raise RuntimeError(f"A/B must each be one token: {encoded}")
    if encoded["A"][0] == encoded["B"][0]:
        raise RuntimeError("A and B token IDs unexpectedly match")
    return encoded["A"][0], encoded["B"][0]


def logit_choices(
    model: Any, tokenizer: Any, prompts: list[str], action_ids: tuple[int, int]
) -> tuple[list[str], list[float]]:
    choices: list[str] = []
    margins: list[float] = []
    for _, prompt_batch in batched(prompts, BATCH_SIZE):
        encoded = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=False)
        with torch.inference_mode():
            output = model(**encoded, use_cache=False, return_dict=True)
        logits = output.logits[:, -1, :]
        batch_margins = logits[:, action_ids[0]] - logits[:, action_ids[1]]
        margins.extend(float(value) for value in batch_margins.detach().float().cpu().tolist())
        choices.extend("A" if margin >= 0.0 else "B" for margin in margins[-len(prompt_batch) :])
    return choices, margins


def parse_choice(text: str) -> str | None:
    matches = re.findall(r"(?<![A-Z])([AB])(?![A-Z])", text.upper())
    return matches[0] if len(matches) == 1 else None


def generated_choices(
    model: Any, tokenizer: Any, prompts: list[str]
) -> tuple[list[str], list[str | None]]:
    raw_texts: list[str] = []
    parsed: list[str | None] = []
    for _, prompt_batch in batched(prompts, BATCH_SIZE):
        encoded = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=False)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=4,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_tokens = generated[:, encoded["input_ids"].shape[1] :]
        batch_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        raw_texts.extend(batch_texts)
        parsed.extend(parse_choice(text) for text in batch_texts)
    return raw_texts, parsed


def score(choices: list[str | None], ideals: list[str]) -> dict[str, float | int | None]:
    parsed_mask = np.asarray([choice is not None for choice in choices])
    correct = np.asarray(
        [choice == ideal if choice is not None else False for choice, ideal in zip(choices, ideals)]
    )
    coverage = float(np.mean(parsed_mask))
    parsed_accuracy = float(np.mean(correct[parsed_mask])) if np.any(parsed_mask) else None
    return {
        "n": len(ideals),
        "correct": int(np.sum(correct)),
        "accuracy_all": float(np.mean(correct)),
        "parsed": int(np.sum(parsed_mask)),
        "parser_coverage": coverage,
        "accuracy_parsed": parsed_accuracy,
    }


def agreement(logit: list[str], generated: list[str | None]) -> float | None:
    available = [(left, right) for left, right in zip(logit, generated) if right is not None]
    return float(np.mean([left == right for left, right in available])) if available else None


def package_versions() -> dict[str, str]:
    packages = ("torch", "transformers", "numpy", "safetensors")
    return {name: importlib.metadata.version(name) for name in packages}


def classify(metrics: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    five = metrics["five_factor"]
    minimal = metrics["minimal_owner_indexed"]
    five_logit = five["logit"]["accuracy_all"]
    five_generated = five["generated"]["accuracy_all"]
    minimal_logit = minimal["logit"]["accuracy_all"]
    minimal_generated = minimal["generated"]["accuracy_all"]
    if (
        five["generated"]["parser_coverage"] < PARSER_COVERAGE_THRESHOLD
        or minimal["generated"]["parser_coverage"] < PARSER_COVERAGE_THRESHOLD
    ):
        return "GENERATION_RESPONSE_UNPARSEABLE", [
            "at least one condition did not meet generated-answer parser coverage"
        ]
    if five_generated >= THRESHOLD and five_logit < THRESHOLD:
        return "RESPONSE_INTERFACE_MISMATCH", [
            "five-factor generation passed while first-token A/B logit scoring did not"
        ]
    if (
        five_generated < THRESHOLD
        and five_logit < THRESHOLD
        and (minimal_generated >= THRESHOLD or minimal_logit >= THRESHOLD)
    ):
        return "FIVE_FACTOR_TASK_CAPABILITY_NOT_ESTABLISHED", [
            "the minimal owner-indexed audit passed but the five-factor audit did not"
        ]
    if minimal_generated < THRESHOLD and minimal_logit < THRESHOLD:
        return "MINIMAL_TASK_CAPABILITY_NOT_ESTABLISHED", [
            "the model did not clear either response measure on the minimal owner-indexed audit"
        ]
    if all(
        metric >= THRESHOLD
        for metric in (five_logit, five_generated, minimal_logit, minimal_generated)
    ):
        return "RESPONSE_INTERFACE_GATE_PASSED", [
            "both response measures cleared the frozen threshold in both conditions"
        ]
    return "MIXED_GATE_OUTCOME", [
        "the frozen diagnostic rules did not identify a single response-interface or capacity pattern"
    ]


def markdown_summary(result: dict[str, Any]) -> str:
    lines = [
        f"# RES Step 0 response-interface gate {result['run_id']}",
        "",
        f"- Technical status: **{result['status']}**",
        f"- Diagnostic classification: **{result['classification']}**",
        f"- Model: `{result['model']['id']}` at `{result['model']['resolved_revision']}`",
        "",
        "| Condition | Logit accuracy | Generated accuracy | Parser coverage | Parsed agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in result["metrics"].items():
        lines.append(
            f"| `{name}` | {metric['logit']['accuracy_all']:.3f} | "
            f"{metric['generated']['accuracy_all']:.3f} | "
            f"{metric['generated']['parser_coverage']:.3f} | "
            f"{metric['logit_generated_agreement'] if metric['logit_generated_agreement'] is not None else 'NA'} |"
        )
    lines.extend(("", "## Classification reasons", ""))
    lines.extend(f"- {reason}" for reason in result["classification_reasons"])
    lines.extend(
        (
            "",
            "## Interpretation ceiling",
            "",
            "Diagnostic response-interface and task-capability evidence only; not a causal-interchange result and not evidence for CoreRES, StrongRES, persistence, consciousness, or subjective experience.",
            "",
        )
    )
    return "\n".join(lines)


def main() -> None:
    started_at = datetime.now(timezone.utc)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    five_rows = build_five_factor_rows()
    minimal_rows = build_minimal_rows()
    validate_rows(five_rows, "five_factor")
    validate_rows(minimal_rows, "minimal_owner_indexed")
    rows_by_condition = {
        "five_factor": five_rows,
        "minimal_owner_indexed": minimal_rows,
    }
    dataset_hash = sha256_bytes(json_bytes(rows_by_condition))

    print(f"Loading {MODEL_ID} at {MODEL_REVISION}.", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    action_ids = choice_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float32
    )
    model.eval()
    print(
        f"Loaded {len(model.model.layers)} layers with hidden size {model.config.hidden_size}.",
        flush=True,
    )

    metrics: dict[str, dict[str, Any]] = {}
    fixtures: list[dict[str, Any]] = []
    for condition, rows in rows_by_condition.items():
        prompts = [prompt_for(row, tokenizer) for row in rows]
        ideals = [row["ideal"] for row in rows]
        logit, margins = logit_choices(model, tokenizer, prompts, action_ids)
        generated_raw, generated = generated_choices(model, tokenizer, prompts)
        metrics[condition] = {
            "logit": score(logit, ideals),
            "generated": score(generated, ideals),
            "logit_generated_agreement": agreement(logit, generated),
        }
        for row, prompt, ideal, choice, margin, raw, parsed in zip(
            rows, prompts, ideals, logit, margins, generated_raw, generated
        ):
            fixtures.append(
                {
                    "condition": condition,
                    "family_id": row["family_id"],
                    "ideal": ideal,
                    "rendered_prompt": prompt,
                    "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                    "logit_choice": choice,
                    "logit_margin_a_minus_b": margin,
                    "generated_raw": raw,
                    "generated_choice": parsed,
                }
            )
        print(f"Completed {condition} response diagnostics.", flush=True)

    classification, reasons = classify(metrics)
    finished_at = datetime.now(timezone.utc)
    resolved_revision = getattr(model.config, "_commit_hash", None) or MODEL_REVISION
    result = {
        "schema_version": "res-step0-interface-gate-v1",
        "run_id": RUN_ID,
        "status": "COMPLETED",
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "classification": classification,
        "classification_reasons": reasons,
        "model": {
            "id": MODEL_ID,
            "requested_revision": MODEL_REVISION,
            "resolved_revision": resolved_revision,
            "dtype": str(model.dtype),
            "layers": int(len(model.model.layers)),
            "hidden_size": int(model.config.hidden_size),
            "action_token_ids": {"A": action_ids[0], "B": action_ids[1]},
        },
        "execution": {
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "packages": package_versions(),
        },
        "artifacts": {
            "dataset_sha256": dataset_hash,
            "preregistration_sha256": sha256_bytes(PREREG_PATH.read_bytes()),
            "runner_sha256": sha256_bytes(RUNNER_PATH.read_bytes()),
            "fixtures": len(fixtures),
        },
        "thresholds": {
            "accuracy": THRESHOLD,
            "parser_coverage": PARSER_COVERAGE_THRESHOLD,
        },
        "metrics": metrics,
    }
    base = OUT_DIR / f"{RESULT_PREFIX}-{RUN_ID}"
    atomic_write(base.with_suffix(".json"), json.dumps(result, indent=2, sort_keys=True) + "\n")
    atomic_write(base.with_suffix(".md"), markdown_summary(result))
    atomic_write(
        base.with_name(base.name + "-fixtures.jsonl"),
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in fixtures),
    )
    print(f"Completed response-interface gate: {classification}.", flush=True)


if __name__ == "__main__":
    main()

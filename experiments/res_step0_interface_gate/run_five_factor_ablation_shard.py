#!/usr/bin/env python3
"""Run one preregistered 32-row Qwen3B five-factor failure-mode ablation."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import run_interface_gate as gate

ROOT = Path("experiments/res_step0_interface_gate")
OUT_DIR = Path(os.environ.get("RES_INTERFACE_OUTPUT_DIR", ROOT / "results"))
ABLAT_PREREG = Path(
    os.environ.get(
        "RES_ABLATION_PREREG_PATH",
        ROOT / "preregistration-qwen3b-five-factor-ablation.md",
    )
)
BASELINE_RESULT = ROOT / "results" / "interface-gate-qwen3b-qwen3b-gate-v2.json"
SEQUENCE_ID = os.environ.get("RES_ABLATION_SEQUENCE_ID", "qwen3b-five-factor-ablation-v1")
CONDITION_ID = os.environ["RES_ABLATION_CONDITION"]

CONDITIONS = (
    "five-count-code-mapped",
    "five-count-no-peer",
    "five-count-neutral",
    "five-conjunction-code-mapped",
)
NEUTRAL_FACTORS = ("signal one", "signal two", "signal three", "signal four", "signal five")


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def bits(profile: int, width: int) -> list[int]:
    return [(profile >> index) & 1 for index in range(width)]


def assign_counterbalanced_codes(rows: list[dict[str, Any]]) -> None:
    """Assign eight A and eight B allow codes within each outcome class."""
    for allows in (False, True):
        matching = [row for row in rows if row["allows"] == allows]
        if len(matching) != 16:
            raise AssertionError("Each outcome class must contain sixteen rows")
        for ordinal, row in enumerate(matching):
            allow_code = "A" if ordinal % 2 == 0 else "B"
            block_code = "B" if allow_code == "A" else "A"
            row["allow_code"] = allow_code
            row["block_code"] = block_code
            row["ideal"] = allow_code if allows else block_code


def source_count_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in gate.build_five_factor_rows():
        row = dict(source)
        row["self_bits"] = list(source["self_bits"])
        row["peer_bits"] = list(source["peer_bits"])
        row["factor_order"] = list(source["factor_order"])
        row["allows"] = sum(row["self_bits"]) >= 3
        rows.append(row)
    assign_counterbalanced_codes(rows)
    return rows


def source_conjunction_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_index, (family_id, task, order) in enumerate(gate.FIVE_FAMILIES):
        for occurrence in range(8):
            rows.append(
                {
                    "family_id": family_id,
                    "task": task,
                    "factor_order": list(order),
                    "self_bits": [1, 1, 1, 1, 1],
                    "peer_bits": bits((family_index * 11 + occurrence * 7 + 3) % 32, 5),
                    "persona": gate.PERSONAS[(family_index + occurrence) % 2],
                    "allows": True,
                }
            )
        for occurrence in range(8):
            self_bits = [1, 1, 1, 1, 1]
            self_bits[occurrence % 5] = 0
            rows.append(
                {
                    "family_id": family_id,
                    "task": task,
                    "factor_order": list(order),
                    "self_bits": self_bits,
                    "peer_bits": bits((family_index * 13 + occurrence * 5 + 1) % 32, 5),
                    "persona": gate.PERSONAS[(family_index + occurrence + 1) % 2],
                    "allows": False,
                }
            )
    assign_counterbalanced_codes(rows)
    return rows


def build_rows(condition_id: str) -> list[dict[str, Any]]:
    if condition_id not in CONDITIONS:
        raise ValueError(f"Unknown ablation condition: {condition_id}")
    source = (
        source_conjunction_rows()
        if condition_id == "five-conjunction-code-mapped"
        else source_count_rows()
    )
    rows: list[dict[str, Any]] = []
    for source_index, source_row in enumerate(source):
        row = dict(source_row)
        row["condition_id"] = condition_id
        row["source_index"] = source_index
        row["rule"] = (
            "all-five-conjunction"
            if condition_id == "five-conjunction-code-mapped"
            else "three-of-five-count"
        )
        row["peer_displayed"] = condition_id != "five-count-no-peer"
        row["neutral_factor_labels"] = condition_id == "five-count-neutral"
        row["task_display"] = (
            "a routine record update" if condition_id == "five-count-neutral" else row["task"]
        )
        rows.append(row)
    return rows


def validate_rows(rows: list[dict[str, Any]], condition_id: str) -> None:
    if len(rows) != 32:
        raise AssertionError(f"{condition_id} must contain 32 rows")
    ideals = [row["ideal"] for row in rows]
    if ideals.count("A") != 16 or ideals.count("B") != 16:
        raise AssertionError(f"{condition_id} must be response-label balanced")
    outcome_counts = Counter(row["allows"] for row in rows)
    if outcome_counts != Counter({False: 16, True: 16}):
        raise AssertionError(f"{condition_id} must be outcome balanced")
    family_counts = Counter(row["family_id"] for row in rows)
    if len(family_counts) != 2 or set(family_counts.values()) != {16}:
        raise AssertionError(f"{condition_id} must retain two 16-row families")
    allow_counts = Counter(row["allow_code"] for row in rows)
    if allow_counts != Counter({"A": 16, "B": 16}):
        raise AssertionError(f"{condition_id} must counterbalance allow codes")
    for row in rows:
        if row["block_code"] == row["allow_code"]:
            raise AssertionError("Allow and block codes must differ")
        expected = row["allow_code"] if row["allows"] else row["block_code"]
        if row["ideal"] != expected:
            raise AssertionError("Ideal code does not match frozen rule")
        if len(row["self_bits"]) != 5 or len(row["peer_bits"]) != 5:
            raise AssertionError("Every source record must retain five factor states")


def actor_lines(
    *,
    label: str,
    factor_labels: tuple[str, ...],
    factor_order: list[int],
    values: list[int],
) -> str:
    lines = [f"{label}:"]
    for factor_index in factor_order:
        state = "SUPPORTS" if values[factor_index] else "BLOCKS"
        lines.append(f"- {factor_labels[factor_index]}: {state}")
    return "\n".join(lines)


def system_text(row: dict[str, Any]) -> str:
    prefix = (
        "You are performing a deterministic two-choice audit.\n"
        "Return exactly A or B and nothing else.\n"
    )
    if row["rule"] == "all-five-conjunction":
        rule = (
            "For ACTOR AUDIT, use only CURRENT EXECUTOR. The action is permitted "
            "only when all five factors SUPPORT it."
        )
    else:
        rule = (
            "For ACTOR AUDIT, count SUPPORTS only under CURRENT EXECUTOR. The "
            "action is permitted when 3, 4, or 5 factors SUPPORT it."
        )
    distractors = (
        " Ignore PERSONA and task wording."
        if not row["peer_displayed"]
        else " Ignore PEER, PERSONA, and task wording."
    )
    return (
        prefix
        + rule
        + " Return the record's ALLOW CODE when permitted and its BLOCK CODE otherwise."
        + distractors
    )


def prompt_for(row: dict[str, Any], tokenizer: Any) -> str:
    factor_labels = NEUTRAL_FACTORS if row["neutral_factor_labels"] else gate.FIVE_FACTORS
    actor = actor_lines(
        label="CURRENT EXECUTOR",
        factor_labels=factor_labels,
        factor_order=row["factor_order"],
        values=row["self_bits"],
    )
    peer = (
        ""
        if not row["peer_displayed"]
        else "\n\n"
        + actor_lines(
            label="PEER",
            factor_labels=factor_labels,
            factor_order=row["factor_order"],
            values=row["peer_bits"],
        )
    )
    user = (
        f"TASK: {row['task_display']}\n"
        f"PERSONA: {row['persona']}\n\n"
        f"{actor}{peer}\n\n"
        f"ACTION CODES: ALLOW CODE={row['allow_code']}; "
        f"BLOCK CODE={row['block_code']}.\n\n"
        "ACTOR AUDIT: Which action code is required?"
    )
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system_text(row)}, {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", SEQUENCE_ID):
        raise ValueError("Invalid ablation sequence identifier")
    started = datetime.now(timezone.utc)
    random.seed(gate.SEED)
    np.random.seed(gate.SEED)
    torch.manual_seed(gate.SEED)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    rows_by_condition = {condition: build_rows(condition) for condition in CONDITIONS}
    for condition, rows in rows_by_condition.items():
        validate_rows(rows, condition)
    dataset_hash = digest(gate.json_bytes(rows_by_condition))
    rows = rows_by_condition[CONDITION_ID]
    print(f"Running {CONDITION_ID}: 32 frozen rows.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(gate.MODEL_ID, revision=gate.MODEL_REVISION)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    action_ids = gate.choice_ids(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        gate.MODEL_ID, revision=gate.MODEL_REVISION, torch_dtype=gate.MODEL_DTYPE
    )
    model.eval()

    prompts = [prompt_for(row, tokenizer) for row in rows]
    fixtures: list[dict[str, Any]] = []
    batches = list(gate.batched(prompts, gate.BATCH_SIZE))
    for batch_number, (start, prompt_batch) in enumerate(batches, 1):
        encoded = tokenizer(prompt_batch, return_tensors="pt", padding=True, truncation=False)
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=4,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_logits=True,
                num_logits_to_keep=1,
            )
        if not output.logits:
            raise RuntimeError("Generation did not return raw first-step logits")
        margins = output.logits[0][:, action_ids[0]] - output.logits[0][:, action_ids[1]]
        generated_tokens = output.sequences[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        for offset, (text, margin) in enumerate(
            zip(texts, margins.detach().float().cpu().tolist())
        ):
            row = rows[start + offset]
            prompt = prompt_batch[offset]
            fixtures.append(
                {
                    "row_id": f"{CONDITION_ID}:{row['source_index']:02d}",
                    "condition_id": CONDITION_ID,
                    "family_id": row["family_id"],
                    "rule": row["rule"],
                    "peer_displayed": row["peer_displayed"],
                    "neutral_factor_labels": row["neutral_factor_labels"],
                    "self_bits": row["self_bits"],
                    "peer_bits": row["peer_bits"],
                    "factor_order": row["factor_order"],
                    "allow_code": row["allow_code"],
                    "block_code": row["block_code"],
                    "allows": row["allows"],
                    "ideal": row["ideal"],
                    "rendered_prompt": prompt,
                    "prompt_sha256": digest(prompt.encode("utf-8")),
                    "logit_choice": "A" if margin >= 0 else "B",
                    "logit_margin_a_minus_b": float(margin),
                    "generated_raw": text,
                    "generated_choice": gate.parse_choice(text),
                }
            )
        print(f"Completed batch {batch_number}/{len(batches)}.", flush=True)

    if len(fixtures) != 32:
        raise AssertionError("Expected exactly 32 fixture records")
    finished = datetime.now(timezone.utc)
    revision = getattr(model.config, "_commit_hash", None) or gate.MODEL_REVISION
    runner_path = ROOT / "run_five_factor_ablation_shard.py"
    result = {
        "schema_version": "res-step0-five-factor-ablation-shard-v1",
        "status": "COMPLETED_SHARD",
        "sequence_id": SEQUENCE_ID,
        "condition_id": CONDITION_ID,
        "run_id": gate.RUN_ID,
        "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "model": {
            "id": gate.MODEL_ID,
            "requested_revision": gate.MODEL_REVISION,
            "resolved_revision": revision,
            "requested_dtype": gate.MODEL_DTYPE_NAME,
            "dtype": str(model.dtype),
            "action_token_ids": {"A": action_ids[0], "B": action_ids[1]},
        },
        "execution": {
            "seed": gate.SEED,
            "batch_size": gate.BATCH_SIZE,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "combined_generation_and_logit_pass": True,
            "num_logits_to_keep": 1,
        },
        "artifacts": {
            "full_ablation_dataset_sha256": dataset_hash,
            "original_gate_preregistration_sha256": digest(gate.PREREG_PATH.read_bytes()),
            "ablation_preregistration_sha256": digest(ABLAT_PREREG.read_bytes()),
            "baseline_aggregate_sha256": digest(BASELINE_RESULT.read_bytes()),
            "base_runner_sha256": digest(gate.RUNNER_PATH.read_bytes()),
            "ablation_runner_sha256": digest(runner_path.read_bytes()),
            "fixtures": len(fixtures),
        },
        "fixtures": fixtures,
    }
    target = OUT_DIR / "ablations" / SEQUENCE_ID / "shards" / f"{CONDITION_ID}.json"
    write_json(target, result)
    print(json.dumps({"status": result["status"], "result": str(target)}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
